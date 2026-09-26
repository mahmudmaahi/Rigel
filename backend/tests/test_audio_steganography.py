import io
import struct
import zlib
import numpy as np
import pytest
from PIL import Image

from dsp_core.audio_codec import encode_pcm16_to_flac, decode_flac_to_pcm16, AudioCodecError
from dsp_core.payload_protocol import pack_payload, unpack_payload, ProtocolError, MAGIC_BYTES, PAYLOAD_TYPE_FLAC
from dsp_core.image_codec import encode_bytes_to_png, decode_png_to_bytes, ImageCodecError, MAX_CAPACITY_BYTES
from app.services.audio_image_service import encode_audio_to_image, decode_image_to_audio

def create_pcm16_sine(freq=440, duration=0.5, sr=16000, channels=1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = (32767 * 0.5 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    if channels == 2:
        return np.column_stack((samples, samples))
    return samples

# --- A. AUDIO CODEC ---
def test_audio_codec_mono():
    original = create_pcm16_sine(channels=1)
    flac = encode_pcm16_to_flac(original, 16000)
    decoded, sr = decode_flac_to_pcm16(flac)
    assert sr == 16000
    assert np.array_equal(original, decoded)

def test_audio_codec_stereo():
    original = create_pcm16_sine(channels=2)
    flac = encode_pcm16_to_flac(original, 16000)
    decoded, sr = decode_flac_to_pcm16(flac)
    assert sr == 16000
    assert np.array_equal(original, decoded)

def test_audio_codec_silence():
    original = np.zeros(16000, dtype=np.int16)
    flac = encode_pcm16_to_flac(original, 16000)
    decoded, _ = decode_flac_to_pcm16(flac)
    assert np.array_equal(original, decoded)
    
def test_audio_codec_short():
    original = np.zeros(10, dtype=np.int16)
    flac = encode_pcm16_to_flac(original, 16000)
    decoded, _ = decode_flac_to_pcm16(flac)
    assert np.array_equal(original, decoded)

def test_audio_codec_invalid_type():
    original = np.zeros(16000, dtype=np.float32)
    with pytest.raises(AudioCodecError):
        encode_pcm16_to_flac(original, 16000)

# --- B. PROTOCOL ---
def test_protocol_valid():
    payload = b"dummy_flac_data"
    packet = pack_payload(payload)
    unpacked = unpack_payload(packet)
    assert payload == unpacked

def test_protocol_invalid_magic():
    packet = pack_payload(b"test")
    corrupted = b"BAD0" + packet[4:]
    with pytest.raises(ProtocolError, match="Invalid magic bytes"):
        unpack_payload(corrupted)

def test_protocol_invalid_type():
    packet = pack_payload(b"test")
    corrupted = packet[:4] + b"\x02" + packet[5:]
    # Fix header crc to pass the check and reach type check... wait, type check is before CRC check in our code!
    with pytest.raises(ProtocolError, match="Unsupported payload type"):
        unpack_payload(corrupted)

def test_protocol_corrupted_header_crc():
    packet = bytearray(pack_payload(b"test"))
    packet[5] = (packet[5] + 1) % 256 # modify length byte
    with pytest.raises(ProtocolError, match="Header checksum validation failed"):
        unpack_payload(bytes(packet))

def test_protocol_corrupted_payload_crc():
    packet = bytearray(pack_payload(b"test"))
    packet[-1] = (packet[-1] + 1) % 256
    with pytest.raises(ProtocolError, match="Payload checksum validation failed"):
        unpack_payload(bytes(packet))

def test_protocol_truncated():
    packet = pack_payload(b"1234567890")
    with pytest.raises(ProtocolError, match="Packet is truncated"):
        unpack_payload(packet[:-2])

# --- C. IMAGE CODEC ---
def test_image_codec_exactness():
    packet = pack_payload(b"hello world")
    png_bytes = encode_bytes_to_png(packet)
    recovered_bytes = decode_png_to_bytes(png_bytes)
    # The image might be larger because it's a square
    assert recovered_bytes[:len(packet)] == packet

# --- D. CAPACITY ---
def test_capacity_limits():
    # Exactly fits
    max_payload = MAX_CAPACITY_BYTES - 17
    # We can't actually allocate a 1MB bytestring for testing without some overhead but it's fine
    packet = pack_payload(b"0" * max_payload)
    assert len(packet) == MAX_CAPACITY_BYTES
    png_bytes = encode_bytes_to_png(packet)
    assert png_bytes # Should pass
    
    # Exceeds
    packet_overflow = pack_payload(b"0" * (max_payload + 1))
    with pytest.raises(ImageCodecError):
        encode_bytes_to_png(packet_overflow)

# --- E. DETERMINISM ---
def test_determinism():
    original = create_pcm16_sine()
    res1 = encode_audio_to_image(original, 16000)
    res2 = encode_audio_to_image(original, 16000)
    assert res1.png_bytes == res2.png_bytes

# --- F. END-TO-END ---
def test_end_to_end_mono():
    original = create_pcm16_sine()
    res_enc = encode_audio_to_image(original, 16000)
    res_dec = decode_image_to_audio(res_enc.png_bytes)
    assert np.array_equal(original, res_dec.samples)
    assert res_dec.sample_rate_hz == 16000
    
def test_end_to_end_stereo():
    original = create_pcm16_sine(channels=2)
    res_enc = encode_audio_to_image(original, 16000)
    res_dec = decode_image_to_audio(res_enc.png_bytes)
    assert np.array_equal(original, res_dec.samples)
    assert res_dec.sample_rate_hz == 16000
