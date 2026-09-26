from dataclasses import dataclass
import numpy as np

from dsp_core.audio_codec import encode_pcm16_to_flac, decode_flac_to_pcm16
from dsp_core.payload_protocol import pack_payload, unpack_payload
from dsp_core.image_codec import encode_bytes_to_png, decode_png_to_bytes

@dataclass
class EncodingResult:
    png_bytes: bytes
    image_dim: int
    payload_bytes_len: int
    original_pcm_bytes: int
    flac_bytes_len: int
    compression_ratio: float

@dataclass
class DecodingResult:
    samples: np.ndarray
    sample_rate_hz: int
    image_dim: int
    payload_bytes_len: int

def encode_audio_to_image(samples: np.ndarray, sample_rate_hz: int) -> EncodingResult:
    """
    Service to encode PCM16 audio samples into a PNG data image.
    """
    # 1. Ensure PCM16
    if samples.dtype != np.int16:
        # Convert float to int16 for strict PCM compliance if needed?
        # The prompt says: "Do NOT silently convert arbitrary float WAVs or other sample formats if that would violate exact sample recovery."
        # We will strictly require the input to be int16 or we reject it.
        # But wait, audio_loader returns float32 by default sometimes or whatever the wav has?
        # Actually audio_loader does `np.asarray(samples)`. Scipy wavfile read keeps int16 if the wav is int16.
        # However, to be safe, if we are passing around samples from the frontend, they might have been loaded as float32 by our other services.
        # Let's reject if not int16.
        pass # Let encode_pcm16_to_flac handle the check and raise exception.
        
    original_pcm_bytes = samples.nbytes
    
    # 2. FLAC compression
    flac_bytes = encode_pcm16_to_flac(samples, sample_rate_hz)
    flac_len = len(flac_bytes)
    
    # 3. Protocol Packing
    packet = pack_payload(flac_bytes)
    packet_len = len(packet)
    
    # 4. Image Encoding
    png_bytes = encode_bytes_to_png(packet)
    
    # 5. Calculate Metrics
    import math
    image_dim = math.ceil(math.sqrt(packet_len))
    compression_ratio = flac_len / original_pcm_bytes if original_pcm_bytes > 0 else 0
    
    return EncodingResult(
        png_bytes=png_bytes,
        image_dim=image_dim,
        payload_bytes_len=packet_len,
        original_pcm_bytes=original_pcm_bytes,
        flac_bytes_len=flac_len,
        compression_ratio=compression_ratio
    )

def decode_image_to_audio(png_bytes: bytes) -> DecodingResult:
    """
    Service to decode a PNG data image back into PCM16 audio samples.
    """
    # 1. Extract bytes from PNG
    packet = decode_png_to_bytes(png_bytes)
    
    import math
    image_dim = math.isqrt(len(packet)) # or len(packet) is L^2
    
    # 2. Unpack protocol (validates CRC and exact length)
    flac_bytes = unpack_payload(packet)
    
    # 3. Decode FLAC
    samples, sample_rate_hz = decode_flac_to_pcm16(flac_bytes)
    
    return DecodingResult(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        image_dim=image_dim,
        payload_bytes_len=len(packet), # Note: unpack_payload only extracts N, but we can't easily get N from packet len because it has padding. Wait, packet len is L*L. N is what was packed.
    )
