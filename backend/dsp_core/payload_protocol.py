import struct
import zlib

class ProtocolError(ValueError):
    pass

MAGIC_BYTES = b"RGL1"
PAYLOAD_TYPE_FLAC = 0x01
HEADER_SIZE = 17

def pack_payload(flac_payload: bytes) -> bytes:
    """
    Pack FLAC payload into RGL1 binary protocol packet.
    """
    payload_len = len(flac_payload)
    
    # Pack bytes 0..8 (Magic, Type, Length)
    header_start = struct.pack("<4sBI", MAGIC_BYTES, PAYLOAD_TYPE_FLAC, payload_len)
    
    # Calculate Header CRC32 (on bytes 0..8)
    header_crc = zlib.crc32(header_start) & 0xFFFFFFFF
    
    # Calculate Payload CRC32
    payload_crc = zlib.crc32(flac_payload) & 0xFFFFFFFF
    
    # Full header
    header = header_start + struct.pack("<II", header_crc, payload_crc)
    
    return header + flac_payload

def unpack_payload(packet: bytes) -> bytes:
    """
    Unpack RGL1 binary protocol packet, validating CRCs and extracting the FLAC payload.
    """
    if len(packet) < HEADER_SIZE:
        raise ProtocolError("Packet is too small to contain a valid header.")
        
    magic, payload_type, payload_len = struct.unpack("<4sBI", packet[0:9])
    
    if magic != MAGIC_BYTES:
        raise ProtocolError("Invalid magic bytes. Not a Rigel encoded image.")
        
    if payload_type != PAYLOAD_TYPE_FLAC:
        raise ProtocolError(f"Unsupported payload type: {payload_type}")
        
    header_crc_expected = zlib.crc32(packet[0:9]) & 0xFFFFFFFF
    header_crc_actual, payload_crc_expected = struct.unpack("<II", packet[9:17])
    
    if header_crc_actual != header_crc_expected:
        raise ProtocolError("Header checksum validation failed. Header is corrupted.")
        
    if len(packet) < HEADER_SIZE + payload_len:
        raise ProtocolError(f"Packet is truncated. Expected {payload_len} payload bytes, got {len(packet) - HEADER_SIZE}.")
        
    payload = packet[HEADER_SIZE : HEADER_SIZE + payload_len]
    
    payload_crc_actual = zlib.crc32(payload) & 0xFFFFFFFF
    if payload_crc_actual != payload_crc_expected:
        raise ProtocolError("Payload checksum validation failed. Payload is corrupted.")
        
    return payload
