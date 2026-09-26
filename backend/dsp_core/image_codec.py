import io
import math
import numpy as np
from PIL import Image

class ImageCodecError(ValueError):
    pass

MAX_IMAGE_DIMENSION = 1024
MAX_CAPACITY_BYTES = MAX_IMAGE_DIMENSION ** 2

def encode_bytes_to_png(packet_bytes: bytes) -> bytes:
    """
    Encodes a byte packet into a grayscale PNG image.
    Uses exactly 1 pixel per byte.
    Pads the remaining pixels in the square image with 0.
    """
    total_bytes = len(packet_bytes)
    
    if total_bytes > MAX_CAPACITY_BYTES:
        raise ImageCodecError(f"Payload of {total_bytes} bytes exceeds maximum capacity of {MAX_CAPACITY_BYTES} bytes.")
        
    L = math.ceil(math.sqrt(total_bytes))
    
    # Pre-allocate flat array for the image
    flat_pixels = np.zeros(L * L, dtype=np.uint8)
    
    # Copy bytes into pixels
    # Since packet_bytes is bytes, we can map directly to uint8
    byte_array = np.frombuffer(packet_bytes, dtype=np.uint8)
    flat_pixels[:total_bytes] = byte_array
    
    # Reshape to 2D
    pixel_matrix = flat_pixels.reshape((L, L))
    
    # Encode to PNG
    img = Image.fromarray(pixel_matrix, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    
    return buf.getvalue()

def decode_png_to_bytes(png_bytes: bytes) -> bytes:
    """
    Decodes a grayscale PNG image back into bytes.
    The exact extraction of the N payload bytes is handled by the protocol layer.
    This function simply flattens the image into bytes.
    """
    buf = io.BytesIO(png_bytes)
    try:
        img = Image.open(buf)
        if img.mode != 'L':
            img = img.convert('L')
    except Exception as e:
        raise ImageCodecError(f"Failed to read PNG image: {e}")
        
    pixel_matrix = np.array(img, dtype=np.uint8)
    flat_pixels = pixel_matrix.flatten()
    
    return flat_pixels.tobytes()
