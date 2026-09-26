import io
import soundfile as sf
import numpy as np

class AudioCodecError(ValueError):
    pass

def encode_pcm16_to_flac(samples: np.ndarray, sample_rate: int) -> bytes:
    """
    Encode PCM16 audio samples to FLAC in memory.
    Ensures that the input is np.int16.
    """
    if samples.dtype != np.int16:
        raise AudioCodecError(f"Expected int16 samples, got {samples.dtype}")
    
    if samples.ndim not in (1, 2):
        raise AudioCodecError(f"Expected 1 or 2 channels, got {samples.ndim} dimensions")
        
    buf = io.BytesIO()
    try:
        sf.write(buf, samples, sample_rate, format='FLAC', subtype='PCM_16')
    except Exception as e:
        raise AudioCodecError(f"Failed to encode FLAC: {e}")
        
    return buf.getvalue()

def decode_flac_to_pcm16(flac_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode FLAC bytes back to PCM16 audio samples.
    Returns (samples, sample_rate).
    """
    buf = io.BytesIO(flac_bytes)
    try:
        samples, sample_rate = sf.read(buf, dtype='int16', always_2d=False)
    except Exception as e:
        raise AudioCodecError(f"Failed to decode FLAC: {e}")
        
    return samples, sample_rate
