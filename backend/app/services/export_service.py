import io
import numpy as np
from scipy.io import wavfile
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

def export_samples_to_wav(samples: list[list[float]], sample_rate_hz: int) -> StreamingResponse:
    """
    Exports floating point samples to a WAV file stream.
    samples: A list of channels, where each channel is a list of floats.
    """
    if not samples:
        raise HTTPException(status_code=400, detail="No audio samples provided.")
    
    if sample_rate_hz <= 0:
        raise HTTPException(status_code=400, detail="Invalid sample rate.")

    n_channels = len(samples)
    if n_channels == 0:
        raise HTTPException(status_code=400, detail="Empty channels.")
    
    n_samples = len(samples[0])
    if n_samples == 0:
        raise HTTPException(status_code=400, detail="Channels contain no samples.")

    # Validate that all channels have the same length
    for ch in samples:
        if len(ch) != n_samples:
            raise HTTPException(status_code=400, detail="All channels must have the same number of samples.")

    # Convert to numpy array of shape [n_samples, n_channels] or [n_samples] for mono
    np_samples = np.array(samples, dtype=np.float32).T

    buffer = io.BytesIO()
    try:
        wavfile.write(buffer, sample_rate_hz, np_samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to encode WAV file.") from exc
    
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=processed_audio.wav"}
    )
