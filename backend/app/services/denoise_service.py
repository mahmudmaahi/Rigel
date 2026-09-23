"""
app/services/denoise_service.py — Module 07: Audio Denoising
=============================================================

FastAPI service layer for the audio denoising feature.

Responsibility:
    - Validate UploadFile
    - Load audio via WAV decoder
    - Convert to float32 normalized
    - Route to the correct dsp_core.denoise method
    - Encode back to WAV
    - Return StreamingResponse
"""

from __future__ import annotations

import io
import numpy as np
from fastapi import HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from scipy.io import wavfile

from app.models.denoise import AudioDenoiseRequest
from app.services.audio_service import _read_and_validate_upload
from dsp_core.audio_loader import AudioDecodeError, load_audio_bytes

from dsp_core.denoise import (
    denoise_spectral_subtraction,
    denoise_wiener_dd,
    denoise_logmmse,
    denoise_omlsa,
)

def _to_float64_normalised(samples: np.ndarray) -> np.ndarray:
    """Convert raw audio samples to float64 in the range [-1.0, 1.0]."""
    dtype = samples.dtype
    if np.issubdtype(dtype, np.integer):
        max_val = float(np.iinfo(dtype).max)
        return samples.astype(np.float64) / max_val
    return samples.astype(np.float64)


def _encode_to_wav(samples: np.ndarray, sample_rate_hz: int) -> bytes:
    # Ensure float32 for export
    samples_f32 = samples.astype(np.float32)
    # Clip strictly to [-1, 1] to prevent WAV export wrap-around
    samples_f32 = np.clip(samples_f32, -1.0, 1.0)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate_hz, samples_f32)
    return buf.getvalue()


async def apply_denoise_to_upload(
    file: UploadFile | None,
    request: AudioDenoiseRequest,
) -> StreamingResponse:
    file_bytes, filename, _ = await _read_and_validate_upload(file)

    try:
        loaded_audio = load_audio_bytes(file_bytes=file_bytes, filename=filename)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    signal = loaded_audio.samples
    sr = loaded_audio.sample_rate_hz

    try:
        if request.method == "spectral_subtraction":
            processed = denoise_spectral_subtraction(
                signal=signal,
                sample_rate_hz=sr,
                alpha=request.alpha,
                beta=request.beta,
                noise_alpha_s=request.noise_alpha_s,
                noise_bias=request.noise_bias
            )
        elif request.method == "wiener":
            processed = denoise_wiener_dd(
                signal=signal,
                sample_rate_hz=sr,
                alpha_dd=request.alpha_dd,
                noise_alpha_s=request.noise_alpha_s,
                noise_bias=request.noise_bias
            )
        elif request.method == "logmmse":
            processed = denoise_logmmse(
                signal=signal,
                sample_rate_hz=sr,
                alpha_dd=request.alpha_dd,
                noise_alpha_s=request.noise_alpha_s,
                noise_bias=request.noise_bias
            )
        elif request.method == "imcra":
            processed = denoise_omlsa(
                signal=signal,
                sample_rate_hz=sr,
                alpha_dd=request.alpha_dd,
                G_min=request.g_min,
                imcra_alpha_s=request.imcra_alpha_s,
                imcra_alpha_d=request.imcra_alpha_d
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail=f"Unknown denoising method: {request.method}"
            )
    except Exception as exc:
        # Catch unexpected DSP errors (e.g., ValueError from bad parameters)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    wav_bytes = _encode_to_wav(processed, sr)

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=denoised_audio.wav"},
    )
