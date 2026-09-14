"""
app/services/filter_service.py — Module 06: Audio Filtering
=============================================================

FastAPI service layer for the audio filtering feature.

Responsibilities:
    - Validate the UploadFile (reusing existing _read_and_validate_upload)
    - Load audio via the existing WAV decoder pipeline
    - Convert samples to float32 (no peak-normalisation)
    - Delegate filter design and application to dsp_core.filtering
    - Encode the filtered result to WAV bytes via scipy.io.wavfile
    - Return a StreamingResponse for download

DSP execution is strictly in dsp_core.filtering. This service handles
only HTTP orchestration.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from scipy.io import wavfile

from app.models.audio import (
    AudioFilterRequest,
    AudioFilterResponse,
    AudioFilterResponseRequest,
)
from app.services.audio_service import _read_and_validate_upload
from dsp_core.audio_loader import AudioDecodeError, load_audio_bytes
from dsp_core.filtering import (
    FilterDesignError,
    FilterFrequencyResponse,
    FilterSpec,
    apply_filter,
    compute_frequency_response,
    design_filter,
)


def _to_float32_normalised(samples: np.ndarray) -> np.ndarray:
    """Convert raw audio samples to float32 in the range [-1.0, 1.0].

    Integer dtypes are scaled by their full-scale value.
    Float dtypes are cast directly (assumed to already be in [-1, 1]).
    No peak-normalisation of the result is performed.
    """
    dtype = samples.dtype
    if np.issubdtype(dtype, np.integer):
        max_val = float(np.iinfo(dtype).max)
        return samples.astype(np.float32) / max_val
    return samples.astype(np.float32)


def _encode_to_wav(samples: np.ndarray, sample_rate_hz: int) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate_hz, samples)
    return buf.getvalue()


def _build_spec(loaded_sample_rate: int, request: AudioFilterRequest) -> FilterSpec:
    """Build a FilterSpec from an AudioFilterRequest and actual sample rate."""
    return FilterSpec(
        filter_type=request.filter_type,
        family=request.family,
        order=request.order,
        sample_rate_hz=loaded_sample_rate,
        cutoff_hz=request.cutoff_hz,
        low_hz=request.low_hz,
        high_hz=request.high_hz,
        ripple_db=request.ripple_db,
        attenuation_db=request.attenuation_db,
        fir_window=request.fir_window,
        transition_bandwidth_hz=request.transition_bandwidth_hz,
        center_hz=request.center_hz,
        gain_db=request.gain_db,
        q_factor=request.q_factor,
    )


async def apply_filter_to_upload(
    file: UploadFile | None,
    request: AudioFilterRequest,
) -> StreamingResponse:
    """Load an uploaded WAV, apply the filter, return processed WAV.

    Pipeline:
        WAV bytes → load_audio_bytes → float32 conversion
        → design_filter → apply_filter → WAV encode → StreamingResponse
    """
    file_bytes, filename, _ = await _read_and_validate_upload(file)

    try:
        loaded_audio = load_audio_bytes(file_bytes=file_bytes, filename=filename)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    samples_f32 = _to_float32_normalised(loaded_audio.samples)
    spec = _build_spec(loaded_audio.sample_rate_hz, request)

    try:
        design = design_filter(spec)
    except FilterDesignError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    try:
        filtered = apply_filter(design, samples_f32)
    except FilterDesignError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    wav_bytes = _encode_to_wav(filtered, loaded_audio.sample_rate_hz)

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=filtered_audio.wav"},
    )


async def get_filter_frequency_response(
    request: AudioFilterResponseRequest,
) -> AudioFilterResponse:
    """Design a filter and return its theoretical frequency response.

    No audio file required. Returns the frequency axis and magnitude in dB
    so the frontend can visualise passband/stopband/cutoff before applying.
    """
    spec = FilterSpec(
        filter_type=request.filter_type,
        family=request.family,
        order=request.order,
        sample_rate_hz=request.sample_rate_hz,
        cutoff_hz=request.cutoff_hz,
        low_hz=request.low_hz,
        high_hz=request.high_hz,
        ripple_db=request.ripple_db,
        attenuation_db=request.attenuation_db,
        fir_window=request.fir_window,
        transition_bandwidth_hz=request.transition_bandwidth_hz,
        center_hz=request.center_hz,
        gain_db=request.gain_db,
        q_factor=request.q_factor,
    )

    try:
        design = design_filter(spec)
    except FilterDesignError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    freq_response: FilterFrequencyResponse = compute_frequency_response(design)

    return AudioFilterResponse(
        filter_type=freq_response.filter_type,
        family=freq_response.family,
        order=freq_response.order,
        sample_rate_hz=request.sample_rate_hz,
        cutoff_hz=freq_response.cutoff_hz,
        low_hz=freq_response.low_hz,
        high_hz=freq_response.high_hz,
        center_hz=freq_response.center_hz,
        frequencies_hz=freq_response.frequencies_hz,
        magnitude_db=freq_response.magnitude_db,
    )
