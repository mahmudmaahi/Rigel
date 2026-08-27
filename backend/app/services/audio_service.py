from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models.audio import (
    AudioAnalyzeResponse,
    AudioMetadata,
    AudioSampleSummary,
    AudioUploadResponse,
    ChannelWaveform,
    WaveformBin,
    WaveformData,
)
from dsp_core.audio_loader import AudioDecodeError, load_audio_bytes
from dsp_core.waveform import (
    DEFAULT_N_BINS,
    ChannelWaveform as DspChannelWaveform,
    WaveformData as DspWaveformData,
    compute_waveform,
)

SUPPORTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/vnd.wave",
    "application/octet-stream",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _read_and_validate_upload(file: UploadFile | None) -> tuple[bytes, str, str]:
    """Validate the uploaded file and return (file_bytes, filename, content_type).

    Raises :class:`fastapi.HTTPException` for all validation failures.
    """
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Select a WAV audio file before uploading.",
        )

    filename = file.filename or ""
    content_type = file.content_type or "application/octet-stream"

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file must have a filename.",
        )

    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only WAV audio files are supported right now.",
        )

    file_bytes = await file.read()
    max_bytes = get_settings().max_audio_upload_bytes

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty.",
        )

    if len(file_bytes) > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"The uploaded audio file is too large. The current limit is {max_mb} MB.",
        )

    return file_bytes, filename, content_type


def _build_metadata(filename: str, content_type: str, loaded_audio) -> AudioMetadata:
    """Construct an AudioMetadata object from a LoadedAudio result."""
    import numpy as np

    samples = loaded_audio.samples
    channels = 1 if samples.ndim == 1 else int(samples.shape[1])
    samples_per_channel = int(samples.shape[0])
    total_samples = int(samples.size)
    duration_seconds = samples_per_channel / loaded_audio.sample_rate_hz

    return AudioMetadata(
        filename=filename,
        content_type=content_type,
        format=loaded_audio.format,
        sample_rate_hz=loaded_audio.sample_rate_hz,
        channels=channels,
        samples_per_channel=samples_per_channel,
        total_samples=total_samples,
        duration_seconds=round(duration_seconds, 6),
        bit_depth=loaded_audio.bit_depth,
        sample_summary=AudioSampleSummary(
            array_shape=list(samples.shape),
            dtype=str(samples.dtype),
            min_value=samples.min().item(),
            max_value=samples.max().item(),
            mean_value=round(float(samples.mean()), 6),
            representation="NumPy ndarray; mono shape is [samples], multi-channel shape is [samples, channels].",
        ),
    )


def _build_waveform_response(dsp_waveform: DspWaveformData) -> WaveformData:
    """Convert a DSP-layer WaveformData into the API-layer WaveformData model."""
    api_channels: list[ChannelWaveform] = []
    for dsp_ch in dsp_waveform.channels:
        api_bins = [
            WaveformBin(
                time_seconds=b.time_seconds,
                min_amplitude=b.min_amplitude,
                max_amplitude=b.max_amplitude,
            )
            for b in dsp_ch.bins
        ]
        api_channels.append(
            ChannelWaveform(channel_index=dsp_ch.channel_index, bins=api_bins)
        )

    return WaveformData(
        n_channels=dsp_waveform.n_channels,
        n_bins=dsp_waveform.n_bins,
        duration_seconds=dsp_waveform.duration_seconds,
        sample_rate_hz=dsp_waveform.sample_rate_hz,
        channels=api_channels,
    )


# ---------------------------------------------------------------------------
# Module 02 — Upload + metadata only
# ---------------------------------------------------------------------------


async def load_uploaded_audio(file: UploadFile | None) -> AudioUploadResponse:
    """Validate, load, and return metadata for an uploaded WAV file.

    No waveform data is computed here; that belongs to Module 03 / analyze.
    """
    file_bytes, filename, content_type = await _read_and_validate_upload(file)

    try:
        loaded_audio = load_audio_bytes(file_bytes=file_bytes, filename=filename)
    except AudioDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    metadata = _build_metadata(filename, content_type, loaded_audio)
    return AudioUploadResponse(status="loaded", metadata=metadata)


# ---------------------------------------------------------------------------
# Module 03 — Upload + metadata + waveform
# ---------------------------------------------------------------------------


async def analyze_uploaded_audio(file: UploadFile | None) -> AudioAnalyzeResponse:
    """Validate, load, compute waveform, and return the full analysis payload.

    The waveform is computed by the DSP core using peak-preserving decimation.
    The raw sample array is not included in the response; it remains on the
    backend side and will be used by future DSP modules.
    """
    file_bytes, filename, content_type = await _read_and_validate_upload(file)

    try:
        loaded_audio = load_audio_bytes(file_bytes=file_bytes, filename=filename)
    except AudioDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    metadata = _build_metadata(filename, content_type, loaded_audio)

    try:
        dsp_waveform = compute_waveform(
            samples=loaded_audio.samples,
            sample_rate_hz=loaded_audio.sample_rate_hz,
            n_bins=DEFAULT_N_BINS,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Waveform computation failed: {exc}",
        ) from exc

    waveform = _build_waveform_response(dsp_waveform)

    return AudioAnalyzeResponse(status="analyzed", metadata=metadata, waveform=waveform)
