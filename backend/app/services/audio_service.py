from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models.audio import (
    AudioAnalyzeResponse,
    AudioMetadata,
    AudioSampleSummary,
    AudioSpectrogramResponse,
    AudioSpectrumResponse,
    AudioUploadResponse,
    ChannelSpectrum,
    ChannelWaveform,
    SpectrogramChannel,
    SpectrogramData,
    SpectrumBin,
    SpectrumData as ApiSpectrumData,
    WaveformBin,
    WaveformData,
)
from dsp_core.audio_loader import AudioDecodeError, load_audio_bytes
from dsp_core.spectrum import (
    ChannelSpectrum as DspChannelSpectrum,
    SpectrumData as DspSpectrumData,
    compute_spectrum,
)
from dsp_core.stft import (
    SpectrogramChannel as DspSpectrogramChannel,
    SpectrogramData as DspSpectrogramData,
    compute_spectrogram,
)
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


# ---------------------------------------------------------------------------
# Module 04 — Upload + metadata + frequency spectrum
# ---------------------------------------------------------------------------


def _build_spectrum_response(dsp_spectrum: DspSpectrumData) -> ApiSpectrumData:
    """Convert a DSP-layer SpectrumData into the API-layer SpectrumData model."""
    api_channels: list[ChannelSpectrum] = []
    for dsp_ch in dsp_spectrum.channels:
        api_bins = [
            SpectrumBin(
                frequency_hz=b.frequency_hz,
                magnitude_dbfs=b.magnitude_dbfs,
            )
            for b in dsp_ch.bins
        ]
        api_channels.append(
            ChannelSpectrum(channel_index=dsp_ch.channel_index, bins=api_bins)
        )

    return ApiSpectrumData(
        n_channels=dsp_spectrum.n_channels,
        n_fft_full=dsp_spectrum.n_fft_full,
        n_display_bins=dsp_spectrum.n_display_bins,
        frequency_resolution_hz=dsp_spectrum.frequency_resolution_hz,
        nyquist_hz=dsp_spectrum.nyquist_hz,
        duration_seconds=dsp_spectrum.duration_seconds,
        sample_rate_hz=dsp_spectrum.sample_rate_hz,
        channels=api_channels,
    )


async def compute_spectrum_for_upload(file: UploadFile | None) -> AudioSpectrumResponse:
    """Validate, load, compute frequency spectrum, and return the spectrum payload.

    Calls dsp_core.spectrum.compute_spectrum which:
      1. Normalizes PCM samples to float amplitude (dBFS reference).
      2. Computes the one-sided FFT per channel with correct amplitude scaling.
      3. Converts amplitudes to dBFS.
      4. Reduces to 1024 display bins for the frontend.

    The raw sample array and full FFT result are NOT sent to the frontend.
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
        dsp_spectrum = compute_spectrum(
            samples=loaded_audio.samples,
            sample_rate_hz=loaded_audio.sample_rate_hz,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Spectrum computation failed: {exc}",
        ) from exc

    spectrum = _build_spectrum_response(dsp_spectrum)

    return AudioSpectrumResponse(status="spectrum_computed", metadata=metadata, spectrum=spectrum)


# ---------------------------------------------------------------------------
# Module 05 — Upload + metadata + spectrogram
# ---------------------------------------------------------------------------


def _build_spectrogram_response(dsp_sg: DspSpectrogramData) -> SpectrogramData:
    """Convert a DSP-layer SpectrogramData into the API-layer SpectrogramData model."""
    api_channels: list[SpectrogramChannel] = [
        SpectrogramChannel(
            channel_index=ch.channel_index,
            time_frames=ch.time_frames,
            freq_bins=ch.freq_bins,
            magnitudes=ch.magnitudes,
        )
        for ch in dsp_sg.channels
    ]

    return SpectrogramData(
        n_channels=dsp_sg.n_channels,
        frame_length=dsp_sg.frame_length,
        hop_length=dsp_sg.hop_length,
        n_frames_full=dsp_sg.n_frames_full,
        n_freq_bins_full=dsp_sg.n_freq_bins_full,
        n_time_display=dsp_sg.n_time_display,
        n_freq_display=dsp_sg.n_freq_display,
        frequency_resolution_hz=dsp_sg.frequency_resolution_hz,
        nyquist_hz=dsp_sg.nyquist_hz,
        time_resolution_seconds=dsp_sg.time_resolution_seconds,
        duration_seconds=dsp_sg.duration_seconds,
        sample_rate_hz=dsp_sg.sample_rate_hz,
        window=dsp_sg.window,
        channels=api_channels,
    )


async def compute_spectrogram_for_upload(
    file: UploadFile | None, window: str = "hann"
) -> AudioSpectrogramResponse:
    """Validate, load, compute STFT spectrogram, and return the spectrogram payload.

    Calls dsp_core.stft.compute_spectrogram which:
      1. Normalizes PCM samples to float amplitude (dBFS reference).
      2. Frames the signal with frame_length=2048, hop_length=512.
      3. Applies the requested window to each frame.
      4. Computes np.fft.rfft per frame.
      5. Converts per-frame magnitudes to dBFS.
      6. Downsamples to 512 time columns × 256 frequency rows for the frontend.

    The raw sample array and full STFT matrix are NOT sent to the frontend.
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
        dsp_sg = compute_spectrogram(
            samples=loaded_audio.samples,
            sample_rate_hz=loaded_audio.sample_rate_hz,
            window=window,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Spectrogram computation failed: {exc}",
        ) from exc

    spectrogram = _build_spectrogram_response(dsp_sg)

    return AudioSpectrogramResponse(
        status="spectrogram_computed",
        metadata=metadata,
        spectrogram=spectrogram,
    )
