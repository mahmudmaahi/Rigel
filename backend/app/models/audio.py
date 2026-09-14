from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Module 02 — Audio metadata
# ---------------------------------------------------------------------------


class AudioSampleSummary(BaseModel):
    array_shape: list[int]
    dtype: str
    min_value: float | int
    max_value: float | int
    mean_value: float
    representation: str


class AudioMetadata(BaseModel):
    filename: str
    content_type: str
    format: str
    sample_rate_hz: int
    channels: int
    samples_per_channel: int
    total_samples: int
    duration_seconds: float
    bit_depth: int | None
    sample_summary: AudioSampleSummary


class AudioUploadResponse(BaseModel):
    status: str
    metadata: AudioMetadata


# ---------------------------------------------------------------------------
# Module 03 — Waveform visualization
# ---------------------------------------------------------------------------


class WaveformBin(BaseModel):
    """One visual bin in a decimated waveform.

    The ``time_seconds`` field represents the **start time** of the bin chunk,
    i.e. the time of the first sample that falls inside that bin.  This
    convention is used consistently for rendering tick marks and for
    translating a click position back into an audio seek time.
    """

    time_seconds: float
    min_amplitude: float
    max_amplitude: float


class ChannelWaveform(BaseModel):
    """Decimated waveform for one audio channel."""

    channel_index: int
    bins: list[WaveformBin]


class WaveformData(BaseModel):
    """Full waveform payload returned to the frontend.

    Attributes
    ----------
    n_channels:
        Number of channels in the original audio.
    n_bins:
        Number of bins actually produced (may be lower than requested for very
        short audio).
    duration_seconds:
        Total audio duration.
    sample_rate_hz:
        Original sample rate.
    channels:
        One ``ChannelWaveform`` per channel, in channel order.
    """

    n_channels: int
    n_bins: int
    duration_seconds: float
    sample_rate_hz: int
    channels: list[ChannelWaveform]


class AudioAnalyzeResponse(BaseModel):
    """Response from ``POST /api/audio/analyze``."""

    status: str
    metadata: AudioMetadata
    waveform: WaveformData


# ---------------------------------------------------------------------------
# Module 04 — Fourier Analysis
# ---------------------------------------------------------------------------


class SpectrumBin(BaseModel):
    """One display bin in the one-sided amplitude magnitude spectrum.

    Attributes
    ----------
    frequency_hz:
        Centre frequency of this display bin in Hz.
    magnitude_dbfs:
        Peak amplitude magnitude in dBFS (decibels relative to digital full
        scale) within this display bin.  0 dBFS = full-scale amplitude.
        Values are negative for signals below full scale.
        This is NOT acoustic sound-pressure dB.
    """

    frequency_hz: float
    magnitude_dbfs: float


class ChannelSpectrum(BaseModel):
    """Magnitude spectrum for one audio channel."""

    channel_index: int
    bins: list[SpectrumBin]


class SpectrumData(BaseModel):
    """Full spectrum payload returned by ``POST /api/audio/spectrum``.

    Attributes
    ----------
    n_channels:
        Number of audio channels.
    n_fft_full:
        Full FFT length used (= number of samples per channel, N).
        Raw one-sided FFT had N//2 + 1 bins before display reduction.
    n_display_bins:
        Number of bins in the returned display spectrum (≤ 1024).
    frequency_resolution_hz:
        Δf = Fs / N — spacing between raw FFT bins.
        NOT the display bin width.
    nyquist_hz:
        Theoretical Nyquist frequency = Fs / 2.
    duration_seconds:
        Total audio duration.
    sample_rate_hz:
        Original sample rate in Hz.
    channels:
        Per-channel display-ready magnitude spectrum.
    """

    n_channels: int
    n_fft_full: int
    n_display_bins: int
    frequency_resolution_hz: float
    nyquist_hz: float
    duration_seconds: float
    sample_rate_hz: int
    channels: list[ChannelSpectrum]


class AudioSpectrumResponse(BaseModel):
    """Response from ``POST /api/audio/spectrum``."""

    status: str
    metadata: AudioMetadata
    spectrum: SpectrumData


# ---------------------------------------------------------------------------
# Module 05 — Spectrogram / STFT
# ---------------------------------------------------------------------------


class SpectrogramChannel(BaseModel):
    """Display-ready STFT spectrogram for one audio channel.

    Attributes
    ----------
    channel_index:
        0-based channel number (0 = left/mono, 1 = right, …).
    time_frames:
        Centre time in seconds for each display time column.
    freq_bins:
        Centre frequency in Hz for each display frequency row.
        Sorted ascending (index 0 = DC, last = near-Nyquist).
    magnitudes:
        2-D magnitude matrix in dBFS.
        Shape: [n_time_display][n_freq_display].
        magnitudes[t][f] is the peak dBFS in the display region at
        time column t and frequency row f.
    """

    channel_index: int
    time_frames: list[float]
    freq_bins: list[float]
    magnitudes: list[list[float]]


class SpectrogramData(BaseModel):
    """Full spectrogram payload returned by ``POST /api/audio/spectrogram``.

    Attributes
    ----------
    n_channels:
        Number of audio channels.
    frame_length:
        STFT window size in samples (L).
    hop_length:
        STFT hop size in samples (H).
    n_frames_full:
        Total STFT frames before display downsampling.
    n_freq_bins_full:
        Total one-sided frequency bins per frame (L//2 + 1).
    n_time_display:
        Number of time columns in the returned display matrix.
    n_freq_display:
        Number of frequency rows in the returned display matrix.
    frequency_resolution_hz:
        Δf = Fs / L — frequency spacing between raw STFT bins.
    nyquist_hz:
        Theoretical Nyquist frequency = Fs / 2.
    time_resolution_seconds:
        Δt = H / Fs — time spacing between raw STFT frames.
    duration_seconds:
        Total audio duration.
    sample_rate_hz:
        Original sample rate in Hz.
    window:
        Analysis window used (e.g. "hann").
    channels:
        One SpectrogramChannel per audio channel.
    """

    n_channels: int
    frame_length: int
    hop_length: int
    n_frames_full: int
    n_freq_bins_full: int
    n_time_display: int
    n_freq_display: int
    frequency_resolution_hz: float
    nyquist_hz: float
    time_resolution_seconds: float
    duration_seconds: float
    sample_rate_hz: int
    window: str
    channels: list[SpectrogramChannel]


class AudioSpectrogramResponse(BaseModel):
    """Response from ``POST /api/audio/spectrogram``."""

    status: str
    metadata: AudioMetadata
    spectrogram: SpectrogramData


# ---------------------------------------------------------------------------
# Module 06 — Audio Export
# ---------------------------------------------------------------------------


class AudioExportRequest(BaseModel):
    sample_rate_hz: int
    samples: list[list[float]]


# ---------------------------------------------------------------------------
# Module 06 — Audio Filtering
# ---------------------------------------------------------------------------

from typing import Literal  # noqa: E402

_FILTER_TYPE = Literal["lowpass", "highpass", "bandpass", "bandstop", "peaking"]
_FILTER_FAMILY = Literal[
    "butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel",
    "fir_window", "fir_remez",
]


class AudioFilterRequest(BaseModel):
    """Parameters for applying a digital filter to an uploaded audio file.

    For lowpass/highpass: set cutoff_hz.
    For bandpass/bandstop: set low_hz and high_hz.
    For peaking EQ (filter_type='peaking'): set center_hz, gain_db, q_factor.
    Chebyshev I + Elliptic: also set ripple_db.
    Chebyshev II + Elliptic: also set attenuation_db.
    FIR window: optionally set fir_window.
    Parks-McClellan (fir_remez): set transition_bandwidth_hz.
    """

    filter_type: _FILTER_TYPE
    family: _FILTER_FAMILY = "butterworth"
    order: int
    cutoff_hz: float | None = None
    low_hz: float | None = None
    high_hz: float | None = None
    # IIR family-specific
    ripple_db: float = 1.0
    attenuation_db: float = 40.0
    # FIR-specific
    fir_window: str = "hamming"
    transition_bandwidth_hz: float = 200.0
    # Peaking EQ
    center_hz: float | None = None
    gain_db: float = 0.0
    q_factor: float = 1.0


class AudioFilterResponseRequest(BaseModel):
    """Parameters for computing the theoretical frequency response of a filter.

    Does not require an audio file; only needs sample_rate_hz to normalise
    cutoff frequencies.
    """

    filter_type: _FILTER_TYPE
    family: _FILTER_FAMILY = "butterworth"
    order: int
    sample_rate_hz: int
    cutoff_hz: float | None = None
    low_hz: float | None = None
    high_hz: float | None = None
    ripple_db: float = 1.0
    attenuation_db: float = 40.0
    fir_window: str = "hamming"
    transition_bandwidth_hz: float = 200.0
    center_hz: float | None = None
    gain_db: float = 0.0
    q_factor: float = 1.0


class AudioFilterResponse(BaseModel):
    """Theoretical frequency response of a designed filter.

    Returned by ``POST /api/audio/filter/response``.
    Used by the frontend to visualise the filter shape before applying.
    """

    filter_type: _FILTER_TYPE
    family: _FILTER_FAMILY
    order: int
    sample_rate_hz: int
    cutoff_hz: float | None
    low_hz: float | None
    high_hz: float | None
    center_hz: float | None
    frequencies_hz: list[float]
    magnitude_db: list[float]

