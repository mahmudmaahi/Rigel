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
