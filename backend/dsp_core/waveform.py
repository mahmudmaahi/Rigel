"""
Waveform decimation and time-axis generation.

This module is framework-independent.  It must not import FastAPI, Starlette,
Next.js, or any other application-layer code.  Its only dependencies are NumPy
and the Python standard library.

Design decisions
----------------
Bin timestamp convention
    Each bin represents a fixed-width chunk of *samples*.  The timestamp stored
    for a bin is the **start time** of that chunk, i.e. the time of the first
    sample that falls inside the bin.

    t_bin[k] = (k * bin_size) / sample_rate_hz

    This keeps the mapping between visual bins and audio time straightforward:
    clicking at bin k seeks to t_bin[k].  The ``WaveformBin.time_seconds``
    field always uses this convention.

Stereo strategy
    Stereo audio is preserved channel-by-channel rather than mixed to mono.
    Mixing to mono would hide stereo panning information and could cause
    phase-cancellation artefacts.  The frontend receives one ``ChannelWaveform``
    object per channel and chooses how to render them (typically as two
    stacked waveform rows).

Downsampling method
    Peak-envelope decimation: for each bin we record the minimum and maximum
    amplitude observed among all samples that fall inside that bin.  This
    guarantees that transient peaks visible at the original sample level are
    still visible in the decimated view, regardless of how aggressively the
    signal is downsampled.

    A naive mean or linear decimation would alias sharp peaks away and give
    the waveform a misleadingly smooth appearance.

Number of bins
    The default is 1 500 bins.  This value was chosen because:
    - It is sufficient visual resolution for typical waveform displays at
      widths of 600–2 000 pixels (roughly 1 bin per pixel or a small multiple).
    - It keeps the JSON payload manageable (≈ 36 KB for 1 500 float32 bins).
    - Callers may override ``n_bins`` for testing or different display sizes.
    For audio shorter than ``n_bins`` samples the actual number of bins is
    clamped to the number of samples, giving 1 bin per sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dsp_core.spectrum import normalize_samples

# Default number of visual bins produced by the decimator.
DEFAULT_N_BINS: int = 1_500


@dataclass(frozen=True)
class WaveformBin:
    """One visual bin in a decimated waveform.

    Attributes
    ----------
    time_seconds:
        Start time of the bin in seconds.  See the module docstring for the
        timestamp convention used throughout this module.
    min_amplitude:
        Minimum sample value observed within the bin.  For integer audio
        (e.g. int16) this is the raw integer value.  For float32 audio it is
        the normalised float.
    max_amplitude:
        Maximum sample value observed within the bin.
    """

    time_seconds: float
    min_amplitude: float
    max_amplitude: float


@dataclass(frozen=True)
class ChannelWaveform:
    """Decimated waveform data for a single audio channel.

    Attributes
    ----------
    channel_index:
        Zero-based index of the channel (0 = left, 1 = right for stereo).
    bins:
        Ordered sequence of :class:`WaveformBin` objects.
    """

    channel_index: int
    bins: list[WaveformBin] = field(default_factory=list)


@dataclass(frozen=True)
class WaveformData:
    """Full decimated waveform result ready for serialisation.

    Attributes
    ----------
    n_channels:
        Number of audio channels in the original signal.
    n_bins:
        Actual number of bins produced (may be less than the requested value
        for very short audio).
    duration_seconds:
        Total duration of the signal in seconds.
    sample_rate_hz:
        Original sample rate.
    channels:
        One :class:`ChannelWaveform` per channel, in channel order.
    """

    n_channels: int
    n_bins: int
    duration_seconds: float
    sample_rate_hz: int
    channels: list[ChannelWaveform] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_waveform(
    samples: np.ndarray,
    sample_rate_hz: int,
    n_bins: int = DEFAULT_N_BINS,
) -> WaveformData:
    """Decimate *samples* into a peak-envelope waveform suitable for display.

    Parameters
    ----------
    samples:
        NumPy array produced by :func:`dsp_core.audio_loader.load_audio_bytes`.
        Mono arrays have shape ``(num_samples,)``.
        Multi-channel arrays have shape ``(num_samples, num_channels)``.
    sample_rate_hz:
        Sample rate of the audio in Hz.
    n_bins:
        Requested number of visual bins.  The actual number may be lower if
        the audio contains fewer samples than ``n_bins``.

    Returns
    -------
    WaveformData
        Fully populated waveform result.

    Raises
    ------
    ValueError
        If ``samples`` is empty, ``sample_rate_hz`` is non-positive, or
        ``n_bins`` is less than 1.
    """
    if samples.size == 0:
        raise ValueError("Cannot compute waveform: the sample array is empty.")
    if sample_rate_hz <= 0:
        raise ValueError(
            f"Cannot compute waveform: invalid sample rate {sample_rate_hz!r}."
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins!r}.")

    # Normalize all samples to [-1.0, 1.0] scale so that int16 uploads
    # and float32 processed outputs share the same amplitude reference.
    signal = normalize_samples(samples).astype(np.float64)

    # Ensure shape is always (num_samples, num_channels).
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]  # mono: (N,) → (N, 1)

    num_samples, num_channels = signal.shape
    duration_seconds = num_samples / sample_rate_hz

    # Clamp n_bins so we never produce more bins than there are samples.
    actual_n_bins = min(n_bins, num_samples)

    channel_waveforms: list[ChannelWaveform] = []

    for ch_idx in range(num_channels):
        channel_signal = signal[:, ch_idx]
        bins = _decimate_channel(
            channel_signal=channel_signal,
            sample_rate_hz=sample_rate_hz,
            n_bins=actual_n_bins,
        )
        channel_waveforms.append(ChannelWaveform(channel_index=ch_idx, bins=bins))

    return WaveformData(
        n_channels=num_channels,
        n_bins=actual_n_bins,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        channels=channel_waveforms,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decimate_channel(
    channel_signal: np.ndarray,
    sample_rate_hz: int,
    n_bins: int,
) -> list[WaveformBin]:
    """Produce ``n_bins`` peak-envelope bins from a 1-D channel signal.

    The function partitions the signal into ``n_bins`` roughly equal-width
    chunks.  NumPy ``array_split`` handles the rounding so that adjacent bins
    differ in width by at most one sample.

    For each chunk the minimum and maximum values are recorded.  The bin
    timestamp is the start time of the chunk (see module docstring).

    Parameters
    ----------
    channel_signal:
        1-D float64 array for a single channel.
    sample_rate_hz:
        Original sample rate.
    n_bins:
        Number of bins to produce.

    Returns
    -------
    list[WaveformBin]
    """
    num_samples = len(channel_signal)
    # Split indices: array_split distributes the remainder across the first
    # bins, meaning most bins have size floor(N/n_bins) and the first
    # (N % n_bins) bins have size floor(N/n_bins) + 1.
    chunks = np.array_split(channel_signal, n_bins)

    bins: list[WaveformBin] = []
    sample_cursor = 0  # tracks the first sample index of the current chunk

    for chunk in chunks:
        if chunk.size == 0:
            # Degenerate: should not happen unless n_bins > num_samples,
            # which is already guarded above, but defensive coding is cheap.
            continue

        time_seconds = sample_cursor / sample_rate_hz
        bins.append(
            WaveformBin(
                time_seconds=round(time_seconds, 9),
                min_amplitude=float(chunk.min()),
                max_amplitude=float(chunk.max()),
            )
        )
        sample_cursor += len(chunk)

    return bins
