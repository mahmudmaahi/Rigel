"""
dsp_core/stft.py — Module 05: Spectrogram / STFT
==================================================

Computes a Short-Time Fourier Transform (STFT) spectrogram for a
discrete-time audio signal, producing a 2-D time-frequency representation
suitable for display.

Mathematical background
-----------------------

The STFT divides the signal into overlapping short frames and applies the
DFT to each frame:

    STFT[m, k] = Σ_{n=0}^{L-1}  x[n + m·H] · w[n] · exp(-j 2πkn/L)

Where:
    L   — frame length in samples (FFT window size)
    H   — hop length in samples (stride between successive frames)
    w[n] — window function applied to each frame (Hann by default)
    m   — frame index (time axis)
    k   — frequency bin index (0 … L//2, one-sided)

Windowing
---------

Each frame is multiplied by a Hann window before the FFT to reduce
spectral leakage caused by the abrupt frame boundaries:

    w[n] = 0.5 · (1 − cos(2πn / (L − 1)))

A Hann window tapers smoothly to zero at both ends. This prevents the
sharp discontinuities that would otherwise inject artificial high-frequency
energy into every frame's spectrum.

Frequency axis
--------------

Identical to Module 04 (single-frame FFT):

    Δf = Fs / L       — frequency resolution per bin (Hz)
    f[k] = k · Fs / L  — centre frequency of bin k (Hz)
    Nyquist = Fs / 2  — maximum representable frequency

Each STFT frame uses np.fft.rfft, which returns L//2 + 1 complex bins.

Time axis
---------

Frame m starts at sample index m · H.  Its centre time is:

    t[m] = (m · H + L/2) / Fs   (seconds)

For a signal of N samples, the number of complete frames is:

    n_frames = 1 + (N − L) // H   (assuming N ≥ L)

Magnitude in dBFS
-----------------

The same normalization convention as Module 04 is used so that both
the spectrum and spectrogram share a consistent dBFS scale:

1. Normalize samples to float amplitude in [−1, 1] (dBFS reference).
2. Multiply each frame by the Hann window.
3. Apply np.fft.rfft.
4. Compute per-bin magnitude: |STFT[m, k]| / L
5. Convert to dBFS: 20 · log10(magnitude + ε)

No one-sided amplitude doubling is applied here. The goal of the
spectrogram is to visualize relative energy distribution over time, and
the per-frame normalization by L is sufficient for that purpose.

Display downsampling
--------------------

Long audio files can produce thousands of frames and hundreds of frequency
bins. To keep the JSON payload small and rendering fast, the spectrogram is
downsampled before being returned to the API:

    N_TIME_BINS   = 512   (maximum time frames sent to frontend)
    N_FREQ_BINS   = 256   (maximum frequency rows sent to frontend)

Time downsampling:
    Consecutive frames are grouped and the maximum dBFS in each group is
    taken (peak-preserving, so short transients are not lost).

Frequency downsampling:
    Consecutive frequency bins are grouped and the maximum dBFS is taken.
    This preserves spectral peaks in the reduced resolution display.

Conceptual processing flow
--------------------------

    samples
      ↓
    normalize to float amplitude (→ dBFS reference)
      ↓
    frame into overlapping windows of length L with hop H
      ↓
    apply Hann window to each frame
      ↓
    rfft per frame → complex STFT matrix [n_frames × (L//2 + 1)]
      ↓
    magnitude |STFT[m, k]| / L
      ↓
    dBFS  20 · log10(magnitude + ε)
      ↓
    downsample time axis → N_TIME_BINS display frames
    downsample freq axis → N_FREQ_BINS display rows
      ↓
    SpectrogramData
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default STFT parameters
DEFAULT_FRAME_LENGTH: int = 2048   # samples per frame (≈ 46 ms at 44.1 kHz)
DEFAULT_HOP_LENGTH: int   = 512    # samples between frames (75% overlap)

# Display resolution caps — keep JSON payload ≈ reasonable size
N_TIME_BINS: int = 512   # maximum time columns returned to frontend
N_FREQ_BINS: int = 256   # maximum frequency rows returned to frontend

# Reference floor for silent bins (prevents log10(0), floor ≈ −240 dBFS)
_DB_EPSILON: float = 1e-12


# ---------------------------------------------------------------------------
# Data classes  (framework-independent — no FastAPI / Pydantic dependencies)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrogramChannel:
    """Display-ready spectrogram for one audio channel.

    Attributes
    ----------
    channel_index:
        0-based channel number (0 = left/mono, 1 = right, …).
    time_frames:
        Centre time in seconds for each display column.
        Length = number of display time bins (≤ N_TIME_BINS).
    freq_bins:
        Centre frequency in Hz for each display row.
        Length = number of display frequency rows (≤ N_FREQ_BINS).
        Sorted ascending (index 0 = DC, last index = near-Nyquist).
    magnitudes:
        2-D magnitude matrix in dBFS.
        Shape: [n_time_display, n_freq_display].
        magnitudes[t][f] is the peak dBFS in the region corresponding
        to display time column t and frequency row f.
        Values are in the range [DB_FLOOR, 0], where DB_FLOOR ≈ −240.
    """

    channel_index: int
    time_frames: list[float]
    freq_bins: list[float]
    magnitudes: list[list[float]]   # [n_time][n_freq]


@dataclass(frozen=True)
class SpectrogramData:
    """Full spectrogram payload, display-ready for the API and frontend.

    Attributes
    ----------
    n_channels:
        Number of audio channels.
    frame_length:
        STFT frame length in samples (L).
    hop_length:
        STFT hop length in samples (H).
    n_frames_full:
        Total STFT frames before display downsampling.
    n_freq_bins_full:
        Total one-sided frequency bins per frame (L//2 + 1) before downsampling.
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
        Name of the analysis window used (e.g. "hann").
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


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------


def get_window(name: str, length: int) -> np.ndarray:
    """Return the specified window function of the given length.
    
    Supported windows:
        - "rectangular"
        - "hamming"
        - "hann"
        - "blackman"
        - "kaiser"
        - "bartlett"
        - "welch"
    """
    if length < 1:
        raise ValueError(f"Window length must be ≥ 1; got {length}.")
    if length == 1:
        return np.ones(1, dtype=np.float64)
        
    name = name.lower().strip()
    
    if name == "rectangular":
        return np.ones(length, dtype=np.float64)
    elif name == "hamming":
        return np.hamming(length).astype(np.float64)
    elif name == "hann":
        return np.hanning(length).astype(np.float64)
    elif name == "blackman":
        return np.blackman(length).astype(np.float64)
    elif name == "kaiser":
        # Fixed beta=14.0 for high dynamic range analysis
        return np.kaiser(length, beta=14.0).astype(np.float64)
    elif name == "bartlett":
        return np.bartlett(length).astype(np.float64)
    elif name == "welch":
        n = np.arange(length, dtype=np.float64)
        M = (length - 1) / 2.0
        return 1.0 - ((n - M) / M) ** 2
    else:
        raise ValueError(f"Unsupported window '{name}'.")



# ---------------------------------------------------------------------------
# Sample normalisation  (identical convention to spectrum.py)
# ---------------------------------------------------------------------------


def _normalize_samples(samples: np.ndarray) -> np.ndarray:
    """Normalize PCM samples to float32 amplitude in [−1, 1].

    For integer dtypes, divides by the dtype's maximum representable value.
    For floating-point dtypes, casts to float32 without rescaling.

    Parameters
    ----------
    samples:
        1-D NumPy array of any numeric dtype.

    Returns
    -------
    float32 array with values in [−1, 1].
    """
    dtype = samples.dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        return (samples.astype(np.float64) / scale).astype(np.float32)
    return samples.astype(np.float32)


# ---------------------------------------------------------------------------
# Core STFT computation
# ---------------------------------------------------------------------------


def _stft_channel(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """Compute the raw STFT magnitude matrix for one channel.

    Each frame is extracted, windowed, and transformed with np.fft.rfft.
    The result is normalized by frame_length.

    Parameters
    ----------
    signal:
        1-D float32 array of normalized samples.
    frame_length:
        Number of samples per frame (L).
    hop_length:
        Number of samples between consecutive frame starts (H).
    window:
        Pre-computed window array of length frame_length.

    Returns
    -------
    float32 NumPy array of shape [n_frames, L//2 + 1].
    Each element is the normalized magnitude |STFT[m, k]| / L.
    """
    n_samples = len(signal)

    if n_samples < frame_length:
        # Pad with zeros so at least one frame can be computed
        signal = np.pad(signal, (0, frame_length - n_samples))
        n_samples = len(signal)

    # Number of complete frames
    n_frames = 1 + (n_samples - frame_length) // hop_length
    n_freq_bins = frame_length // 2 + 1

    magnitudes = np.zeros((n_frames, n_freq_bins), dtype=np.float32)

    for m in range(n_frames):
        start = m * hop_length
        frame = signal[start : start + frame_length].astype(np.float64)

        # Apply analysis window
        windowed = frame * window

        # One-sided FFT — returns L//2 + 1 complex coefficients
        X = np.fft.rfft(windowed)

        # Normalized magnitude: |X[k]| / L
        magnitudes[m] = (np.abs(X) / frame_length).astype(np.float32)

    return magnitudes


def _magnitudes_to_dbfs(magnitudes: np.ndarray) -> np.ndarray:
    """Convert a 2-D magnitude matrix to dBFS.

    magnitude_dBFS[m, k] = 20 · log10(magnitude[m, k] + ε)

    Parameters
    ----------
    magnitudes:
        float32 array of shape [n_frames, n_freq_bins] with values ≥ 0.

    Returns
    -------
    float32 array of same shape with dBFS values.
    """
    return (20.0 * np.log10(magnitudes.astype(np.float64) + _DB_EPSILON)).astype(np.float32)


# ---------------------------------------------------------------------------
# Display downsampling
# ---------------------------------------------------------------------------


def _downsample_time(
    magnitudes_dbfs: np.ndarray,
    n_display: int,
) -> np.ndarray:
    """Reduce the time axis of the magnitude matrix to at most n_display columns.

    Uses peak-preserving grouping: each display column stores the maximum
    dBFS in its group of raw frames, so short transients are not lost.

    Parameters
    ----------
    magnitudes_dbfs:
        float32 array of shape [n_frames, n_freq_bins].
    n_display:
        Target number of display time columns.

    Returns
    -------
    float32 array of shape [n_display_actual, n_freq_bins].
    n_display_actual = min(n_display, n_frames).
    """
    n_frames = magnitudes_dbfs.shape[0]
    actual = min(n_display, n_frames)

    if actual == n_frames:
        return magnitudes_dbfs

    # Split frames into 'actual' roughly equal groups and take the max per group
    indices = np.array_split(np.arange(n_frames), actual)
    result = np.stack(
        [magnitudes_dbfs[idx].max(axis=0) for idx in indices if len(idx) > 0],
        axis=0,
    )
    return result.astype(np.float32)


def _downsample_freq(
    magnitudes_dbfs: np.ndarray,
    n_display: int,
) -> np.ndarray:
    """Reduce the frequency axis of the magnitude matrix to at most n_display rows.

    Uses peak-preserving grouping: each display row stores the maximum
    dBFS in its group of raw frequency bins, so narrow peaks are preserved.

    Parameters
    ----------
    magnitudes_dbfs:
        float32 array of shape [n_time, n_freq_bins].
    n_display:
        Target number of display frequency rows.

    Returns
    -------
    float32 array of shape [n_time, n_display_actual].
    n_display_actual = min(n_display, n_freq_bins).
    """
    n_freq = magnitudes_dbfs.shape[1]
    actual = min(n_display, n_freq)

    if actual == n_freq:
        return magnitudes_dbfs

    indices = np.array_split(np.arange(n_freq), actual)
    result = np.stack(
        [magnitudes_dbfs[:, idx].max(axis=1) for idx in indices if len(idx) > 0],
        axis=1,
    )
    return result.astype(np.float32)


def _build_time_axis(
    n_frames_full: int,
    n_display: int,
    frame_length: int,
    hop_length: int,
    sample_rate_hz: int,
) -> list[float]:
    """Compute the centre time (in seconds) for each display time column.

    The centre time of raw frame m is:
        t[m] = (m · H + L/2) / Fs

    Display columns are formed by grouping consecutive raw frames, so
    each display column's time is the centre of its group's time range.
    """
    n_actual = min(n_display, n_frames_full)
    indices = np.array_split(np.arange(n_frames_full), n_actual)
    times = []
    for idx in indices:
        if len(idx) == 0:
            continue
        m_centre = (idx[0] + idx[-1]) / 2.0
        t = (m_centre * hop_length + frame_length / 2.0) / sample_rate_hz
        times.append(round(float(t), 6))
    return times


def _build_freq_axis(
    n_freq_bins_full: int,
    n_display: int,
    frame_length: int,
    sample_rate_hz: int,
) -> list[float]:
    """Compute the centre frequency (in Hz) for each display frequency row.

    Raw bin k has centre frequency:
        f[k] = k · Fs / L

    Display rows are formed by grouping consecutive raw bins, so each
    display row's frequency is the centre of its group's frequency range.
    """
    n_actual = min(n_display, n_freq_bins_full)
    raw_freqs = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate_hz)
    indices = np.array_split(np.arange(n_freq_bins_full), n_actual)
    freqs = []
    for idx in indices:
        if len(idx) == 0:
            continue
        centre = float(raw_freqs[(idx[0] + idx[-1]) // 2])
        freqs.append(round(centre, 3))
    return freqs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_spectrogram(
    samples: np.ndarray,
    sample_rate_hz: int,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    window: str = "hann",
    n_time_bins: int = N_TIME_BINS,
    n_freq_bins: int = N_FREQ_BINS,
) -> SpectrogramData:
    """Compute a display-ready STFT spectrogram.

    Processing flow per channel:
        samples  →  normalize  →  frame  →  windowing  →  rfft
            →  magnitude / L  →  dBFS  →  downsample time
            →  downsample freq  →  SpectrogramChannel

    Parameters
    ----------
    samples:
        NumPy array. Mono: shape [N]. Stereo/multi: shape [N, C].
    sample_rate_hz:
        Sample rate in Hz.
    frame_length:
        STFT window size in samples (default 2048).
        Controls frequency resolution: Δf = Fs / frame_length.
    hop_length:
        Stride between successive frames in samples (default 512).
        Controls time resolution: Δt = hop_length / Fs.
        Overlap = (frame_length − hop_length) / frame_length.
    window:
        Analysis window name (e.g. "hann", "hamming", "rectangular").
    n_time_bins:
        Maximum number of time columns to return (default N_TIME_BINS = 512).
    n_freq_bins:
        Maximum number of frequency rows to return (default N_FREQ_BINS = 256).

    Returns
    -------
    SpectrogramData with per-channel display-ready magnitude matrices.

    Raises
    ------
    ValueError:
        If frame_length < 2, hop_length < 1, or window is unsupported.
    """
    if frame_length < 2:
        raise ValueError(f"frame_length must be ≥ 2; got {frame_length}.")
    if hop_length < 1:
        raise ValueError(f"hop_length must be ≥ 1; got {hop_length}.")

    win = get_window(window, frame_length)

    # Split into per-channel 1-D arrays
    if samples.ndim == 1:
        channel_arrays = [samples]
    else:
        channel_arrays = [samples[:, c] for c in range(samples.shape[1])]

    n_channels = len(channel_arrays)
    n_samples = len(channel_arrays[0])
    duration_seconds = n_samples / sample_rate_hz
    frequency_resolution_hz = sample_rate_hz / frame_length
    nyquist_hz = sample_rate_hz / 2.0
    time_resolution_seconds = hop_length / sample_rate_hz

    # Compute full frame count (for metadata)
    padded_len = max(n_samples, frame_length)
    n_frames_full = 1 + (padded_len - frame_length) // hop_length
    n_freq_bins_full = frame_length // 2 + 1

    channel_spectrograms: list[SpectrogramChannel] = []

    for ch_idx, channel_samples in enumerate(channel_arrays):
        # 1. Normalize to float amplitude (dBFS reference)
        normalised = _normalize_samples(channel_samples)

        # 2. Compute STFT magnitude matrix [n_frames_full, n_freq_bins_full]
        raw_magnitudes = _stft_channel(normalised, frame_length, hop_length, win)

        # Actual frame count may differ from prediction if padding was needed
        actual_n_frames = raw_magnitudes.shape[0]

        # 3. Convert to dBFS
        raw_dbfs = _magnitudes_to_dbfs(raw_magnitudes)

        # 4. Downsample time axis
        ds_time = _downsample_time(raw_dbfs, n_time_bins)

        # 5. Downsample frequency axis
        ds_both = _downsample_freq(ds_time, n_freq_bins)

        n_time_display = ds_both.shape[0]
        n_freq_display = ds_both.shape[1]

        # 6. Build display axes
        time_axis = _build_time_axis(
            actual_n_frames, n_time_bins, frame_length, hop_length, sample_rate_hz
        )
        freq_axis = _build_freq_axis(
            n_freq_bins_full, n_freq_bins, frame_length, sample_rate_hz
        )

        # 7. Convert magnitude matrix to nested Python list (JSON-serializable)
        magnitudes_list: list[list[float]] = [
            [round(float(ds_both[t, f]), 2) for f in range(n_freq_display)]
            for t in range(n_time_display)
        ]

        channel_spectrograms.append(
            SpectrogramChannel(
                channel_index=ch_idx,
                time_frames=time_axis,
                freq_bins=freq_axis,
                magnitudes=magnitudes_list,
            )
        )

    first_ch = channel_spectrograms[0]

    return SpectrogramData(
        n_channels=n_channels,
        frame_length=frame_length,
        hop_length=hop_length,
        n_frames_full=n_frames_full,
        n_freq_bins_full=n_freq_bins_full,
        n_time_display=len(first_ch.time_frames),
        n_freq_display=len(first_ch.freq_bins),
        frequency_resolution_hz=round(frequency_resolution_hz, 6),
        nyquist_hz=nyquist_hz,
        time_resolution_seconds=round(time_resolution_seconds, 9),
        duration_seconds=round(duration_seconds, 6),
        sample_rate_hz=sample_rate_hz,
        window=window,
        channels=channel_spectrograms,
    )
