"""
dsp_core/spectrum.py — Module 04: Fourier Analysis
====================================================

Computes the one-sided amplitude magnitude spectrum (in dBFS) of a
discrete-time audio signal using both a naive DFT (educational) and
the NumPy FFT (production).

Mathematical background
-----------------------

The Discrete Fourier Transform (DFT) of a length-N signal x[n] is:

    X[k] = Σ_{n=0}^{N-1}  x[n] · exp(-j 2πkn/N)     k = 0, 1, ..., N-1

For a real-valued signal of length N the DFT output is conjugate-symmetric.
The negative-frequency coefficients (k > N/2) are mirrors of the
positive-frequency ones, so we keep only the one-sided spectrum:

    k = 0, 1, ..., N//2   (N//2 + 1 bins total — produced by np.fft.rfft)

Frequency axis
--------------

Each raw FFT bin k corresponds to physical frequency:

    f[k] = k · Fs / N        (Hz)

This gives:

    frequency resolution:  Δf = Fs / N   (Hz per bin)
    Nyquist frequency:     f_Nyquist = Fs / 2  (theoretical maximum)

For even N the final rFFT bin (k = N//2) is exactly at Fs/2.
For odd N the final rFFT bin is slightly below Fs/2.

One-sided amplitude scaling
---------------------------

np.fft.rfft discards the negative-frequency bins. To preserve the correct
amplitude in the one-sided spectrum, multiply interior bins by 2:

    Even N:
        A[0]      = |X[0]|   / N                (DC — not doubled)
        A[k]      = 2|X[k]| / N    1 ≤ k < N//2 (interior positive freqs)
        A[N//2]   = |X[N//2]| / N               (Nyquist bin — not doubled)

    Odd N:
        A[0]      = |X[0]|   / N                (DC — not doubled)
        A[k]      = 2|X[k]| / N    1 ≤ k ≤ (N-1)//2  (all remaining bins)
        (No exact Nyquist bin exists for odd N)

Magnitude in dBFS
-----------------

Samples are normalized to floating-point amplitude in [-1, 1] before the
FFT, so that the resulting magnitude has a consistent full-scale reference:

    0 dBFS corresponds to full-scale amplitude (A[k] = 1.0).
    Negative dBFS values represent signals below full scale.

This is NOT acoustic sound-pressure dB. The reference is digital full scale.

    magnitude_dBFS[k] = 20 · log10(A[k] + ε)

The small ε = 1e-12 prevents log10(0) for silent bins.

Complexity
----------

    Naive DFT:  O(N²)    — two nested loops, direct application of the equation
    NumPy FFT:  O(N log N) — Cooley-Tukey divide-and-conquer

Conceptual processing flow
--------------------------

    samples
      ↓
    normalize to float amplitude (→ dBFS reference)
      ↓
    FFT  (np.fft.rfft)
      ↓
    magnitude  |X[k]|
      ↓
    one-sided amplitude scaling  A[k]
      ↓
    dBFS conversion  20·log10(A[k] + ε)
      ↓
    display reduction  (1024 peak-preserving bins for the frontend)

Display reduction vs raw FFT resolution
----------------------------------------

The raw FFT has frequency resolution Δf = Fs / N. For long audio this
produces hundreds of thousands of bins (e.g. N = 1,469,952 → ~735,000 bins).

The display reduction step maps those raw bins onto N_DISPLAY_BINS = 1024
display bins for efficient canvas rendering. Each display bin spans a
frequency region of width:

    display_bin_width = (Fs / 2) / N_DISPLAY_BINS

and stores the maximum dBFS within that region (peak-preserving, so
important spectral peaks are not averaged away).

This display reduction is a visualization convenience step. It is NOT part
of the DFT/FFT mathematics and does not affect the underlying computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of frequency bins returned to the frontend for visualization.
# Keeps the JSON payload ≈ 16 KB regardless of audio length.
# Analogous to DEFAULT_N_BINS = 1500 in waveform.py.
N_DISPLAY_BINS: int = 1024

# Reference floor to prevent log10(0) for silent bins (−240 dBFS effectively).
_DB_EPSILON: float = 1e-12


# ---------------------------------------------------------------------------
# Data classes  (framework-independent, no FastAPI / Pydantic dependencies)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumBin:
    """One display bin in the magnitude spectrum.

    Attributes
    ----------
    frequency_hz:
        Centre frequency of this display bin in Hz.
    magnitude_dbfs:
        Peak one-sided amplitude magnitude in dBFS within this display bin.
        0 dBFS = full-scale amplitude (normalized signal value of 1.0).
    """

    frequency_hz: float
    magnitude_dbfs: float


@dataclass(frozen=True)
class ChannelSpectrum:
    """Magnitude spectrum for one audio channel."""

    channel_index: int
    bins: list[SpectrumBin]


@dataclass(frozen=True)
class SpectrumData:
    """Full spectrum payload, display-ready for the API and frontend.

    Attributes
    ----------
    n_channels:
        Number of audio channels in the original file.
    n_fft_full:
        Full FFT length used (= number of samples per channel, N).
        The raw one-sided FFT had N//2 + 1 bins before display reduction.
    n_display_bins:
        Number of bins in the returned (display-reduced) spectrum.
        Always ≤ N_DISPLAY_BINS = 1024.
    frequency_resolution_hz:
        Δf = Fs / N — frequency spacing between raw FFT bins.
        NOT the display bin width.
    nyquist_hz:
        Theoretical Nyquist frequency = Fs / 2.
        For even N the final raw bin is exactly at Fs/2.
        For odd N the final raw bin is slightly below Fs/2.
    duration_seconds:
        Total audio duration in seconds.
    sample_rate_hz:
        Original sample rate in Hz.
    channels:
        One ChannelSpectrum per audio channel, in channel order.
    """

    n_channels: int
    n_fft_full: int
    n_display_bins: int
    frequency_resolution_hz: float
    nyquist_hz: float
    duration_seconds: float
    sample_rate_hz: int
    channels: list[ChannelSpectrum]


# ---------------------------------------------------------------------------
# Sample normalisation
# ---------------------------------------------------------------------------


def normalize_samples(samples: np.ndarray) -> np.ndarray:
    """Normalize PCM samples to float32 amplitude in [-1.0, 1.0].

    Normalizing before the FFT gives the spectrum a consistent full-scale
    amplitude reference so that the result can be expressed meaningfully in
    dBFS (0 dBFS = full-scale amplitude of 1.0).

    Supports integer PCM (int8, int16, int32) and floating-point WAV samples.
    For integer types, divides by the maximum representable positive integer.
    For floating-point types, casts to float32 without rescaling.

    Parameters
    ----------
    samples:
        NumPy array of any numeric dtype (1-D channel array).

    Returns
    -------
    float32 array with values in [-1.0, 1.0].
    """
    dtype = samples.dtype

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        return (samples.astype(np.float64) / scale).astype(np.float32)

    # Float32 / float64 WAV — cast without rescaling
    return samples.astype(np.float32)


# ---------------------------------------------------------------------------
# Naive DFT — O(N²), educational reference
# ---------------------------------------------------------------------------


def compute_dft_naive(signal: np.ndarray) -> np.ndarray:
    """Compute the DFT of a real signal using the definition directly.

    This O(N²) implementation exists purely for education. The student can
    read this code and directly connect each line to the DFT equation:

        X[k] = Σ_{n=0}^{N-1}  x[n] · exp(-j 2πkn/N)

    Complexity:  O(N²)  — two nested loops each of length N.

    Do NOT call this in production. For actual spectrum computation
    use compute_spectrum(), which uses the O(N log N) FFT.

    Parameters
    ----------
    signal:
        1-D real-valued NumPy array of length N.

    Returns
    -------
    Complex NumPy array of length N — the full two-sided DFT.
    """
    N = len(signal)
    X = np.zeros(N, dtype=complex)

    for k in range(N):          # for each output frequency bin k …
        for n in range(N):      # … sum the contribution of every sample n
            X[k] += signal[n] * np.exp(-1j * 2 * np.pi * k * n / N)

    return X


# ---------------------------------------------------------------------------
# Production FFT helpers — O(N log N)
# ---------------------------------------------------------------------------


def _compute_one_sided_magnitude(
    signal: np.ndarray,
    sample_rate_hz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided amplitude magnitude spectrum via np.fft.rfft.

    Uses np.fft.rfft which exploits the conjugate symmetry of a real input
    and returns only the positive-frequency coefficients (bins 0 → N//2).

    One-sided amplitude scaling is applied to account for the discarded
    negative-frequency bins:

        Even N:
            A[0]     = |X[0]|   / N                (DC — not doubled)
            A[k]     = 2|X[k]| / N   1 ≤ k < N//2  (interior bins — doubled)
            A[N//2]  = |X[N//2]| / N               (Nyquist bin — not doubled)

        Odd N:
            A[0]     = |X[0]|   / N                (DC — not doubled)
            A[k]     = 2|X[k]| / N   1 ≤ k ≤ (N-1)//2  (all others — doubled)
            (No exact Nyquist bin; the last bin is slightly below Fs/2)

    Complexity:  O(N log N)

    Parameters
    ----------
    signal:
        1-D float32 array, already normalized to [-1, 1].
    sample_rate_hz:
        Sample rate in Hz, used to build the frequency axis.

    Returns
    -------
    frequencies_hz:
        float32 array of length N//2 + 1 with the frequency for each bin:
            f[k] = k · Fs / N
    amplitudes:
        float32 array of length N//2 + 1 with one-sided amplitude A[k].
    """
    N = len(signal)

    # np.fft.rfft — returns N//2 + 1 complex coefficients
    X = np.fft.rfft(signal)

    # Frequency axis:  f[k] = k · Fs / N
    # np.fft.rfftfreq handles even/odd N correctly.
    frequencies_hz = np.fft.rfftfreq(N, d=1.0 / sample_rate_hz).astype(np.float32)

    # Start with raw magnitudes normalized by N
    amplitudes = np.abs(X).astype(np.float64) / N

    # Apply one-sided scaling.
    # Interior bins (exclude index 0 and, for even N, the last Nyquist bin)
    # receive factor of 2 to compensate for the discarded mirror bins.
    if N % 2 == 0:
        # Even N: last bin (index N//2) is the exact Nyquist bin — not doubled
        amplitudes[1:-1] *= 2
    else:
        # Odd N: no exact Nyquist bin; all bins after DC are interior — doubled
        amplitudes[1:] *= 2

    return frequencies_hz, amplitudes.astype(np.float32)


def _amplitudes_to_dbfs(amplitudes: np.ndarray) -> np.ndarray:
    """Convert linear amplitudes to dBFS.

    dBFS: decibels relative to digital full scale.
    0 dBFS = amplitude of 1.0 (full scale).
    Values below full scale are negative.

        magnitude_dBFS[k] = 20 · log10(A[k] + ε)

    The epsilon (1e-12) prevents log10(0) for silent bins (~-240 dBFS floor).

    Parameters
    ----------
    amplitudes:
        Linear one-sided amplitude array A[k] in [0, 1].

    Returns
    -------
    float32 array of dBFS values.
    """
    return (20.0 * np.log10(amplitudes + _DB_EPSILON)).astype(np.float32)


# ---------------------------------------------------------------------------
# Display reduction  (visualization step — NOT part of the DFT mathematics)
# ---------------------------------------------------------------------------


def _downsample_spectrum(
    frequencies_hz: np.ndarray,
    magnitudes_dbfs: np.ndarray,
    n_display_bins: int,
) -> list[SpectrumBin]:
    """Reduce the raw FFT spectrum to n_display_bins for the frontend.

    This is a visualization convenience step, not part of the DFT/FFT.

    Strategy — peak-preserving linear-frequency binning:
        Divide [0, Nyquist] into n_display_bins equal-width regions.
        For each region, record:
            - centre frequency of the region
            - maximum magnitude_dBFS among all raw FFT bins in the region

    Preserving the maximum (rather than averaging) ensures that narrow
    spectral peaks are not smoothed away during the reduction.

    The display bin width is:
        display_bin_width = (Fs / 2) / n_display_bins
    This is NOT the same as the raw FFT frequency resolution Δf = Fs / N.

    Parameters
    ----------
    frequencies_hz:
        Raw FFT frequency axis, length M = N//2 + 1.
    magnitudes_dbfs:
        Raw FFT dBFS magnitudes, same length M.
    n_display_bins:
        Target number of output display bins (≤ N_DISPLAY_BINS).

    Returns
    -------
    List of SpectrumBin, length ≤ n_display_bins.
    """
    nyquist_hz = float(frequencies_hz[-1])
    n_raw = len(frequencies_hz)

    # Clamp to raw bin count (very short audio may have fewer raw bins)
    actual_bins = min(n_display_bins, n_raw)

    bin_width_hz = nyquist_hz / actual_bins
    display_bins: list[SpectrumBin] = []

    for b in range(actual_bins):
        f_low = b * bin_width_hz
        f_high = f_low + bin_width_hz
        f_center = f_low + bin_width_hz / 2.0

        # Indices of raw FFT bins whose frequency falls in [f_low, f_high)
        mask = (frequencies_hz >= f_low) & (frequencies_hz < f_high)

        if np.any(mask):
            peak_dbfs = float(np.max(magnitudes_dbfs[mask]))
        else:
            # No raw bin in this display region — snap to nearest raw bin
            nearest_idx = int(np.argmin(np.abs(frequencies_hz - f_center)))
            peak_dbfs = float(magnitudes_dbfs[nearest_idx])

        display_bins.append(
            SpectrumBin(
                frequency_hz=round(f_center, 3),
                magnitude_dbfs=round(peak_dbfs, 3),
            )
        )

    return display_bins


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_spectrum(
    samples: np.ndarray,
    sample_rate_hz: int,
    n_display_bins: int = N_DISPLAY_BINS,
) -> SpectrumData:
    """Compute the display-ready one-sided amplitude magnitude spectrum.

    Processing flow:
        samples  →  normalize  →  FFT  →  |X[k]|  →  one-sided scaling
            →  dBFS  →  display reduction  →  SpectrumData

    The raw FFT uses all samples (N = samples per channel), giving the best
    frequency resolution Δf = Fs / N. The result is then reduced to
    n_display_bins for the frontend (display reduction is a visualization
    step, not part of the spectrum mathematics).

    Parameters
    ----------
    samples:
        NumPy array. Mono: shape [N]. Stereo: shape [N, C].
    sample_rate_hz:
        Sample rate in Hz.
    n_display_bins:
        Target number of display bins for the frontend (default 1024).

    Returns
    -------
    SpectrumData with per-channel display bins and spectrum metadata.
    """
    if samples.ndim == 1:
        channel_arrays = [samples]
    else:
        # Shape [N, C] — split into C independent 1-D channel arrays
        channel_arrays = [samples[:, c] for c in range(samples.shape[1])]

    n_channels = len(channel_arrays)
    n_samples = len(channel_arrays[0])
    duration_seconds = n_samples / sample_rate_hz
    frequency_resolution_hz = sample_rate_hz / n_samples  # Δf = Fs / N
    nyquist_hz = sample_rate_hz / 2.0                     # theoretical Nyquist

    channel_spectra: list[ChannelSpectrum] = []

    for ch_idx, channel_samples in enumerate(channel_arrays):
        # 1. Normalize to float amplitude  →  dBFS reference
        normalised = normalize_samples(channel_samples)

        # 2. One-sided FFT with correct amplitude scaling
        frequencies_hz, amplitudes = _compute_one_sided_magnitude(normalised, sample_rate_hz)

        # 3. Convert amplitude to dBFS
        magnitudes_dbfs = _amplitudes_to_dbfs(amplitudes)

        # 4. Reduce to display bins (visualization step)
        display_bins = _downsample_spectrum(frequencies_hz, magnitudes_dbfs, n_display_bins)

        channel_spectra.append(ChannelSpectrum(channel_index=ch_idx, bins=display_bins))

    actual_display_bins = len(channel_spectra[0].bins) if channel_spectra else 0

    return SpectrumData(
        n_channels=n_channels,
        n_fft_full=n_samples,
        n_display_bins=actual_display_bins,
        frequency_resolution_hz=round(frequency_resolution_hz, 6),
        nyquist_hz=nyquist_hz,
        duration_seconds=round(duration_seconds, 6),
        sample_rate_hz=sample_rate_hz,
        channels=channel_spectra,
    )
