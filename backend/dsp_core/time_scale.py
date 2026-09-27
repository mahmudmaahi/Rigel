"""
dsp_core/time_scale.py — Module 09 Phase B–C: Speed and Phase-Vocoder Time Stretch
======================================================================================

Phase B — Speed / Resampling
------------------------------
Changes playback speed, which simultaneously alters duration and pitch.

Convention:
    speed = output_playback_rate / original_playback_rate

    speed = 1.0  →  unchanged  (duration = T, pitch = f)
    speed = 2.0  →  half duration, pitch doubled
    speed = 0.5  →  double duration, pitch halved

Implementation:
    scipy.signal.resample_poly with polyphase FIR anti-aliasing.
    Arbitrary float speed ratios are approximated by a rational p/q
    fraction using Python's fractions.Fraction with a bounded denominator
    of 1000 to keep filter length manageable.

Phase C — Phase Vocoder Time Stretch
--------------------------------------
Changes duration while approximately preserving pitch.

Convention:
    S = T_output / T_input

    S = 1.0  →  unchanged
    S = 2.0  →  twice as long, pitch preserved
    S = 0.5  →  half as long, pitch preserved

Implementation (from first principles — no librosa):
    1. Frame the signal with a Hann analysis window.
    2. Compute complex STFT (np.fft.rfft per frame).
    3. Track instantaneous frequency via phase difference.
    4. Accumulate synthesis phases using a stretched synthesis hop.
    5. Reconstruct via overlap-add with synthesis window normalization.

Phase-vocoder mathematics
--------------------------
For each STFT bin k:

    ω_k = 2π k / N                         — bin centre frequency (rad/sample)
    ΔΦ_expected = ω_k · Ha                  — expected phase advance over Ha samples

    ΔΦ_actual = φ[k,m] − φ[k,m−1]          — actual phase difference

    δφ = wrap(ΔΦ_actual − ΔΦ_expected)      — deviation from expected, in (−π, π]

    ω_true = ω_k + δφ / Ha                  — instantaneous frequency (rad/sample)

    Ψ[k,m] = Ψ[k,m−1] + ω_true · Hs        — synthesis phase accumulator

Output frame:
    X_synth[k,m] = |X[k,m]| · exp(j Ψ[k,m])

Overlap-add with synthesis Hann window, normalized by the sum of squared
overlapping windows so amplitude remains unity across the output.

Known limitations:
    - Transient smearing: Phase vocoder cannot maintain transient sharpness
      (attack of a drum hit, consonants in speech) across stretch factors.
    - Phasiness / metallic colouring on polyphonic material.
    - Phase vocoder is well-suited for slowly evolving tonal signals; artefacts
      are more prominent on percussive or highly transient material.
    - The identity case S=1.0 should reconstruct approximately but may show
      small numerical error from floating-point accumulation over many frames.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from dsp_core.voice_gain import VoiceDSPError, _require_finite_input, _require_finite_output

# ---------------------------------------------------------------------------
# Constants for the Phase Vocoder
# ---------------------------------------------------------------------------

PV_N: int = 2048   # FFT / analysis window length
PV_HA: int = 512   # analysis hop length  (75 % overlap)


# ---------------------------------------------------------------------------
# Phase B — Speed / Resampling
# ---------------------------------------------------------------------------


def _speed_to_rational(speed: float, max_denominator: int = 1000) -> tuple[int, int]:
    """Convert a float speed factor to an integer up/down ratio for resample_poly.

    resample_poly(x, up, down) produces output of length  len(x) * up / down.

    For a speed change we want:
        output_length = input_length / speed
    Therefore:
        up / down = 1 / speed
    So:  up = denominator(speed),  down = numerator(speed)

    Examples
    --------
    speed = 2.0  →  Fraction(2, 1)  →  (up=1, down=2)  →  half length  ✓
    speed = 0.5  →  Fraction(1, 2)  →  (up=2, down=1)  →  double length ✓
    speed = 1.25 →  Fraction(5, 4)  →  (up=4, down=5)  →  0.8× length  ✓

    Parameters
    ----------
    speed:
        Playback speed ratio (> 0).
    max_denominator:
        Maximum allowed denominator in the rational approximation.

    Returns
    -------
    (up, down) integers for resample_poly.
    """
    frac = Fraction(speed).limit_denominator(max_denominator)
    # up = 1/speed's numerator relationship → denom of speed, numer of speed
    return int(frac.denominator), int(frac.numerator)


def _apply_raw_resampling(
    samples: np.ndarray,
    sample_rate_hz: int,
    speed: float,
) -> np.ndarray:
    """Change playback speed using high-quality polyphase resampling.

    Duration and pitch change proportionally.

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel [N, C].
    sample_rate_hz:
        Sample rate of the audio in Hz.  Used for validation only;
        the output has the same sample rate (pitch is embedded in data).
    speed:
        Output playback speed relative to original.
        Must be finite and > 0.  Typical range: 0.25–4.0.

    Returns
    -------
    float64 array.  Shape: [M] or [M, C] where M ≈ N / speed.

    Raises
    ------
    ValueError:
        If *speed* is ≤ 0, NaN, or Inf.
    VoiceDSPError:
        If input or output contains NaN/Inf.
    """
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"speed must be a finite positive number; got {speed!r}.")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}.")

    x = samples.astype(np.float64, copy=False)
    _require_finite_input(x)

    # Unity: skip resampling to avoid any numerical drift
    if abs(speed - 1.0) < 1e-9:
        return x.copy()

    up, down = _speed_to_rational(speed)

    mono = x.ndim == 1
    channels = [x] if mono else [x[:, c] for c in range(x.shape[1])]

    resampled_channels = []
    for ch in channels:
        # resample_poly applies a windowed-FIR polyphase filter for anti-aliasing
        y_ch = resample_poly(ch, up, down).astype(np.float64)
        resampled_channels.append(y_ch)

    y = resampled_channels[0] if mono else np.stack(resampled_channels, axis=1)
    _require_finite_output(y, "speed")
    return y


# ---------------------------------------------------------------------------
# Phase C — Phase Vocoder Time Stretch (from first principles)
# ---------------------------------------------------------------------------


def _hann_window(length: int) -> np.ndarray:
    """Return a periodic Hann window of the given length (float64)."""
    return np.hanning(length).astype(np.float64)


def _wrap_phase(delta: np.ndarray) -> np.ndarray:
    """Wrap phase differences into the principal interval (−π, π]."""
    return ((delta + np.pi) % (2.0 * np.pi)) - np.pi


def _phase_vocoder_channel(
    signal: np.ndarray,
    stretch: float,
    n_fft: int = PV_N,
    hop_analysis: int = PV_HA,
) -> np.ndarray:
    """Phase-vocoder time stretch for one channel (1-D float64 array).

    Parameters
    ----------
    signal:
        1-D float64 array.
    stretch:
        S = T_output / T_input.
    n_fft:
        Analysis FFT size N.
    hop_analysis:
        Analysis hop Ha.

    Returns
    -------
    1-D float64 array of approximate length len(signal) * stretch.
    """
    n = len(signal)
    hop_synth_float = stretch * hop_analysis
    hop_synth = int(round(hop_synth_float))
    if hop_synth < 1:
        hop_synth = 1

    win = _hann_window(n_fft)
    n_bins = n_fft // 2 + 1

    # Bin centre frequencies in radians/sample
    omega_k = 2.0 * np.pi * np.arange(n_bins) / n_fft  # shape [n_bins]

    # Expected phase advance per analysis hop for each bin
    expected_advance = omega_k * hop_analysis  # shape [n_bins]

    # -----------------------------------------------------------------------
    # 1. Analysis: compute complex STFT frame by frame
    # -----------------------------------------------------------------------
    # Zero-pad signal so the last frame is fully covered
    pad = n_fft  # extra padding to handle tail cleanly
    padded = np.concatenate([np.zeros(n_fft // 2), signal, np.zeros(pad)])
    n_padded = len(padded)

    # Frame count
    n_frames_analysis = 1 + (n_padded - n_fft) // hop_analysis

    # Store complex STFT frames
    stft_matrix = np.zeros((n_frames_analysis, n_bins), dtype=np.complex128)

    for m in range(n_frames_analysis):
        start = m * hop_analysis
        frame = padded[start : start + n_fft]
        stft_matrix[m] = np.fft.rfft(frame * win)

    # -----------------------------------------------------------------------
    # 2. Phase-vocoder: compute synthesis phases frame by frame
    # -----------------------------------------------------------------------
    # Phase accumulators and previous-frame phase
    synth_phase = np.angle(stft_matrix[0])   # initialize from first frame
    prev_phase = np.angle(stft_matrix[0])

    # Estimate output length and allocate overlap-add buffer
    n_output_est = int(np.ceil(n_frames_analysis * hop_synth + n_fft))
    output_buf = np.zeros(n_output_est, dtype=np.float64)
    norm_buf = np.zeros(n_output_est, dtype=np.float64)

    win_sq = win ** 2  # for overlap-add normalization

    for m in range(n_frames_analysis):
        mag = np.abs(stft_matrix[m])
        curr_phase = np.angle(stft_matrix[m])

        if m == 0:
            synth_phase = curr_phase.copy()
        else:
            # Phase difference between consecutive analysis frames
            delta_phi = curr_phase - prev_phase
            # Subtract expected advance and wrap to (−π, π]
            deviation = _wrap_phase(delta_phi - expected_advance)
            # Instantaneous frequency (radians/sample)
            omega_true = omega_k + deviation / hop_analysis
            # Advance synthesis phase accumulator
            synth_phase += omega_true * hop_synth

        # Build output frame: original magnitude, new synthesis phase
        X_synth = mag * np.exp(1j * synth_phase)
        frame_out = np.fft.irfft(X_synth, n=n_fft) * win

        # Overlap-add into output buffer
        out_start = m * hop_synth
        out_end = out_start + n_fft
        if out_end <= n_output_est:
            output_buf[out_start:out_end] += frame_out
            norm_buf[out_start:out_end] += win_sq

        prev_phase = curr_phase

    # -----------------------------------------------------------------------
    # 3. Normalize by the overlapping window power
    # -----------------------------------------------------------------------
    nonzero = norm_buf > 1e-12
    output_buf[nonzero] /= norm_buf[nonzero]

    # -----------------------------------------------------------------------
    # 4. Trim to expected output length, accounting for the centering offset
    # -----------------------------------------------------------------------
    # The n_fft//2 pre-padding shifts output by hop_synth frames; trim it.
    offset = int(round((n_fft // 2) * stretch))
    expected_out_len = int(round(n * stretch))

    output = output_buf[offset : offset + expected_out_len]

    # Pad to exact target length if trimming overshot
    if len(output) < expected_out_len:
        output = np.concatenate([output, np.zeros(expected_out_len - len(output))])

    return output


def apply_time_stretch(
    samples: np.ndarray,
    sample_rate_hz: int,
    stretch: float,
    n_fft: int = PV_N,
    hop_analysis: int = PV_HA,
) -> np.ndarray:
    """Time-stretch audio using the phase vocoder.

    Changes duration while approximately preserving pitch.

    Convention:
        S = T_output / T_input
        S = 1.0  →  unchanged
        S = 2.0  →  twice as long
        S = 0.5  →  half duration

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel [N, C].
    sample_rate_hz:
        Sample rate in Hz.  Used for validation only.
    stretch:
        Stretch factor S.  Must be finite and > 0.  Practical range: 0.25–4.0.
    n_fft:
        Analysis FFT window size (default 2048).
    hop_analysis:
        Analysis hop Ha (default 512).

    Returns
    -------
    float64 array.  Shape: [M] or [M, C] where M ≈ N * stretch.

    Raises
    ------
    ValueError:
        If *stretch* ≤ 0, NaN, or Inf; or if n_fft < 4; or hop_analysis < 1.
    VoiceDSPError:
        If input or output contains NaN/Inf.
    """
    if not np.isfinite(stretch) or stretch <= 0.0:
        raise ValueError(f"stretch must be a finite positive number; got {stretch!r}.")
    if n_fft < 4 or n_fft % 2 != 0:
        raise ValueError(f"n_fft must be an even integer ≥ 4; got {n_fft}.")
    if hop_analysis < 1:
        raise ValueError(f"hop_analysis must be ≥ 1; got {hop_analysis}.")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}.")

    x = samples.astype(np.float64, copy=False)
    _require_finite_input(x)

    mono = x.ndim == 1
    channels = [x] if mono else [x[:, c] for c in range(x.shape[1])]

    stretched_channels = []
    for ch in channels:
        y_ch = _phase_vocoder_channel(ch, stretch, n_fft, hop_analysis)
        stretched_channels.append(y_ch)

    y = stretched_channels[0] if mono else np.stack(stretched_channels, axis=1)
    _require_finite_output(y, "time_stretch")
    return y


def apply_speed(
    samples: np.ndarray,
    sample_rate_hz: int,
    speed: float,
) -> np.ndarray:
    """Change playback speed using polyphase resampling (tape-style).

    Duration and pitch change together, proportionally — this is the
    classic "tape speed" effect, distinct from Time Stretch (which
    preserves pitch). A thin wrapper around `_apply_raw_resampling`.

    Convention:
        speed = 1.0  ->  unchanged  (duration = T, pitch = f)
        speed = 2.0  ->  half duration, pitch doubled
        speed = 0.5  ->  double duration, pitch halved

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel [N, C].
    sample_rate_hz:
        Sample rate in Hz.
    speed:
        Speed factor. Must be finite and > 0. Practical range: 0.25–4.0.

    Returns
    -------
    float64 array. Shape: [M] or [M, C] where M ≈ N / speed.
    """
    return _apply_raw_resampling(samples, sample_rate_hz, speed)
