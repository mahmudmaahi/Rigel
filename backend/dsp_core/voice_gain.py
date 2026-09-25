"""
dsp_core/voice_gain.py — Module 09 Phase A: Gain and Level
============================================================

Implements amplitude transformations on audio signals:

    1. Linear gain:        y[n] = g · x[n]
    2. dB gain:            g = 10^(G_dB / 20),  then linear gain
    3. Peak normalization: g = target_peak / max(|x|)
    4. RMS normalization:  g = target_rms / RMS(x)

All functions operate on float64 NumPy arrays (mono 1-D or
multi-channel 2-D [N, C]).  Integer inputs are rejected; callers must
normalize to float before calling (use stft._normalize_samples if
loading from WAV).

Numerical-safety contract
--------------------------
- Finite input is verified before processing.
- Finite output is verified after processing.
- NaN or Inf in the output raises VoiceDSPError — never silently zeroed.
- Division by zero is detected and raised explicitly.
- Output values above 1.0 in magnitude are ALLOWED and returned as-is;
  peak clipping is the caller's responsibility via the explicit
  prevent_clipping chain option, not a hidden side-effect here.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# DSP-level error
# ---------------------------------------------------------------------------


class VoiceDSPError(RuntimeError):
    """Raised when a Voice Lab DSP computation fails or produces invalid output.

    Distinct from ValueError (bad parameters) so callers can distinguish
    mathematical failures from API contract violations.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_finite_input(x: np.ndarray, name: str = "input") -> None:
    """Raise VoiceDSPError if *x* contains NaN or Inf."""
    if not np.all(np.isfinite(x)):
        raise VoiceDSPError(
            f"The {name} array contains NaN or Inf values. "
            "Check the audio source before calling Voice Lab DSP functions."
        )


def _require_finite_output(y: np.ndarray, op: str) -> None:
    """Raise VoiceDSPError if the output of operation *op* contains NaN/Inf."""
    if not np.all(np.isfinite(y)):
        raise VoiceDSPError(
            f"Voice Lab '{op}' produced NaN or Inf in the output array. "
            "This is a DSP-level bug; inspect the input and parameters."
        )


def _to_float64(x: np.ndarray) -> np.ndarray:
    """Return x as float64 without modifying the caller's array."""
    return x.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_linear_gain(samples: np.ndarray, gain: float) -> np.ndarray:
    """Apply a linear amplitude gain factor.

    y[n] = gain · x[n]

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel 2-D [N, C].
        Must already be in floating-point representation.
    gain:
        Dimensionless linear multiplier.  May be < 1 (attenuation),
        = 1 (unity), or > 1 (amplification).  Must be finite and ≥ 0.

    Returns
    -------
    float64 array of same shape as *samples*.

    Raises
    ------
    ValueError:
        If *gain* is negative, NaN, or Inf.
    VoiceDSPError:
        If the input or computed output contains NaN/Inf.
    """
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError(f"gain must be a finite non-negative number; got {gain!r}.")
    x = _to_float64(samples)
    _require_finite_input(x)
    y = gain * x
    _require_finite_output(y, "linear_gain")
    return y


def apply_db_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a gain expressed in decibels.

    Conversion:
        g = 10^(G_dB / 20)

    A 0 dB gain is unity (no change).
    A +6 dB gain approximately doubles amplitude.
    A -6 dB gain approximately halves amplitude.

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel 2-D [N, C].
    gain_db:
        Gain in dBFS.  Range: (−∞, +∞).  Must be finite.

    Returns
    -------
    float64 array of same shape.

    Raises
    ------
    ValueError:
        If *gain_db* is NaN or Inf.
    VoiceDSPError:
        If input or output contains NaN/Inf.
    """
    if not np.isfinite(gain_db):
        raise ValueError(f"gain_db must be a finite number; got {gain_db!r}.")
    g = 10.0 ** (gain_db / 20.0)
    return apply_linear_gain(samples, g)


def apply_peak_normalization(
    samples: np.ndarray,
    target_peak: float = 1.0,
) -> np.ndarray:
    """Normalize so that the maximum absolute sample equals *target_peak*.

    g = target_peak / max(|x|)
    y[n] = g · x[n]

    All-zero input is returned unchanged (cannot normalize silence).

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel 2-D [N, C].
    target_peak:
        Desired peak amplitude.  Default 1.0 (full-scale normalized).
        Must be > 0 and finite.

    Returns
    -------
    float64 array of same shape.

    Raises
    ------
    ValueError:
        If *target_peak* ≤ 0 or not finite.
    VoiceDSPError:
        If input or output contains NaN/Inf.
    """
    if not np.isfinite(target_peak) or target_peak <= 0.0:
        raise ValueError(
            f"target_peak must be a positive finite number; got {target_peak!r}."
        )
    x = _to_float64(samples)
    _require_finite_input(x)

    peak = float(np.max(np.abs(x)))
    if peak == 0.0:
        # All-zero signal — cannot normalize; return as-is.
        return x.copy()

    g = target_peak / peak
    y = g * x
    _require_finite_output(y, "peak_normalization")
    return y


def apply_rms_normalization(
    samples: np.ndarray,
    target_rms: float = 0.1,
) -> np.ndarray:
    """Normalize so that the RMS amplitude equals *target_rms*.

    RMS(x) = sqrt( mean(x²) )
    g      = target_rms / RMS(x)
    y[n]   = g · x[n]

    All-zero input is returned unchanged.

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel 2-D [N, C].
    target_rms:
        Desired RMS amplitude.  Default 0.1.  Must be > 0 and finite.

    Returns
    -------
    float64 array of same shape.

    Raises
    ------
    ValueError:
        If *target_rms* ≤ 0 or not finite.
    VoiceDSPError:
        If input or output contains NaN/Inf.
    """
    if not np.isfinite(target_rms) or target_rms <= 0.0:
        raise ValueError(
            f"target_rms must be a positive finite number; got {target_rms!r}."
        )
    x = _to_float64(samples)
    _require_finite_input(x)

    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms == 0.0:
        # All-zero signal — cannot normalize; return as-is.
        return x.copy()

    g = target_rms / rms
    y = g * x
    _require_finite_output(y, "rms_normalization")
    return y


# ---------------------------------------------------------------------------
# Measurement helpers (used by the API layer to report clipping risk)
# ---------------------------------------------------------------------------


def measure_peak(samples: np.ndarray) -> float:
    """Return the maximum absolute sample value across all channels."""
    return float(np.max(np.abs(samples.astype(np.float64))))


def measure_rms(samples: np.ndarray) -> float:
    """Return the RMS amplitude across all channels."""
    x = samples.astype(np.float64)
    return float(np.sqrt(np.mean(x ** 2)))
