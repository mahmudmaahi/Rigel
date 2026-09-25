"""
dsp_core/pitch_shift.py — Module 09 Phase D: Pitch Shifting
=============================================================

Shifts pitch by a specified number of semitones while approximately
preserving duration.

Mathematical pipeline
---------------------
Let:
    Δs  = semitone shift (positive = higher pitch)
    r   = 2^(Δs / 12)     — pitch ratio

To shift pitch by r while preserving duration:

    Step 1 — Time stretch by S = 1/r
        Duration becomes T/r.
        Pitch is unchanged (phase vocoder preserves instantaneous frequency).

    Step 2 — Resample at speed r
        Duration is restored: (T/r) · r = T   ✓
        All frequencies are multiplied by r:  f → f·r   ✓

Concrete examples
-----------------

    +12 semitones  →  r = 2.0
        Step 1: stretch by 0.5 → half duration, 440 Hz → 440 Hz
        Step 2: speed 2.0      → restore duration, 440 Hz → 880 Hz

    -12 semitones  →  r = 0.5
        Step 1: stretch by 2.0 → double duration, 440 Hz → 440 Hz
        Step 2: speed 0.5      → restore duration, 440 Hz → 220 Hz

    +1 semitone    →  r ≈ 1.05946
        Step 1: stretch by ≈ 0.9439 → slightly shorter, 440 Hz → 440 Hz
        Step 2: speed ≈ 1.05946    → restore duration, 440 Hz → 466.16 Hz

Duration is approximately preserved; small residual errors arise from
rational approximation in the resampler and phase-vocoder rounding.

Known limitations
-----------------
- Phase-vocoder artefacts (transient smearing, phasiness) apply at the
  time-stretch stage and carry through into the pitch-shifted output.
- Pitch accuracy depends on the quality of the phase-vocoder's frequency
  estimation; polyphonic signals may show blurring.
- For large shifts (> ±12 semitones) artefacts become more audible.
"""

from __future__ import annotations

import math

import numpy as np

from dsp_core.voice_gain import VoiceDSPError, _require_finite_input, _require_finite_output
from dsp_core.time_scale import apply_time_stretch, _apply_raw_resampling, PV_N, PV_HA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEMITONES_PER_OCTAVE: int = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def semitones_to_ratio(semitones: float) -> float:
    """Convert a semitone interval to a linear frequency ratio.

    r = 2^(semitones / 12)

    Examples
    --------
    >>> round(semitones_to_ratio(12), 6)
    2.0
    >>> round(semitones_to_ratio(-12), 6)
    0.5
    >>> round(semitones_to_ratio(1), 6)
    1.059463        # approximately 440 → 466.16 Hz
    """
    return 2.0 ** (semitones / SEMITONES_PER_OCTAVE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_pitch_shift(
    samples: np.ndarray,
    sample_rate_hz: int,
    semitones: float,
    n_fft: int = PV_N,
    hop_analysis: int = PV_HA,
) -> np.ndarray:
    """Shift pitch by *semitones* while approximately preserving duration.

    Pipeline:
        1. Compute r = 2^(semitones/12).
        2. Phase-vocoder time stretch by S = 1/r.
        3. Polyphase resample at speed r.

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel [N, C].
    sample_rate_hz:
        Sample rate in Hz (used for validation and resampler).
    semitones:
        Number of semitones to shift.  Positive = higher pitch.
        Range: (−24, +24) is practical; extreme values increase artefacts.
        Must be finite.
    n_fft:
        Phase-vocoder FFT window size (default 2048).
    hop_analysis:
        Phase-vocoder analysis hop (default 512).

    Returns
    -------
    float64 array approximately the same length as *samples*.

    Raises
    ------
    ValueError:
        If *semitones* is NaN or Inf.
    VoiceDSPError:
        If input or output contains NaN/Inf.
    """
    if not math.isfinite(semitones):
        raise ValueError(f"semitones must be a finite number; got {semitones!r}.")

    x = samples.astype(np.float64, copy=False)
    _require_finite_input(x)

    # Zero shift: identity pass-through
    if abs(semitones) < 1e-9:
        return x.copy()

    r = semitones_to_ratio(semitones)
    stretch = r   # Step 1: stretch by r to lengthen (then speed r restores it)

    # Step 1: time stretch (duration ×r, pitch unchanged)
    stretched = apply_time_stretch(x, sample_rate_hz, stretch, n_fft, hop_analysis)

    # Step 2: resample at speed r (duration /r, all frequencies ×r)
    shifted = _apply_raw_resampling(stretched, sample_rate_hz, r)

    _require_finite_output(shifted, "pitch_shift")
    return shifted
