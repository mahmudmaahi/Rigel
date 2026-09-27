"""
app/services/voice_lab_service.py — Module 09: Voice Laboratory Service
========================================================================

Orchestrates the Voice Lab processing chain:

    Input → Gain → Speed → TimeStretch → PitchShift → Effect → Output

CRITICAL INVARIANT: Every recomputation begins from the original input
array passed in the request.  This function is pure: it never reads or
writes any mutable state outside its arguments.

FastAPI layer passes validated request data here; DSP modules perform
the mathematics.  This service converts between the API surface (lists
of floats, Pydantic models) and the DSP layer (NumPy arrays).
"""

from __future__ import annotations

import numpy as np

from app.models.voice_lab import (
    VoiceProcessRequest,
    VoiceProcessResponse,
)
from dsp_core.voice_gain import (
    apply_db_gain,
    apply_linear_gain,
    apply_peak_normalization,
    apply_rms_normalization,
    measure_peak,
    measure_rms,
    VoiceDSPError,
)
from dsp_core.time_scale import apply_speed, apply_time_stretch
from dsp_core.pitch_shift import apply_pitch_shift
from dsp_core.effects import (
    apply_delay,
    apply_chorus,
    apply_soft_distortion,
    apply_reverb,
)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _samples_to_numpy(samples: list[list[float]]) -> np.ndarray:
    """Convert list-of-channels to a [N] or [N, C] float64 NumPy array.

    Input:  [[ch0_s0, ch0_s1, ...], [ch1_s0, ...]]   (C channels × N samples)
    Output: shape [N] if C=1, else [N, C].
    """
    arr = np.array(samples, dtype=np.float64)   # shape [C, N]
    if arr.shape[0] == 1:
        return arr[0]                             # mono: [N]
    return arr.T                                  # stereo: [N, C]


def _numpy_to_samples(arr: np.ndarray) -> list[list[float]]:
    """Convert [N] or [N, C] NumPy array to list-of-channels."""
    if arr.ndim == 1:
        return [arr.tolist()]
    return [arr[:, c].tolist() for c in range(arr.shape[1])]


# ---------------------------------------------------------------------------
# Neutral-parameter bypass checks
# ---------------------------------------------------------------------------

_SPEED_IDENTITY_TOL = 1e-9
_STRETCH_IDENTITY_TOL = 1e-9
_SEMITONES_IDENTITY_TOL = 1e-9


def _is_speed_neutral(speed: float) -> bool:
    return abs(speed - 1.0) < _SPEED_IDENTITY_TOL


def _is_stretch_neutral(stretch: float) -> bool:
    return abs(stretch - 1.0) < _STRETCH_IDENTITY_TOL


def _is_pitch_neutral(semitones: float) -> bool:
    return abs(semitones) < _SEMITONES_IDENTITY_TOL


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def process_voice_lab(req: VoiceProcessRequest) -> VoiceProcessResponse:
    """Execute the Voice Lab processing chain from the original input.

    Chain order (deterministic, sequential):
        Effect 1 → Effect 2 → Effect 3 → ...

    Returns a VoiceProcessResponse with processed samples, metadata,
    and clipping information.

    Raises
    ------
    VoiceDSPError:
        If any DSP operation produces NaN/Inf output.
    ValueError:
        If any operation parameter is out of range.
    """
    # Convert input to numpy — this is the ORIGINAL that we always start from.
    x = _samples_to_numpy(req.samples)
    sr = req.sample_rate_hz
    input_samples = x.shape[0]
    input_channels = 1 if x.ndim == 1 else x.shape[1]
    input_duration = input_samples / sr

    current = x.copy()
    applied: list[str] = []

    # ---------------------------------------------------------- Effect Chain
    for effect in req.effect_chain:
        if not effect.enabled:
            applied.append(f"{effect.op}(bypassed)")
            continue

        op = effect.op

        if op == "gain":
            if effect.mode == "linear":
                current = apply_linear_gain(current, effect.gain_value)
                applied.append(f"gain(linear, {effect.gain_value:.4g})")
            elif effect.mode == "db":
                if abs(effect.gain_value) > 1e-9:
                    current = apply_db_gain(current, effect.gain_value)
                    applied.append(f"gain({effect.gain_value:+.2f} dB)")
                else:
                    applied.append("gain(0dB)")
            elif effect.mode == "peak":
                current = apply_peak_normalization(current, effect.gain_value)
                applied.append(f"peak_norm(target={effect.gain_value:.4g})")
            elif effect.mode == "rms":
                current = apply_rms_normalization(current, effect.gain_value)
                applied.append(f"rms_norm(target={effect.gain_value:.4g})")

        elif op == "speed":
            if not _is_speed_neutral(effect.speed):
                current = apply_speed(current, sr, effect.speed)
                applied.append(f"speed({effect.speed:.4g}x)")
            else:
                applied.append("speed(1.0x)")

        elif op == "time_stretch":
            if not _is_stretch_neutral(effect.stretch):
                current = apply_time_stretch(current, sr, effect.stretch)
                applied.append(f"time_stretch({effect.stretch:.4g}x)")
            else:
                applied.append("time_stretch(1.0x)")

        elif op == "pitch_shift":
            if not _is_pitch_neutral(effect.semitones):
                current = apply_pitch_shift(current, sr, effect.semitones)
                applied.append(f"pitch_shift({effect.semitones:+.2f} semitones)")
            else:
                applied.append("pitch_shift(0 st)")

        elif op == "echo_delay":
            current = apply_delay(current, sr, effect.delay_ms, effect.feedback, effect.mix)
            applied.append(f"echo_delay({effect.delay_ms}ms, fb={effect.feedback}, mix={effect.mix})")

        elif op == "chorus":
            current = apply_chorus(
                current, sr, effect.rate_hz, effect.depth_ms, effect.base_delay_ms, effect.mix
            )
            applied.append(
                f"chorus(rate={effect.rate_hz}Hz, depth={effect.depth_ms}ms, "
                f"base={effect.base_delay_ms}ms, mix={effect.mix})"
            )

        elif op == "soft_distortion":
            current = apply_soft_distortion(current, effect.drive)
            applied.append(f"soft_distortion(drive={effect.drive})")

        elif op == "reverb":
            current = apply_reverb(current, sr, effect.room_size, effect.decay, effect.wet)
            applied.append(f"reverb(room={effect.room_size:.2f}, decay={effect.decay:.2f}, wet={effect.wet:.2f})")

    # ------------------------------------------------- Measure output quality
    raw_peak = measure_peak(current)
    raw_rms = measure_rms(current)
    clipping_risk = raw_peak > 1.0
    clipping_prevented = False

    # Explicit clipping prevention (only if requested — never silent)
    if req.prevent_clipping and clipping_risk:
        current = apply_peak_normalization(current, 1.0)
        clipping_prevented = True
        applied.append("prevent_clipping(peak_norm → 1.0)")

    output_samples = current.shape[0]
    output_duration = output_samples / sr

    return VoiceProcessResponse(
        status="processed",
        sample_rate_hz=sr,
        samples=_numpy_to_samples(current),
        input_duration_s=round(input_duration, 6),
        output_duration_s=round(output_duration, 6),
        input_channels=input_channels,
        output_peak=round(float(raw_peak), 8),
        output_rms=round(float(raw_rms), 8),
        clipping_risk=clipping_risk,
        clipping_prevented=clipping_prevented,
        operations_applied=applied,
    )
