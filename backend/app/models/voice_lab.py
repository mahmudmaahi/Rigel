"""
app/models/voice_lab.py — Module 09: Voice Laboratory API Models
=================================================================

Strict Pydantic models for the POST /api/audio/voice/process endpoint.

Processing chain (canonical, non-reorderable):
    INPUT → Gain → Speed → TimeStretch → PitchShift → Effect → OUTPUT

Each operation model carries explicit typed fields.  Arbitrary dicts
and **kwargs are deliberately NOT used.

Output safety policy
--------------------
- DSP output is returned as-is.
- If output peak > 1.0, clipping_risk=True is set in the response.
- If prevent_clipping=True in the request, peak normalization is applied
  as an explicit final step (not silently embedded in any DSP operation).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Operation models
# ---------------------------------------------------------------------------


class GainOperation(BaseModel):
    """Apply amplitude gain.

    mode:
        'linear'  → g = gain_value
        'db'      → g = 10^(gain_value / 20)
        'peak'    → normalize peak to gain_value (default 1.0)
        'rms'     → normalize RMS to gain_value (default 0.1)
    """
    op: Literal["gain"] = "gain"
    enabled: bool = True
    mode: Literal["linear", "db", "peak", "rms"] = "db"
    gain_value: float = 0.0      # 0 dB = unity for 'db' mode


class SpeedOperation(BaseModel):
    """Change playback speed using polyphase resampling.

    speed = 1.0  →  identity
    speed = 2.0  →  half duration, pitch doubled
    speed = 0.5  →  double duration, pitch halved
    """
    op: Literal["speed"] = "speed"
    enabled: bool = True
    speed: float = 1.0           # Must be > 0


class TimeStretchOperation(BaseModel):
    """Phase-vocoder time stretch (preserves pitch).

    stretch = T_output / T_input
    stretch = 1.0  →  identity
    stretch = 2.0  →  twice as long
    stretch = 0.5  →  half as long
    """
    op: Literal["time_stretch"] = "time_stretch"
    enabled: bool = True
    stretch: float = 1.0         # Must be > 0


class PitchShiftOperation(BaseModel):
    """Shift pitch while approximately preserving duration.

    semitones = 0    →  identity
    semitones = 12   →  one octave up (440 → 880 Hz)
    semitones = -12  →  one octave down (440 → 220 Hz)
    """
    op: Literal["pitch_shift"] = "pitch_shift"
    enabled: bool = True
    semitones: float = 0.0


class TremoloOperation(BaseModel):
    """Sinusoidal amplitude modulation.

    a[n] = (1 - depth) + depth * (1 + sin(2π f_LFO n / Fs)) / 2
    y[n] = x[n] * a[n]
    """
    op: Literal["tremolo"] = "tremolo"
    enabled: bool = True
    rate_hz: float = 5.0         # LFO frequency in Hz
    depth: float = 0.5           # Modulation depth in [0, 1]


class RingModulationOperation(BaseModel):
    """Multiply by a cosine carrier signal.

    y[n] = x[n] * cos(2π f_c n / Fs)
    """
    op: Literal["ring_modulation"] = "ring_modulation"
    enabled: bool = True
    carrier_hz: float = 440.0    # Carrier frequency in Hz


class EchoDelayOperation(BaseModel):
    """Recursive delay / echo effect.

    y[n] = x[n] + mix * feedback * y[n - D]
    """
    op: Literal["echo_delay"] = "echo_delay"
    enabled: bool = True
    delay_ms: float = 300.0      # Delay time in milliseconds
    feedback: float = 0.4        # Feedback gain [0, 1)
    mix: float = 0.5             # Wet/dry mix [0, 1]


class ChorusOperation(BaseModel):
    """Modulated delay line chorus.

    D[n] = base_delay_ms + depth_ms * sin(2π f_LFO n / Fs)
    y[n] = (1 - mix) * x[n] + mix * x_delayed[n]
    """
    op: Literal["chorus"] = "chorus"
    enabled: bool = True
    rate_hz: float = 1.5         # LFO rate in Hz
    depth_ms: float = 3.0        # Modulation depth in ms
    base_delay_ms: float = 15.0  # Base delay in ms (must be > depth_ms)
    mix: float = 0.5             # Wet/dry mix [0, 1]


class SoftDistortionOperation(BaseModel):
    """Tanh waveshaping distortion.

    y[n] = tanh(drive * x[n]) / tanh(drive)
    """
    op: Literal["soft_distortion"] = "soft_distortion"
    enabled: bool = True
    drive: float = 3.0           # Saturation drive β in (0, 100]



class ReverbOperation(BaseModel):
    """Schroeder Reverberator.
    """
    op: Literal["reverb"] = "reverb"
    enabled: bool = True
    room_size: float = 0.5       # Scale for delay lengths [0.1, 1.0]
    decay: float = 0.5           # Feedback coefficient [0.0, 0.99]
    wet: float = 0.3             # Wet/dry mix [0.0, 1.0]


# Discriminated union of all effect operations
AnyEffect = Annotated[
    Union[
        GainOperation,
        SpeedOperation,
        TimeStretchOperation,
        PitchShiftOperation,
        TremoloOperation,
        RingModulationOperation,
        EchoDelayOperation,
        ChorusOperation,
        SoftDistortionOperation,
        ReverbOperation,
    ],
    Field(discriminator="op"),
]


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class VoiceProcessRequest(BaseModel):
    """Request body for POST /api/audio/voice/process.

    The canonical processing chain is:
        Input → Gain → Speed → TimeStretch → PitchShift → Effect → Output

    Operations with neutral parameters are bypassed:
        gain: mode=db, gain_value=0.0
        speed: speed=1.0
        time_stretch: stretch=1.0
        pitch_shift: semitones=0.0
        effect: None

    All processing begins from the original uploaded audio.
    Results are never accumulated/chained from a previous call.

    prevent_clipping:
        If True, and output peak > 1.0, apply peak normalization as an
        explicit final step.  This is separate from the DSP operations.
    """
    # Samples as list-of-channels: [[ch0_s0, ch0_s1,...], [ch1_s0, ...]]
    samples: list[list[float]]
    sample_rate_hz: int

    effect_chain: list[AnyEffect] = Field(default_factory=list)

    prevent_clipping: bool = False


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class VoiceProcessResponse(BaseModel):
    """Response from POST /api/audio/voice/process.

    Contains:
    - Processed samples (list-of-channels, float)
    - Input/output metadata
    - Clipping/peak information
    - Operation summary
    """
    status: str
    sample_rate_hz: int

    # Per-channel samples [channels][samples]
    samples: list[list[float]]

    input_duration_s: float
    output_duration_s: float
    input_channels: int

    output_peak: float
    output_rms: float
    clipping_risk: bool           # True if output peak > 1.0 before any normalization
    clipping_prevented: bool      # True if prevent_clipping was applied

    operations_applied: list[str]
