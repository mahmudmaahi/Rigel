"""
dsp_core/effects.py — Module 09 Phase E: Classical Audio Effects
==================================================================

Implements classical time-domain / frequency-domain audio effects:

    1. Delay / Echo     — recursive comb filter
    2. Chorus           — modulated delay line with linear interpolation
    3. Soft Distortion  — tanh-based waveshaping
    4. Schroeder Reverb  — parallel comb filters + serial allpass filters

Design principles
-----------------
- Every function accepts mono 1-D or multi-channel 2-D [N, C] float arrays.
- Neutral/identity parameters produce output identical (or numerically
  negligible) to the input — documented per function.
- Finite input is validated; finite output is verified.
- np.nan_to_num is NOT used as a bug mask.
- Output may exceed ±1.0 in amplitude (no silent normalization/clipping).
- Stereo signals are processed channel-by-channel to avoid cross-channel
  coupling in time-varying delay lines.
"""

from __future__ import annotations

import numpy as np

from dsp_core.voice_gain import VoiceDSPError, _require_finite_input, _require_finite_output


# ---------------------------------------------------------------------------
# 1. Delay / Echo
# ---------------------------------------------------------------------------


def apply_delay(
    samples: np.ndarray,
    sample_rate_hz: int,
    delay_ms: float,
    feedback: float,
    mix: float = 0.5,
) -> np.ndarray:
    """Apply a recursive comb filter (delay / echo effect).

    Difference equation (single-channel):
        y[n] = x[n] + mix · y[n - D]   where D = round(delay_ms · Fs / 1000)

    The recursive feedback path is:
        y[n - D] is the previously written output sample D samples ago.

    Stability condition: |feedback · mix| < 1.

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel [N, C].
    sample_rate_hz:
        Audio sample rate in Hz.
    delay_ms:
        Delay time in milliseconds.  Must be > 0 and ≤ 5000 ms.
    feedback:
        Gain of the delayed signal fed back into the delay line.
        Range: [0, 0.99].  feedback = 0 → single echo, no repeats.
        feedback = 0.99 → many decaying repeats.
    mix:
        Mix of the delayed signal.  Range: [0, 1].  Default 0.5.
        mix = 0 → dry (identity), mix = 1 → full wet.

    Returns
    -------
    float64 array of same shape as *samples*.
    """
    if not np.isfinite(delay_ms) or delay_ms <= 0.0 or delay_ms > 5000.0:
        raise ValueError(f"delay_ms must be in (0, 5000]; got {delay_ms!r}.")
    if not np.isfinite(feedback) or feedback < 0.0 or feedback >= 1.0:
        raise ValueError(f"feedback must be in [0, 1); got {feedback!r}.")
    if not np.isfinite(mix) or mix < 0.0 or mix > 1.0:
        raise ValueError(f"mix must be in [0, 1]; got {mix!r}.")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}.")

    x = samples.astype(np.float64, copy=False)
    _require_finite_input(x)

    delay_samples = int(round(delay_ms * sample_rate_hz / 1000.0))
    if delay_samples < 1:
        delay_samples = 1

    mono = x.ndim == 1
    channels = [x] if mono else [x[:, c] for c in range(x.shape[1])]

    # y[n] = x[n] + mix * y[n-D]
    processed = []
    for ch in channels:
        n_samples = len(ch)
        y = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            delayed = y[i - delay_samples] if i >= delay_samples else 0.0
            y[i] = ch[i] + mix * feedback * delayed
        processed.append(y)

    y_out = processed[0] if mono else np.stack(processed, axis=1)
    _require_finite_output(y_out, "delay")
    return y_out


# ---------------------------------------------------------------------------
# 2. Chorus
# ---------------------------------------------------------------------------


def _linear_interp(buf: np.ndarray, idx: float) -> float:
    """Read a value from *buf* at fractional index using linear interpolation.

    Parameters
    ----------
    buf:
        1-D float64 array representing the delay buffer.
    idx:
        Fractional read index into *buf*.  Must be in [0, len(buf) - 1].

    Returns
    -------
    Linearly interpolated float value.
    """
    i0 = int(idx)
    frac = idx - i0
    i1 = min(i0 + 1, len(buf) - 1)
    return (1.0 - frac) * buf[i0] + frac * buf[i1]


def _apply_chorus_channel(
    ch: np.ndarray,
    sample_rate_hz: int,
    rate_hz: float,
    depth_ms: float,
    base_delay_ms: float,
    mix: float,
) -> np.ndarray:
    """Apply chorus to a single channel."""
    n_samples = len(ch)
    # Convert ms to samples
    base_d = base_delay_ms * sample_rate_hz / 1000.0
    mod_d = depth_ms * sample_rate_hz / 1000.0

    # Buffer size must accommodate maximum possible delay
    max_delay_samples = int(np.ceil(base_d + mod_d)) + 2
    buf = np.zeros(max_delay_samples, dtype=np.float64)

    y = np.zeros(n_samples, dtype=np.float64)
    write_pos = 0

    for i in range(n_samples):
        # Write current sample to circular buffer
        buf[write_pos % max_delay_samples] = ch[i]

        # Compute time-varying delay in samples
        lfo = np.sin(2.0 * np.pi * rate_hz * i / sample_rate_hz)
        delay_samp = base_d + mod_d * lfo

        # Read position (going back in time from write_pos)
        read_pos_raw = write_pos - delay_samp
        # Wrap read position to buffer range
        read_pos = read_pos_raw % max_delay_samples
        # Linear interpolation across buffer boundary
        i0 = int(read_pos)
        frac = read_pos - i0
        i0 = i0 % max_delay_samples
        i1 = (i0 + 1) % max_delay_samples
        delayed = (1.0 - frac) * buf[i0] + frac * buf[i1]

        y[i] = (1.0 - mix) * ch[i] + mix * delayed
        write_pos += 1

    return y


def apply_chorus(
    samples: np.ndarray,
    sample_rate_hz: int,
    rate_hz: float = 1.5,
    depth_ms: float = 3.0,
    base_delay_ms: float = 15.0,
    mix: float = 0.5,
) -> np.ndarray:
    """Apply a classic chorus effect using a sinusoidally modulated delay.

    The delay time varies as:
        D[n] = D_base + D_mod · sin(2π · f_LFO · n / Fs)

    A fractional sample at D[n] is read using linear interpolation.

    Output (per sample):
        y[n] = (1 - mix) · x[n] + mix · x_delayed[n]

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel [N, C].
    sample_rate_hz:
        Audio sample rate in Hz.
    rate_hz:
        LFO rate in Hz.  Range: (0, 10] Hz.  Default 1.5 Hz.
    depth_ms:
        Modulation depth in milliseconds (half the total delay swing).
        Range: [0, 20] ms.  Default 3.0 ms.
    base_delay_ms:
        Base (center) delay time in milliseconds.
        Must be > depth_ms to avoid negative delay.  Range: (0, 50] ms.  Default 15 ms.
    mix:
        Wet/dry mix.  Range [0, 1].  0 = dry, 1 = fully wet.  Default 0.5.

    Returns
    -------
    float64 array of same shape as *samples*.
    """
    if not np.isfinite(rate_hz) or rate_hz <= 0.0 or rate_hz > 20.0:
        raise ValueError(f"rate_hz must be in (0, 20] Hz; got {rate_hz!r}.")
    if not np.isfinite(depth_ms) or depth_ms < 0.0 or depth_ms > 20.0:
        raise ValueError(f"depth_ms must be in [0, 20] ms; got {depth_ms!r}.")
    if not np.isfinite(base_delay_ms) or base_delay_ms <= 0.0 or base_delay_ms > 50.0:
        raise ValueError(f"base_delay_ms must be in (0, 50] ms; got {base_delay_ms!r}.")
    if base_delay_ms <= depth_ms:
        raise ValueError(
            f"base_delay_ms ({base_delay_ms}) must be > depth_ms ({depth_ms}) "
            "to prevent negative delay."
        )
    if not np.isfinite(mix) or mix < 0.0 or mix > 1.0:
        raise ValueError(f"mix must be in [0, 1]; got {mix!r}.")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}.")

    x = samples.astype(np.float64, copy=False)
    _require_finite_input(x)

    mono = x.ndim == 1
    channels = [x] if mono else [x[:, c] for c in range(x.shape[1])]

    processed = [
        _apply_chorus_channel(ch, sample_rate_hz, rate_hz, depth_ms, base_delay_ms, mix)
        for ch in channels
    ]

    y = processed[0] if mono else np.stack(processed, axis=1)
    _require_finite_output(y, "chorus")
    return y


# ---------------------------------------------------------------------------
# 3. Soft Distortion
# ---------------------------------------------------------------------------


def apply_soft_distortion(
    samples: np.ndarray,
    drive: float,
) -> np.ndarray:
    """Apply tanh-based soft clipping (waveshaping distortion).

    y[n] = tanh(β · x[n]) / tanh(β)

    where β = drive controls the amount of nonlinear compression.

    The normalisation by tanh(β) ensures unity gain for a full-scale input
    of 1.0, so the output level is comparable to the input level.

    Drive effect:
        drive → 0   : linear (identity limit)
        drive = 1   : mild saturation
        drive = 10  : heavy saturation / limiting
        drive → ∞   : hard clipping (asymptotically)

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel [N, C].
    drive:
        Saturation drive β.  Range: (0, 100].

    Returns
    -------
    float64 array of same shape as *samples*.  Values are in the range
    (-1/tanh(drive), 1/tanh(drive)) · max_input, bounded well within
    (-1, 1) when input is full-scale.
    """
    if not np.isfinite(drive) or drive <= 0.0 or drive > 100.0:
        raise ValueError(f"drive must be in (0, 100]; got {drive!r}.")

    x = samples.astype(np.float64, copy=False)
    _require_finite_input(x)

    denom = float(np.tanh(drive))
    if denom < 1e-15:
        # drive is so small that tanh(drive) ≈ 0; treat as identity
        return x.copy()

    y = np.tanh(drive * x) / denom
    _require_finite_output(y, "soft_distortion")
    return y


# ---------------------------------------------------------------------------
# 4. Schroeder Reverb
# ---------------------------------------------------------------------------

def apply_reverb(
    samples: np.ndarray,
    sample_rate_hz: int,
    room_size: float = 0.5,
    decay: float = 0.5,
    wet: float = 0.3,
) -> np.ndarray:
    """Apply classical Schroeder-style reverb.

    Input -> 4 parallel Feedback Comb Filters -> sum -> 2 serial All-Pass Filters -> Output.

    Parameters
    ----------
    samples:
        Float NumPy array — mono 1-D [N] or multi-channel [N, C].
    sample_rate_hz:
        Sample rate in Hz.
    room_size:
        0.1 to 1.0 (scales base delay lengths)
    decay:
        0.0 to 0.99 (feedback coefficient)
    wet:
        0.0 to 1.0 (mix)
    """
    if not np.isfinite(room_size) or room_size < 0.1 or room_size > 1.0:
        raise ValueError(f"room_size must be in [0.1, 1.0]; got {room_size!r}.")
    if not np.isfinite(decay) or decay < 0.0 or decay > 0.99:
        raise ValueError(f"decay must be in [0.0, 0.99]; got {decay!r}.")
    if not np.isfinite(wet) or wet < 0.0 or wet > 1.0:
        raise ValueError(f"wet must be in [0.0, 1.0]; got {wet!r}.")

    x = samples.astype(np.float64, copy=False)
    _require_finite_input(x)

    if wet < 1e-9:
        return x.copy()

    from scipy.signal import lfilter

    # Base delays in ms (mutually prime-ish)
    fbcf_delays_ms = [29.7, 37.1, 41.1, 43.7]
    apf_delays_ms = [5.0, 1.7]

    # Feedback gain (scaled by decay)
    g_fbcf = decay * 0.85 # ensure stability
    g_apf = 0.7

    def _apply_fbcf(signal, delay_ms):
        D = int((delay_ms * room_size / 1000.0) * sample_rate_hz)
        D = max(1, D)
        b = np.zeros(D + 1); b[0] = 1.0
        a = np.zeros(D + 1); a[0] = 1.0; a[D] = -g_fbcf
        return lfilter(b, a, signal)

    def _apply_apf(signal, delay_ms):
        D = int((delay_ms * room_size / 1000.0) * sample_rate_hz)
        D = max(1, D)
        b = np.zeros(D + 1); b[0] = -g_apf; b[D] = 1.0
        a = np.zeros(D + 1); a[0] = 1.0; a[D] = -g_apf
        return lfilter(b, a, signal)

    mono = x.ndim == 1
    channels = [x] if mono else [x[:, c] for c in range(x.shape[1])]
    
    out_channels = []
    for ch in channels:
        # Parallel FBCF
        fbcf_out = np.zeros_like(ch)
        for d in fbcf_delays_ms:
            fbcf_out += _apply_fbcf(ch, d)
        
        # Scale to prevent blowup from 4 parallel filters
        fbcf_out *= 0.25

        # Serial APF
        apf_out = fbcf_out
        for d in apf_delays_ms:
            apf_out = _apply_apf(apf_out, d)
        
        # Mix
        y_ch = (1.0 - wet) * ch + wet * apf_out
        out_channels.append(y_ch)
    
    y = out_channels[0] if mono else np.stack(out_channels, axis=1)
    _require_finite_output(y, "reverb")
    return y
