"""
dsp_core/filtering.py — Module 06: Audio Filtering
====================================================

Implements all classical IIR and FIR frequency-selective filter families plus
a practical parametric EQ biquad, all using numerically stable representations.

IMPLEMENTED FILTER FAMILIES (from AGENT_INSTRUCTIONS Module 06 roadmap):
=========================================================================

1.  Basic Frequency-Selective Filtering
        All four modes: lowpass, highpass, bandpass, bandstop
        (Every family below implements these four types.)

2.  Butterworth (family='butterworth')
        Maximally flat magnitude response in the passband. No passband ripple.
        The monotonically decreasing response is smooth in both passband and
        stopband. Good general-purpose starting point.
        |H(jΩ)|² = 1 / (1 + (Ω/Ωc)^{2N})

3.  Chebyshev Type I (family='chebyshev1')
        Equiripple (constant-amplitude ripple) in the passband, monotonically
        decreasing in the stopband. Achieves a sharper transition than
        Butterworth for the same order, at the cost of passband distortion.
        Controlled by rp (passband ripple in dB, e.g. 0.5–3 dB).

4.  Chebyshev Type II (family='chebyshev2')
        Flat (maximally smooth) passband, equiripple stopband. Sharper
        transition than Butterworth for comparable constraints. The stopband
        attenuation is set by rs (e.g. 40–80 dB).

5.  Elliptic (family='elliptic')
        Equiripple in both passband and stopband. Achieves the sharpest
        transition band for a given order among all classical IIR designs.
        Requires both rp (passband ripple, dB) and rs (stopband atten., dB).

6.  Bessel (family='bessel')
        Maximally flat group delay (approximately linear phase) in the
        passband. This makes the step/impulse response clean with minimal
        overshoot. The tradeoff is poorer magnitude selectivity compared
        with Butterworth or sharper designs. Ideal when phase fidelity
        matters more than sharp roll-off.

7.  Numerical Stability via SOS/Biquad (applied to all IIR families)
        All IIR designs are computed directly in SOS form using
        scipy's `output='sos'` parameter, bypassing the numerically
        dangerous direct-form polynomial representation. See the SOS
        section below for details.

8.  FIR Window-Designed Filters (family='fir_window')
        FIR (Finite Impulse Response) filters designed using the windowing
        method. Always stable (no feedback), exact linear-phase response,
        predictable phase characteristics. The window determines the
        passband-stopband tradeoff. Supported windows: hamming, hann,
        blackman, bartlett, kaiser.
        Applied via zero-phase filtfilt(b, [1], signal).

9.  Parks-McClellan / Equiripple FIR (family='fir_remez')
        Advanced FIR design using the Remez exchange algorithm
        (scipy.signal.remez). Minimises the maximum deviation (equiripple)
        from the desired frequency response using an iterative Chebyshev
        approximation. Produces near-optimal FIR filters (minimax sense)
        for a given number of taps and transition bandwidth.
        Requires specifying a transition_bandwidth_hz.

10. Practical Audio Filters — Parametric EQ (filter_type='peaking')
        A biquad peaking equalizer implemented via the Audio EQ Cookbook
        (R. Bristow-Johnson) formulas. Boosts or cuts a frequency band by
        gain_db at center frequency center_hz with bandwidth Q (quality factor).
        Unlike frequency-selective filters, a peaking EQ modifies amplitude
        in a narrow band without blocking anything. Useful for tone shaping.

Mathematical background
-----------------------

IIR filters — difference equation:
    y[n] = b[0]*x[n] + b[1]*x[n-1] + ... − a[1]*y[n-1] − a[2]*y[n-2] − ...

The frequency response:
    H(e^{jω}) = B(e^{jω}) / A(e^{jω})

FIR filters:
    y[n] = Σ_{k=0}^{N} b[k] · x[n-k]
    H(e^{jω}) = Σ_{k=0}^{N} b[k] · e^{-jωk}
    A(z) = 1  (no feedback, always stable)

SOS — why not direct-form?
--------------------------
A naive high-order IIR stores all coefficients in a single polynomial.
For orders N ≥ 5–6, coefficient values become very large or very small,
causing catastrophic floating-point cancellation.

SOS factorises H(z) = H₁(z)·H₂(z)·...·H_K(z), where each H_k is:
    H_k(z) = (b₀ + b₁z⁻¹ + b₂z⁻²) / (1 + a₁z⁻¹ + a₂z⁻²)

Each section has small, well-conditioned coefficients — numerically stable
even at high filter orders.

Zero-phase filtering (sosfiltfilt / filtfilt)
---------------------------------------------
Standard causal filtering introduces group delay (time delay that varies by
frequency), distorting the signal in time.

For offline audio (this module), we apply zero-phase forward-backward filtering:
  1. Filter forward → produces causal output with N×delay
  2. Reverse signal → time-reverse
  3. Filter again → compensates the delay
  4. Reverse result → original time orientation, zero net phase shift

Net result: H_eff(e^{jω}) = |H(e^{jω})|² — squared magnitude, zero phase.
The effective order is doubled (2N), giving a sharper transition.

This is NOT real-time/causal. For real-time, use sosfilt/lfilter instead.

Peaking EQ biquad (Audio EQ Cookbook)
--------------------------------------
For a peaking EQ at center frequency f₀ with gain G dB and quality factor Q:
    A = 10^(G/40)        (amplitude factor)
    ω₀ = 2π·f₀/fs       (angular frequency)
    α = sin(ω₀)/(2Q)     (bandwidth parameter)

    b₀ = 1 + α·A
    b₁ = −2·cos(ω₀)
    b₂ = 1 − α·A
    a₀ = 1 + α/A
    a₁ = −2·cos(ω₀)
    a₂ = 1 − α/A

Filter order tradeoffs
----------------------
    Higher order → steeper transition → stronger stopband attenuation
                → more computation → potentially greater edge artifacts (filtfilt)
                → higher risk of numerical issues in direct-form (mitigated by SOS)

There is no universally best filter design. The correct choice depends on:
  - Passband flatness requirements (Butterworth/Bessel/Cheb-II)
  - Transition sharpness (Elliptic > Cheb-I/II > Butterworth > Bessel)
  - Phase/time-domain behavior (Bessel, then FIR/linear-phase)
  - Stopband attenuation (Elliptic > Cheb-II/I > Butterworth)
  - Stability (all are stable in SOS; FIR always stable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import (
    bessel,
    butter,
    cheby1,
    cheby2,
    ellip,
    filtfilt,
    firwin,
    freqz,
    remez,
    sosfiltfilt,
    sosfreqz,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FilterType = Literal["lowpass", "highpass", "bandpass", "bandstop", "peaking"]
FilterFamily = Literal[
    "butterworth",
    "chebyshev1",
    "chebyshev2",
    "elliptic",
    "bessel",
    "fir_window",
    "fir_remez",
]

_IIR_FAMILIES: frozenset[str] = frozenset(
    {"butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel"}
)
_FIR_FAMILIES: frozenset[str] = frozenset({"fir_window", "fir_remez"})
_VALID_FIR_WINDOWS: frozenset[str] = frozenset(
    {"hamming", "hann", "blackman", "bartlett", "kaiser"}
)

_MIN_SIGNAL_FACTOR = 9  # sosfiltfilt minimum length multiplier


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterSpec:
    """Complete specification for a digital filter.

    Attributes
    ----------
    filter_type:
        Frequency-selective mode: lowpass, highpass, bandpass, bandstop.
        Special: 'peaking' for a parametric EQ biquad.
    family:
        Filter design family (see module docstring).
    order:
        Filter order (IIR) or numtaps - 1 (FIR).
    sample_rate_hz:
        Sample rate of the target audio.
    cutoff_hz:
        Single cutoff frequency for lowpass/highpass.
    low_hz / high_hz:
        Band edges for bandpass/bandstop.
    ripple_db:
        Passband ripple in dB for Chebyshev I and Elliptic. Default 1.0 dB.
    attenuation_db:
        Minimum stopband attenuation in dB for Chebyshev II and Elliptic.
        Default 40.0 dB.
    fir_window:
        Window function for 'fir_window' family. Default 'hamming'.
    transition_bandwidth_hz:
        Transition bandwidth in Hz for 'fir_remez' (Parks-McClellan).
    center_hz:
        Centre frequency in Hz for the 'peaking' EQ filter.
    gain_db:
        Boost (+) or cut (-) in dB for the 'peaking' EQ.
    q_factor:
        Quality factor (centre freq / bandwidth) for the 'peaking' EQ.
    """

    filter_type: FilterType
    family: FilterFamily
    order: int
    sample_rate_hz: int
    cutoff_hz: float | None = None
    low_hz: float | None = None
    high_hz: float | None = None
    ripple_db: float = 1.0
    attenuation_db: float = 40.0
    fir_window: str = "hamming"
    transition_bandwidth_hz: float = 200.0
    center_hz: float | None = None
    gain_db: float = 0.0
    q_factor: float = 1.0


@dataclass(frozen=True)
class FilterDesignResult:
    """Structured result of a filter design operation.

    For IIR filters (Butterworth, Chebyshev, Elliptic, Bessel):
        sos is set; b_fir is None.

    For FIR filters (fir_window, fir_remez):
        b_fir is set (tap coefficients); sos is None.

    For the peaking EQ biquad:
        Implemented as a single SOS section; sos is set.
    """

    sos: np.ndarray | None
    b_fir: np.ndarray | None
    is_fir: bool
    n_sections: int
    spec: FilterSpec


@dataclass(frozen=True)
class FilterFrequencyResponse:
    """Frequency response evaluated from the filter design.

    Attributes
    ----------
    frequencies_hz:
        Frequency axis in Hz.
    magnitude_db:
        Magnitude response in dB (0 dB = unity gain in passband).
    cutoff_hz / low_hz / high_hz / center_hz:
        Cutoff / band edge / centre frequencies from the spec.
    filter_type:
        The filter type from the spec.
    family:
        The filter family from the spec.
    order:
        Filter order from the spec.
    """

    frequencies_hz: list[float]
    magnitude_db: list[float]
    cutoff_hz: float | None
    low_hz: float | None
    high_hz: float | None
    center_hz: float | None
    filter_type: FilterType
    family: FilterFamily
    order: int


@dataclass(frozen=True)
class FilteredAudio:
    sample_rate_hz: int
    samples: np.ndarray
    n_channels: int
    filter_spec: FilterSpec


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FilterDesignError(ValueError):
    """Raised when filter parameters are invalid or design fails."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_spec(spec: FilterSpec) -> None:
    """Raise FilterDesignError for any invalid parameter combination."""
    nyquist = spec.sample_rate_hz / 2.0

    if spec.sample_rate_hz <= 0:
        raise FilterDesignError(
            f"Sample rate must be positive, got {spec.sample_rate_hz} Hz."
        )

    if spec.order < 1:
        raise FilterDesignError(f"Filter order must be ≥ 1, got {spec.order}.")

    # IIR and FIR have different practical order limits.
    # FIR order = numtaps - 1; practical FIR filters typically use 32–512 taps.
    # IIR filters are efficient at low order; very high IIR orders risk instability.
    if spec.family in _IIR_FAMILIES and spec.filter_type != "peaking":
        if spec.order > 20:
            raise FilterDesignError(
                f"IIR filter order must be ≤ 20, got {spec.order}. "
                "Very high orders risk numerical instability even with SOS. "
                "Use a FIR filter for very high orders."
            )
    elif spec.family in _FIR_FAMILIES:
        if spec.order > 999:
            raise FilterDesignError(
                f"FIR filter order (numtaps - 1) must be ≤ 999, got {spec.order}."
            )

    # Peaking EQ validation
    if spec.filter_type == "peaking":
        if spec.center_hz is None:
            raise FilterDesignError("center_hz is required for a peaking EQ filter.")
        if spec.center_hz <= 0 or spec.center_hz >= nyquist:
            raise FilterDesignError(
                f"center_hz ({spec.center_hz:.1f} Hz) must be in (0, {nyquist:.1f}) Hz."
            )
        if spec.q_factor <= 0:
            raise FilterDesignError(f"q_factor must be positive, got {spec.q_factor}.")
        return  # No further validation needed for peaking EQ

    # Frequency-selective filter validation
    if spec.filter_type in ("lowpass", "highpass"):
        if spec.cutoff_hz is None:
            raise FilterDesignError(
                f"cutoff_hz is required for a {spec.filter_type} filter."
            )
        if spec.cutoff_hz <= 0:
            raise FilterDesignError(
                f"Cutoff frequency must be positive, got {spec.cutoff_hz} Hz."
            )
        if spec.cutoff_hz >= nyquist:
            raise FilterDesignError(
                f"Cutoff frequency ({spec.cutoff_hz:.1f} Hz) must be strictly below "
                f"Nyquist ({nyquist:.1f} Hz) for a {spec.sample_rate_hz} Hz signal."
            )

    elif spec.filter_type in ("bandpass", "bandstop"):
        if spec.low_hz is None or spec.high_hz is None:
            raise FilterDesignError(
                f"Both low_hz and high_hz are required for {spec.filter_type}."
            )
        if spec.low_hz <= 0:
            raise FilterDesignError(f"low_hz must be positive, got {spec.low_hz} Hz.")
        if spec.high_hz <= 0:
            raise FilterDesignError(f"high_hz must be positive, got {spec.high_hz} Hz.")
        if spec.low_hz >= spec.high_hz:
            raise FilterDesignError(
                f"low_hz ({spec.low_hz:.1f} Hz) must be strictly less than "
                f"high_hz ({spec.high_hz:.1f} Hz)."
            )
        if spec.high_hz >= nyquist:
            raise FilterDesignError(
                f"high_hz ({spec.high_hz:.1f} Hz) must be strictly below "
                f"Nyquist ({nyquist:.1f} Hz)."
            )
    else:
        raise FilterDesignError(
            f"Unknown filter_type: '{spec.filter_type}'. "
            "Expected: lowpass, highpass, bandpass, bandstop, peaking."
        )

    # Family-specific parameter validation
    if spec.family == "chebyshev1":
        if spec.ripple_db <= 0:
            raise FilterDesignError(
                f"ripple_db must be positive for Chebyshev I, got {spec.ripple_db}."
            )
    elif spec.family == "chebyshev2":
        if spec.attenuation_db <= 0:
            raise FilterDesignError(
                f"attenuation_db must be positive for Chebyshev II, "
                f"got {spec.attenuation_db}."
            )
    elif spec.family == "elliptic":
        if spec.ripple_db <= 0:
            raise FilterDesignError(
                f"ripple_db must be positive for Elliptic, got {spec.ripple_db}."
            )
        if spec.attenuation_db <= 0:
            raise FilterDesignError(
                f"attenuation_db must be positive for Elliptic, "
                f"got {spec.attenuation_db}."
            )
    elif spec.family == "fir_window":
        if spec.fir_window not in _VALID_FIR_WINDOWS:
            raise FilterDesignError(
                f"Unknown FIR window '{spec.fir_window}'. "
                f"Supported: {sorted(_VALID_FIR_WINDOWS)}."
            )
    elif spec.family == "fir_remez":
        if spec.transition_bandwidth_hz <= 0:
            raise FilterDesignError(
                f"transition_bandwidth_hz must be positive, "
                f"got {spec.transition_bandwidth_hz}."
            )
        # Check transition bands fit within valid frequency range
        _validate_remez_bands(spec, nyquist)
    elif spec.family not in _IIR_FAMILIES:
        raise FilterDesignError(
            f"Unknown filter family: '{spec.family}'. "
            "Expected: butterworth, chebyshev1, chebyshev2, elliptic, bessel, "
            "fir_window, fir_remez."
        )


def _validate_remez_bands(spec: FilterSpec, nyquist: float) -> None:
    """Validate that Parks-McClellan transition bands are non-overlapping and in-range."""
    tw = spec.transition_bandwidth_hz

    if spec.filter_type in ("lowpass", "highpass"):
        fc = spec.cutoff_hz
        assert fc is not None
        low_edge = fc - tw / 2
        high_edge = fc + tw / 2
        if low_edge <= 0:
            raise FilterDesignError(
                f"Transition band lower edge ({low_edge:.1f} Hz) is ≤ 0. "
                f"Reduce transition_bandwidth_hz ({tw:.1f} Hz) or increase cutoff_hz."
            )
        if high_edge >= nyquist:
            raise FilterDesignError(
                f"Transition band upper edge ({high_edge:.1f} Hz) is ≥ Nyquist "
                f"({nyquist:.1f} Hz). Reduce transition_bandwidth_hz or lower cutoff_hz."
            )
    else:
        low_hz = spec.low_hz
        high_hz = spec.high_hz
        assert low_hz is not None and high_hz is not None
        if low_hz - tw / 2 <= 0:
            raise FilterDesignError(
                f"Low transition band lower edge ({low_hz - tw/2:.1f} Hz) is ≤ 0. "
                "Reduce transition_bandwidth_hz or increase low_hz."
            )
        if high_hz + tw / 2 >= nyquist:
            raise FilterDesignError(
                f"High transition band upper edge ({high_hz + tw/2:.1f} Hz) is ≥ Nyquist. "
                "Reduce transition_bandwidth_hz or decrease high_hz."
            )
        if low_hz + tw / 2 >= high_hz - tw / 2:
            raise FilterDesignError(
                "Transition bands overlap. Increase the gap between low_hz and high_hz, "
                "or reduce transition_bandwidth_hz."
            )


# ---------------------------------------------------------------------------
# Filter design
# ---------------------------------------------------------------------------


def _build_wn(spec: FilterSpec, nyquist: float) -> float | list[float]:
    """Compute normalised cutoff Wn for IIR scipy calls."""
    if spec.filter_type in ("lowpass", "highpass"):
        return spec.cutoff_hz / nyquist  # type: ignore[operator]
    return [spec.low_hz / nyquist, spec.high_hz / nyquist]  # type: ignore[operator,list-item]


def design_filter(spec: FilterSpec) -> FilterDesignResult:
    """Design a digital filter from a FilterSpec.

    Returns a FilterDesignResult containing either an SOS matrix (IIR/peaking)
    or FIR tap coefficients (b_fir). See the module docstring for the
    mathematical details of each filter family.

    Parameters
    ----------
    spec:
        A FilterSpec. Validation is performed internally.

    Returns
    -------
    FilterDesignResult

    Raises
    ------
    FilterDesignError
        If parameters are invalid or the design algorithm fails.
    """
    _validate_spec(spec)

    # -----------------------------------------------------------------------
    # Peaking EQ biquad (Audio EQ Cookbook — R. Bristow-Johnson)
    # -----------------------------------------------------------------------
    if spec.filter_type == "peaking":
        return _design_peaking_eq(spec)

    nyquist = spec.sample_rate_hz / 2.0
    btype = spec.filter_type  # scipy uses same btype names

    # -----------------------------------------------------------------------
    # IIR families
    # -----------------------------------------------------------------------
    if spec.family in _IIR_FAMILIES:
        wn = _build_wn(spec, nyquist)
        
        # We explicitly pass arguments instead of unpacking a dict (**common) 
        # to satisfy static type checkers (e.g. Pylance/MyPy).
        # We also wrap the return in np.asarray to guarantee the type.
        if spec.family == "butterworth":
            sos = np.asarray(butter(N=spec.order, Wn=wn, btype=btype, analog=False, output="sos"))

        elif spec.family == "chebyshev1":
            sos = np.asarray(cheby1(N=spec.order, rp=spec.ripple_db, Wn=wn, btype=btype, analog=False, output="sos"))

        elif spec.family == "chebyshev2":
            sos = np.asarray(cheby2(N=spec.order, rs=spec.attenuation_db, Wn=wn, btype=btype, analog=False, output="sos"))

        elif spec.family == "elliptic":
            sos = np.asarray(ellip(N=spec.order, rp=spec.ripple_db, rs=spec.attenuation_db, Wn=wn, btype=btype, analog=False, output="sos"))

        elif spec.family == "bessel":
            sos = np.asarray(bessel(N=spec.order, Wn=wn, btype=btype, analog=False, output="sos", norm="phase"))

        else:
            raise FilterDesignError(f"Unhandled IIR family: {spec.family}")

        return FilterDesignResult(
            sos=sos,
            b_fir=None,
            is_fir=False,
            n_sections=sos.shape[0],
            spec=spec,
        )

    # -----------------------------------------------------------------------
    # FIR — Window method (firwin)
    # -----------------------------------------------------------------------
    if spec.family == "fir_window":
        return _design_fir_window(spec)

    # -----------------------------------------------------------------------
    # FIR — Parks-McClellan / Equiripple (remez)
    # -----------------------------------------------------------------------
    if spec.family == "fir_remez":
        return _design_fir_remez(spec)

    raise FilterDesignError(f"Unknown filter family: '{spec.family}'.")


# ---------------------------------------------------------------------------
# Peaking EQ biquad
# ---------------------------------------------------------------------------


def _design_peaking_eq(spec: FilterSpec) -> FilterDesignResult:
    """Design a biquad peaking EQ using the Audio EQ Cookbook formulas.

    A peaking EQ boosts or cuts a frequency band by gain_db dB at center_hz
    with a bandwidth determined by q_factor. Unlike frequency-selective filters,
    it never fully blocks any frequency.

    Reference: R. Bristow-Johnson, "Audio EQ Cookbook"
    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
    """
    f0 = spec.center_hz
    assert f0 is not None
    fs = spec.sample_rate_hz

    A = 10.0 ** (spec.gain_db / 40.0)  # amplitude factor (square root of power)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * spec.q_factor)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    # Normalise by a0 to form standard SOS row: [b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]
    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

    return FilterDesignResult(
        sos=sos,
        b_fir=None,
        is_fir=False,
        n_sections=1,
        spec=spec,
    )


# ---------------------------------------------------------------------------
# FIR Window design
# ---------------------------------------------------------------------------


def _design_fir_window(spec: FilterSpec) -> FilterDesignResult:
    """Design a linear-phase FIR filter using the windowing method.

    FIR (Finite Impulse Response) filters are always stable (no poles), have
    linear phase (when symmetric) and predictable behaviour. The windowing
    method applies a window function to the ideal sinc impulse response, trading
    off transition bandwidth against stopband attenuation:

        Window       | Stopband atten. | Transition width
        -------------|-----------------|------------------
        Rectangular  | ~13 dB          | Narrowest (Gibbs ringing)
        Hamming      | ~41 dB          | Moderate
        Hann         | ~44 dB          | Moderate
        Blackman     | ~74 dB          | Wider
        Kaiser       | Variable        | Configurable via β

    numtaps = order + 1. For odd-symmetric filters (highpass, bandstop),
    numtaps is forced to be odd if order+1 is even.
    """
    numtaps = spec.order + 1

    btype = spec.filter_type
    # firwin requires odd numtaps for highpass and bandstop
    if btype in ("highpass", "bandstop") and numtaps % 2 == 0:
        numtaps += 1

    fs = spec.sample_rate_hz

    window_param = spec.fir_window
    if window_param == "kaiser":
        from scipy.signal import kaiser_beta
        window_param = ("kaiser", kaiser_beta(spec.attenuation_db))

    if btype == "lowpass":
        b = firwin(numtaps, spec.cutoff_hz, window=window_param,
                   pass_zero=True, fs=fs)
    elif btype == "highpass":
        b = firwin(numtaps, spec.cutoff_hz, window=window_param,
                   pass_zero=False, fs=fs)
    elif btype == "bandpass":
        b = firwin(numtaps, [spec.low_hz, spec.high_hz], window=window_param,
                   pass_zero=False, fs=fs)
    elif btype == "bandstop":
        b = firwin(numtaps, [spec.low_hz, spec.high_hz], window=window_param,
                   pass_zero=True, fs=fs)
    else:
        raise FilterDesignError(
            f"FIR window method does not support filter_type='{btype}'."
        )

    return FilterDesignResult(
        sos=None,
        b_fir=b,
        is_fir=True,
        n_sections=len(b),  # n_sections = numtaps for FIR
        spec=spec,
    )


# ---------------------------------------------------------------------------
# FIR Parks-McClellan (remez)
# ---------------------------------------------------------------------------


def _design_fir_remez(spec: FilterSpec) -> FilterDesignResult:
    """Design an equiripple FIR filter using the Parks-McClellan algorithm.

    The Remez exchange algorithm iteratively finds the FIR filter of a given
    order that minimises the maximum deviation from the desired response over
    the specified bands. This is optimal in the minimax (Chebyshev) sense:
    no other FIR filter with the same numtaps will have a smaller maximum
    error given the same band specification.

    The transition_bandwidth_hz parameter specifies the width of the transition
    band between passband and stopband. Larger values allow tighter specs with
    fewer taps; smaller values require more taps for the same performance.
    """
    numtaps = spec.order + 1
    tw = spec.transition_bandwidth_hz
    fs = spec.sample_rate_hz
    nyquist = fs / 2.0

    btype = spec.filter_type

    if btype == "lowpass":
        fc = spec.cutoff_hz
        assert fc is not None
        bands = [0.0, fc - tw / 2, fc + tw / 2, nyquist]
        desired = [1.0, 0.0]
    elif btype == "highpass":
        fc = spec.cutoff_hz
        assert fc is not None
        bands = [0.0, fc - tw / 2, fc + tw / 2, nyquist]
        desired = [0.0, 1.0]
    elif btype == "bandpass":
        lo, hi = spec.low_hz, spec.high_hz
        assert lo is not None and hi is not None
        bands = [0.0, lo - tw / 2, lo + tw / 2, hi - tw / 2, hi + tw / 2, nyquist]
        desired = [0.0, 1.0, 0.0]
    elif btype == "bandstop":
        lo, hi = spec.low_hz, spec.high_hz
        assert lo is not None and hi is not None
        bands = [0.0, lo - tw / 2, lo + tw / 2, hi - tw / 2, hi + tw / 2, nyquist]
        desired = [1.0, 0.0, 1.0]
    else:
        raise FilterDesignError(
            f"Parks-McClellan does not support filter_type='{btype}'."
        )

    try:
        b = remez(numtaps, bands, desired, fs=fs)
    except Exception as exc:
        raise FilterDesignError(
            f"Parks-McClellan design failed: {exc}. "
            "Try increasing numtaps (order), reducing transition_bandwidth_hz, "
            "or adjusting the cutoff frequency."
        ) from exc

    return FilterDesignResult(
        sos=None,
        b_fir=b,
        is_fir=True,
        n_sections=len(b),
        spec=spec,
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def apply_filter(design: FilterDesignResult, samples: np.ndarray) -> np.ndarray:
    """Apply a designed filter to audio samples using zero-phase filtering.

    IIR (Butterworth, Chebyshev, Elliptic, Bessel, Peaking EQ):
        Uses sosfiltfilt — forward-backward filtering with SOS cascade.

    FIR (Window, Parks-McClellan):
        Uses filtfilt(b, [1.0], signal) — forward-backward filtering with
        FIR tap coefficients. For FIR this is equivalent to applying the
        filter twice in opposite directions, preserving linear phase.

    Zero-phase filtering is appropriate for offline audio processing.
    It must NOT be used in real-time/causal applications.

    Multi-channel audio is processed one channel at a time with the same
    filter. The channel count, sample count, and sample rate are preserved.

    Parameters
    ----------
    design:
        A FilterDesignResult from design_filter().
    samples:
        Audio samples as float32 or integer ndarray. Shape: (N,) mono or
        (N, C) for C channels. No peak-normalisation is applied.

    Returns
    -------
    np.ndarray
        Filtered samples, same shape, dtype float32.

    Raises
    ------
    FilterDesignError
        If the signal is too short for stable filtfilt padding.
    """
    # Convert to float32 without peak-normalisation.
    if not np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float32)
    else:
        samples = samples.astype(np.float32)

    is_mono = samples.ndim == 1
    signal_length = samples.shape[0]

    if design.is_fir:
        # FIR: minimum padlen for filtfilt is 3*(numtaps-1)/2
        b = design.b_fir
        assert b is not None
        numtaps = len(b)
        min_len = 3 * (numtaps - 1) + 1
        if signal_length < min_len:
            raise FilterDesignError(
                f"Signal too short for stable FIR zero-phase filtering. "
                f"Signal has {signal_length} samples but at least {min_len} are "
                f"required for a {numtaps}-tap filter. "
                "Use a lower filter order or a longer audio clip."
            )
        a = np.array([1.0], dtype=np.float64)
        b_f64 = b.astype(np.float64)

        if is_mono:
            return filtfilt(b_f64, a, samples.astype(np.float64)).astype(np.float32)

        n_channels = samples.shape[1]
        output = np.empty_like(samples, dtype=np.float32)
        for ch in range(n_channels):
            output[:, ch] = filtfilt(b_f64, a, samples[:, ch].astype(np.float64)).astype(np.float32)
        return output

    else:
        # IIR (SOS): minimum signal length for sosfiltfilt
        sos = design.sos
        assert sos is not None
        n_sections = design.n_sections
        min_len = _MIN_SIGNAL_FACTOR * n_sections
        if signal_length < min_len:
            raise FilterDesignError(
                f"Signal too short for stable IIR zero-phase filtering. "
                f"Signal has {signal_length} samples but at least {min_len} are "
                f"required for a {n_sections}-section filter. "
                "Use a lower filter order or a longer audio clip."
            )

        if is_mono:
            return sosfiltfilt(sos, samples).astype(np.float32)

        n_channels = samples.shape[1]
        output = np.empty_like(samples, dtype=np.float32)
        for ch in range(n_channels):
            output[:, ch] = sosfiltfilt(sos, samples[:, ch])
        return output


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------


def compute_frequency_response(
    design: FilterDesignResult,
    n_points: int = 512,
) -> FilterFrequencyResponse:
    """Compute the theoretical frequency response of a designed filter.

    For IIR (SOS): uses scipy.signal.sosfreqz.
    For FIR (b): uses scipy.signal.freqz.

    The magnitude is returned in dB (20·log10(|H|)) with a floor at −120 dB.

    Parameters
    ----------
    design:
        A FilterDesignResult from design_filter().
    n_points:
        Number of frequency points (default 512).

    Returns
    -------
    FilterFrequencyResponse
    """
    fs = design.spec.sample_rate_hz

    if design.is_fir:
        b = design.b_fir
        assert b is not None
        _w, _h = freqz(b, worN=n_points, fs=fs)
    else:
        sos = design.sos
        assert sos is not None
        _w, _h = sosfreqz(sos, worN=n_points, fs=fs)

    # Type hint cast for Pylance
    w = np.asarray(_w)
    h = np.asarray(_h)

    magnitude = np.abs(h)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-6))

    return FilterFrequencyResponse(
        frequencies_hz=w.tolist(),
        magnitude_db=magnitude_db.tolist(),
        cutoff_hz=design.spec.cutoff_hz,
        low_hz=design.spec.low_hz,
        high_hz=design.spec.high_hz,
        center_hz=design.spec.center_hz,
        filter_type=design.spec.filter_type,
        family=design.spec.family,
        order=design.spec.order,
    )
