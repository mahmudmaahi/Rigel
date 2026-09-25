"""
tests/test_voice_lab_phases_a_to_d.py — Module 09: Voice Laboratory, Phases A–D
==================================================================================

Tests for:
    Phase A — Gain and Level (voice_gain.py)
    Phase B — Speed / Resampling (time_scale.apply_speed)
    Phase C — Phase Vocoder Time Stretch (time_scale.apply_time_stretch)
    Phase D — Pitch Shift (pitch_shift.apply_pitch_shift)

Conventions used in these tests
---------------------------------
- All test signals are synthetic (sine waves, silence, impulses) so
  frequency measurements are exact and reproducible.
- "Dominant frequency" is measured via np.argmax of the magnitude spectrum
  applied to the output signal.
- Duration is measured by len(output) / sample_rate.
- We do NOT expect exact sample counts for every resampling ratio because
  rational approximation may differ by ±1 sample.  Tolerances are documented.
- Phase-vocoder reconstruction tolerance is measured and reported per test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dsp_core.voice_gain import (
    VoiceDSPError,
    apply_db_gain,
    apply_linear_gain,
    apply_peak_normalization,
    apply_rms_normalization,
    measure_peak,
    measure_rms,
)
from dsp_core.time_scale import _apply_raw_resampling, apply_time_stretch, apply_speed
from dsp_core.pitch_shift import apply_pitch_shift, semitones_to_ratio


# ---------------------------------------------------------------------------
# Test-signal helpers
# ---------------------------------------------------------------------------


def _sine(
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono float64 sine wave."""
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def _stereo(
    freq_l: float = 440.0,
    freq_r: float = 880.0,
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a [N, 2] float64 stereo signal."""
    l_ch = _sine(freq_l, duration_s, sample_rate, amplitude)
    r_ch = _sine(freq_r, duration_s, sample_rate, amplitude)
    return np.stack([l_ch, r_ch], axis=1)


def _dominant_freq(signal: np.ndarray, sample_rate: int) -> float:
    """Return the dominant frequency (Hz) of a mono signal via FFT magnitude."""
    x = signal.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    return float(freqs[np.argmax(spectrum)])


def _duration_s(signal: np.ndarray, sample_rate: int) -> float:
    """Return the duration in seconds for a mono or multi-channel array."""
    return signal.shape[0] / sample_rate


# =========================================================================
# PHASE A — GAIN AND LEVEL
# =========================================================================


class TestLinearGain:
    def test_unity_gain_is_identity(self):
        x = _sine(440.0, 1.0)
        y = apply_linear_gain(x, 1.0)
        np.testing.assert_allclose(y, x, rtol=1e-12)

    def test_double_gain(self):
        x = _sine(440.0, 0.1)
        y = apply_linear_gain(x, 2.0)
        np.testing.assert_allclose(y, 2.0 * x, rtol=1e-12)

    def test_zero_gain_produces_silence(self):
        x = _sine(440.0, 0.1)
        y = apply_linear_gain(x, 0.0)
        assert np.all(y == 0.0)

    def test_attenuation(self):
        x = _sine(440.0, 0.1)
        y = apply_linear_gain(x, 0.5)
        np.testing.assert_allclose(y, 0.5 * x, rtol=1e-12)

    def test_output_is_float64(self):
        x = _sine().astype(np.float32)
        y = apply_linear_gain(x, 1.0)
        assert y.dtype == np.float64

    def test_negative_gain_raises(self):
        with pytest.raises(ValueError, match="gain"):
            apply_linear_gain(_sine(), -0.1)

    def test_inf_gain_raises(self):
        with pytest.raises(ValueError):
            apply_linear_gain(_sine(), math.inf)

    def test_nan_input_raises(self):
        x = _sine()
        x[100] = np.nan
        with pytest.raises(VoiceDSPError):
            apply_linear_gain(x, 1.0)

    def test_stereo_shape_preserved(self):
        x = _stereo()
        y = apply_linear_gain(x, 2.0)
        assert y.shape == x.shape
        np.testing.assert_allclose(y, 2.0 * x, rtol=1e-12)


class TestDbGain:
    def test_0_db_is_unity(self):
        x = _sine(440.0, 0.1)
        y = apply_db_gain(x, 0.0)
        np.testing.assert_allclose(y, x, rtol=1e-12)

    def test_plus_6_db_approximately_doubles(self):
        x = _sine(440.0, 0.1, amplitude=0.5)
        y = apply_db_gain(x, 6.0)
        # 10^(6/20) ≈ 1.99526, not exactly 2.0
        expected_gain = 10.0 ** (6.0 / 20.0)
        np.testing.assert_allclose(y, expected_gain * x, rtol=1e-12)

    def test_minus_6_db_approximately_halves(self):
        x = _sine(440.0, 0.1, amplitude=0.5)
        y = apply_db_gain(x, -6.0)
        expected_gain = 10.0 ** (-6.0 / 20.0)
        np.testing.assert_allclose(y, expected_gain * x, rtol=1e-12)

    def test_plus_20_db(self):
        x = _sine(440.0, 0.1, amplitude=0.1)
        y = apply_db_gain(x, 20.0)
        expected_gain = 10.0 ** (20.0 / 20.0)  # = 10.0 exactly
        np.testing.assert_allclose(y, expected_gain * x, rtol=1e-12)

    def test_nan_db_raises(self):
        with pytest.raises(ValueError):
            apply_db_gain(_sine(), math.nan)

    def test_output_may_exceed_unity(self):
        """High dB gain is allowed — clipping is not silently applied."""
        x = _sine(440.0, 0.1, amplitude=0.5)
        y = apply_db_gain(x, 40.0)
        assert measure_peak(y) > 1.0   # not silently clipped


class TestPeakNormalization:
    def test_peak_is_target(self):
        x = _sine(440.0, 1.0, amplitude=0.3)
        y = apply_peak_normalization(x, 1.0)
        assert abs(measure_peak(y) - 1.0) < 1e-10

    def test_custom_target_peak(self):
        x = _sine(440.0, 0.5, amplitude=0.3)
        y = apply_peak_normalization(x, 0.7)
        assert abs(measure_peak(y) - 0.7) < 1e-10

    def test_zero_signal_returned_unchanged(self):
        x = np.zeros(1000, dtype=np.float64)
        y = apply_peak_normalization(x)
        assert np.all(y == 0.0)

    def test_already_at_peak(self):
        x = _sine(440.0, 0.5, amplitude=1.0)
        y = apply_peak_normalization(x, 1.0)
        # The gain applied is target/peak which is not exactly 1.0 due to float64;
        # we check that the result is very close, not bitwise-identical.
        np.testing.assert_allclose(y, x, rtol=1e-6)

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            apply_peak_normalization(_sine(), 0.0)
        with pytest.raises(ValueError):
            apply_peak_normalization(_sine(), -1.0)

    def test_stereo(self):
        x = _stereo(amplitude=0.4)
        y = apply_peak_normalization(x, 1.0)
        assert abs(measure_peak(y) - 1.0) < 1e-10
        assert y.shape == x.shape


class TestRmsNormalization:
    def test_rms_is_target(self):
        x = _sine(440.0, 1.0, amplitude=0.3)
        target = 0.1
        y = apply_rms_normalization(x, target)
        assert abs(measure_rms(y) - target) < 1e-10

    def test_zero_signal_returned_unchanged(self):
        x = np.zeros(1000, dtype=np.float64)
        y = apply_rms_normalization(x)
        assert np.all(y == 0.0)

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            apply_rms_normalization(_sine(), 0.0)

    def test_stereo(self):
        x = _stereo(amplitude=0.6)
        y = apply_rms_normalization(x, 0.1)
        assert abs(measure_rms(y) - 0.1) < 1e-10
        assert y.shape == x.shape


# =========================================================================
# PHASE B — SPEED / RESAMPLING
# =========================================================================


class TestApplyRawResampling:
    SR = 44100

    def test_unity_speed_is_identity(self):
        x = _sine(440.0, 1.0, self.SR)
        y = _apply_raw_resampling(x, self.SR, 1.0)
        np.testing.assert_allclose(y, x, rtol=1e-12)

    def test_2x_speed_halves_duration(self):
        x = _sine(440.0, 1.0, self.SR)
        y = _apply_raw_resampling(x, self.SR, 2.0)
        expected_len = len(x) // 2
        # Allow ±1 sample tolerance from rational approximation
        assert abs(len(y) - expected_len) <= 2, (
            f"Expected ≈{expected_len} samples, got {len(y)}"
        )

    def test_2x_speed_doubles_frequency(self):
        x = _sine(440.0, 1.0, self.SR)
        y = _apply_raw_resampling(x, self.SR, 2.0)
        dom = _dominant_freq(y, self.SR)
        # Frequency should be ≈880 Hz after 2× speed
        assert abs(dom - 880.0) < 20.0, f"Expected ≈880 Hz, got {dom:.1f} Hz"

    def test_half_speed_doubles_duration(self):
        x = _sine(440.0, 1.0, self.SR)
        y = _apply_raw_resampling(x, self.SR, 0.5)
        expected_len = len(x) * 2
        assert abs(len(y) - expected_len) <= 2

    def test_half_speed_halves_frequency(self):
        x = _sine(440.0, 1.0, self.SR)
        y = _apply_raw_resampling(x, self.SR, 0.5)
        dom = _dominant_freq(y, self.SR)
        assert abs(dom - 220.0) < 20.0, f"Expected ≈220 Hz, got {dom:.1f} Hz"

    def test_1_25x_speed(self):
        x = _sine(440.0, 1.0, self.SR)
        y = _apply_raw_resampling(x, self.SR, 1.25)
        expected_len = int(len(x) / 1.25)
        assert abs(len(y) - expected_len) <= 5

    def test_silence_returns_finite_zeros(self):
        x = np.zeros(self.SR, dtype=np.float64)
        y = _apply_raw_resampling(x, self.SR, 2.0)
        assert np.all(np.isfinite(y))
        assert np.allclose(y, 0.0, atol=1e-14)

    def test_stereo_shape(self):
        x = _stereo(440.0, 880.0, 1.0, self.SR)
        y = _apply_raw_resampling(x, self.SR, 2.0)
        assert y.ndim == 2
        assert y.shape[1] == 2

    def test_stereo_duration_correct(self):
        x = _stereo(440.0, 880.0, 1.0, self.SR)
        y = _apply_raw_resampling(x, self.SR, 2.0)
        assert abs(y.shape[0] - len(x) // 2) <= 2

    def test_odd_length(self):
        x = _sine(440.0, 0.1, self.SR)[:4411]  # odd length
        y = _apply_raw_resampling(x, self.SR, 2.0)
        assert len(y) > 0
        assert np.all(np.isfinite(y))

    def test_even_length(self):
        x = _sine(440.0, 0.1, self.SR)[:4410]  # even length
        y = _apply_raw_resampling(x, self.SR, 2.0)
        assert len(y) > 0
        assert np.all(np.isfinite(y))

    def test_near_nyquist_no_aliasing(self):
        """A tone near Nyquist/2 should not alias to a wrong frequency."""
        # Tone at ¼ Nyquist = Fs/4 Hz
        fs = self.SR
        freq = fs // 4  # e.g. 11025 Hz at 44100
        x = _sine(freq, 1.0, fs, amplitude=0.5)
        # Apply 0.5x speed: new duration = 2s, new pitch = fs/8 = 5512.5 Hz
        y = _apply_raw_resampling(x, fs, 0.5)
        dom = _dominant_freq(y, fs)
        assert abs(dom - freq * 0.5) < 200.0

    def test_invalid_speed_raises(self):
        with pytest.raises(ValueError):
            _apply_raw_resampling(_sine(), self.SR, 0.0)
        with pytest.raises(ValueError):
            _apply_raw_resampling(_sine(), self.SR, -1.0)
        with pytest.raises(ValueError):
            _apply_raw_resampling(_sine(), self.SR, math.nan)
        with pytest.raises(ValueError):
            _apply_raw_resampling(_sine(), self.SR, math.inf)

    def test_output_finite(self):
        x = _sine(440.0, 0.5, self.SR)
        y = _apply_raw_resampling(x, self.SR, 1.5)
        assert np.all(np.isfinite(y))

    def test_multiple_sample_rates(self):
        for sr in [8000, 16000, 22050, 44100, 48000]:
            x = _sine(200.0, 0.5, sr)
            y = _apply_raw_resampling(x, sr, 2.0)
            assert np.all(np.isfinite(y))


# =========================================================================
# PHASE C — PHASE VOCODER TIME STRETCH
# =========================================================================


class TestApplyTimeStretch:
    SR = 44100

    # ---- Identity ----

    def test_identity_stretch_duration(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 1.0)
        # Within 1% of original length
        assert abs(len(y) - len(x)) <= len(x) * 0.01

    def test_identity_stretch_frequency(self):
        """S=1.0: dominant frequency should be preserved."""
        x = _sine(440.0, 2.0, self.SR)
        y = apply_time_stretch(x, self.SR, 1.0)
        dom = _dominant_freq(y, self.SR)
        assert abs(dom - 440.0) < 10.0, f"Expected ≈440 Hz, got {dom:.1f} Hz"

    def test_identity_reconstruction_error(self):
        """S=1.0: reconstructed signal should be close to original."""
        x = _sine(440.0, 1.0, self.SR, amplitude=0.5)
        y = apply_time_stretch(x, self.SR, 1.0)
        # Align lengths for comparison
        min_len = min(len(x), len(y))
        err = np.sqrt(np.mean((x[:min_len] - y[:min_len]) ** 2))
        # Reconstruction error should be small compared to signal amplitude (0.5)
        assert err < 0.05, f"Identity reconstruction RMS error: {err:.6f}"

    # ---- 2× stretch ----

    def test_2x_stretch_doubles_duration(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 2.0)
        expected = len(x) * 2
        # Allow ±2% for phase-vocoder hop rounding
        assert abs(len(y) - expected) <= expected * 0.02

    def test_2x_stretch_preserves_frequency(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 2.0)
        dom = _dominant_freq(y, self.SR)
        assert abs(dom - 440.0) < 10.0, f"Expected ≈440 Hz, got {dom:.1f} Hz"

    def test_2x_stretch_1khz(self):
        x = _sine(1000.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 2.0)
        dom = _dominant_freq(y, self.SR)
        assert abs(dom - 1000.0) < 15.0

    # ---- 0.5× stretch ----

    def test_half_stretch_halves_duration(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 0.5)
        expected = len(x) // 2
        assert abs(len(y) - expected) <= expected * 0.02

    def test_half_stretch_preserves_frequency(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 0.5)
        dom = _dominant_freq(y, self.SR)
        assert abs(dom - 440.0) < 10.0

    # ---- Silence ----

    def test_silence_returns_finite(self):
        x = np.zeros(self.SR, dtype=np.float64)
        y = apply_time_stretch(x, self.SR, 2.0)
        assert np.all(np.isfinite(y))

    def test_silence_is_approximately_zero(self):
        x = np.zeros(self.SR, dtype=np.float64)
        y = apply_time_stretch(x, self.SR, 2.0)
        assert np.max(np.abs(y)) < 1e-10

    # ---- Very short signals ----

    def test_very_short_signal(self):
        """Signals shorter than n_fft should not crash."""
        x = _sine(440.0, 0.01, self.SR)  # ~441 samples < 2048
        y = apply_time_stretch(x, self.SR, 2.0)
        assert len(y) > 0
        assert np.all(np.isfinite(y))

    # ---- Stereo ----

    def test_stereo_shape(self):
        x = _stereo(440.0, 880.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 2.0)
        assert y.ndim == 2
        assert y.shape[1] == 2

    def test_stereo_duration(self):
        x = _stereo(440.0, 880.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 2.0)
        expected = x.shape[0] * 2
        assert abs(y.shape[0] - expected) <= expected * 0.02

    def test_stereo_left_frequency(self):
        x = _stereo(440.0, 880.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 2.0)
        dom_l = _dominant_freq(y[:, 0], self.SR)
        assert abs(dom_l - 440.0) < 15.0

    def test_stereo_right_frequency(self):
        x = _stereo(440.0, 880.0, 1.0, self.SR)
        y = apply_time_stretch(x, self.SR, 2.0)
        dom_r = _dominant_freq(y[:, 1], self.SR)
        assert abs(dom_r - 880.0) < 20.0

    # ---- Output is finite ----

    def test_output_finite_general(self):
        x = _sine(440.0, 0.5, self.SR)
        for s in [0.5, 1.0, 1.5, 2.0]:
            y = apply_time_stretch(x, self.SR, s)
            assert np.all(np.isfinite(y)), f"stretch={s} produced NaN/Inf"

    # ---- Boundary/padding ----

    def test_output_non_empty(self):
        for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
            x = _sine(440.0, 0.5, self.SR)
            y = apply_time_stretch(x, self.SR, s)
            assert len(y) > 0

    # ---- Invalid parameters ----

    def test_invalid_stretch_raises(self):
        with pytest.raises(ValueError):
            apply_time_stretch(_sine(), self.SR, 0.0)
        with pytest.raises(ValueError):
            apply_time_stretch(_sine(), self.SR, -1.0)
        with pytest.raises(ValueError):
            apply_time_stretch(_sine(), self.SR, math.nan)
        with pytest.raises(ValueError):
            apply_time_stretch(_sine(), self.SR, math.inf)

    def test_nan_input_raises(self):
        x = _sine(440.0, 0.5, self.SR)
        x[100] = np.nan
        with pytest.raises(VoiceDSPError):
            apply_time_stretch(x, self.SR, 2.0)


# =========================================================================
# PHASE D — PITCH SHIFT
# =========================================================================


class TestSemitonesToRatio:
    def test_zero_semitones_is_unity(self):
        assert abs(semitones_to_ratio(0) - 1.0) < 1e-12

    def test_plus_12_is_2(self):
        assert abs(semitones_to_ratio(12) - 2.0) < 1e-10

    def test_minus_12_is_half(self):
        assert abs(semitones_to_ratio(-12) - 0.5) < 1e-10

    def test_plus_1_semitone(self):
        r = semitones_to_ratio(1)
        # 2^(1/12) ≈ 1.059463
        assert abs(r - 1.059463) < 1e-5

    def test_plus_24_is_4(self):
        assert abs(semitones_to_ratio(24) - 4.0) < 1e-10


class TestApplyPitchShift:
    SR = 44100
    # Tolerance for measured frequency (Hz).  The phase vocoder + resampler
    # pipeline introduces artifacts; we allow up to 3% frequency error.
    FREQ_TOL_PCT = 3.0

    def _freq_tol(self, expected_hz: float) -> float:
        return expected_hz * self.FREQ_TOL_PCT / 100.0

    # ---- Zero shift (identity) ----

    def test_zero_shift_is_identity(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 0.0)
        np.testing.assert_allclose(y, x, rtol=1e-12)

    # ---- +1 semitone: 440 → 466.16 Hz ----

    def test_plus_1_semitone_frequency(self):
        x = _sine(440.0, 2.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 1.0)
        dom = _dominant_freq(y, self.SR)
        expected = 440.0 * semitones_to_ratio(1)  # ≈ 466.16 Hz
        assert abs(dom - expected) <= self._freq_tol(expected), (
            f"Expected ≈{expected:.2f} Hz, got {dom:.2f} Hz"
        )

    def test_plus_1_semitone_duration(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 1.0)
        dur_x = _duration_s(x, self.SR)
        dur_y = _duration_s(y, self.SR)
        assert abs(dur_y - dur_x) < 0.05, (
            f"Duration changed: {dur_x:.4f}s → {dur_y:.4f}s"
        )

    # ---- +12 semitones: 440 → ~880 Hz ----

    def test_plus_12_semitones_frequency(self):
        x = _sine(440.0, 2.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 12.0)
        dom = _dominant_freq(y, self.SR)
        expected = 880.0
        assert abs(dom - expected) <= self._freq_tol(expected), (
            f"Expected ≈880 Hz, got {dom:.2f} Hz"
        )

    def test_plus_12_semitones_duration(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 12.0)
        dur_x = _duration_s(x, self.SR)
        dur_y = _duration_s(y, self.SR)
        assert abs(dur_y - dur_x) < 0.05

    # ---- -12 semitones: 440 → ~220 Hz ----

    def test_minus_12_semitones_frequency(self):
        x = _sine(440.0, 2.0, self.SR)
        y = apply_pitch_shift(x, self.SR, -12.0)
        dom = _dominant_freq(y, self.SR)
        expected = 220.0
        assert abs(dom - expected) <= self._freq_tol(expected), (
            f"Expected ≈220 Hz, got {dom:.2f} Hz"
        )

    def test_minus_12_semitones_duration(self):
        x = _sine(440.0, 1.0, self.SR)
        y = apply_pitch_shift(x, self.SR, -12.0)
        dur_x = _duration_s(x, self.SR)
        dur_y = _duration_s(y, self.SR)
        assert abs(dur_y - dur_x) < 0.05

    # ---- Silence ----

    def test_silence_returns_finite(self):
        x = np.zeros(self.SR, dtype=np.float64)
        y = apply_pitch_shift(x, self.SR, 12.0)
        assert np.all(np.isfinite(y))

    def test_silence_is_approximately_zero(self):
        x = np.zeros(self.SR, dtype=np.float64)
        y = apply_pitch_shift(x, self.SR, 12.0)
        assert np.max(np.abs(y)) < 1e-10

    # ---- Short signals ----

    def test_short_signal_does_not_crash(self):
        x = _sine(440.0, 0.05, self.SR)
        y = apply_pitch_shift(x, self.SR, 7.0)
        assert len(y) > 0
        assert np.all(np.isfinite(y))

    # ---- Stereo ----

    def test_stereo_shape(self):
        x = _stereo(440.0, 880.0, 1.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 12.0)
        assert y.ndim == 2
        assert y.shape[1] == 2

    def test_stereo_duration(self):
        x = _stereo(440.0, 880.0, 1.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 12.0)
        dur_x = _duration_s(x, self.SR)
        dur_y = _duration_s(y, self.SR)
        assert abs(dur_y - dur_x) < 0.05

    def test_stereo_left_frequency_shifted(self):
        x = _stereo(440.0, 220.0, 2.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 12.0)
        dom_l = _dominant_freq(y[:, 0], self.SR)
        assert abs(dom_l - 880.0) <= self._freq_tol(880.0)

    def test_stereo_right_frequency_shifted(self):
        x = _stereo(440.0, 220.0, 2.0, self.SR)
        y = apply_pitch_shift(x, self.SR, 12.0)
        dom_r = _dominant_freq(y[:, 1], self.SR)
        assert abs(dom_r - 440.0) <= self._freq_tol(440.0)

    # ---- Multiple sample rates ----

    def test_multiple_sample_rates(self):
        for sr in [16000, 22050, 44100, 48000]:
            x = _sine(200.0, 1.0, sr)
            y = apply_pitch_shift(x, sr, 12.0)
            assert np.all(np.isfinite(y))

    # ---- Output is finite ----

    def test_output_finite(self):
        x = _sine(440.0, 1.0, self.SR)
        for s in [-12.0, -7.0, 0.0, 7.0, 12.0]:
            y = apply_pitch_shift(x, self.SR, s)
            assert np.all(np.isfinite(y)), f"semitones={s} produced NaN/Inf"

    # ---- Invalid parameters ----

    def test_nan_semitones_raises(self):
        with pytest.raises(ValueError):
            apply_pitch_shift(_sine(), self.SR, math.nan)

    def test_inf_semitones_raises(self):
        with pytest.raises(ValueError):
            apply_pitch_shift(_sine(), self.SR, math.inf)

    def test_nan_input_raises(self):
        x = _sine()
        x[50] = np.nan
        with pytest.raises(VoiceDSPError):
            apply_pitch_shift(x, self.SR, 7.0)
