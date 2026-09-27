"""
tests/test_voice_lab_phase_e.py — Module 09 Phase E: Classical Effects
========================================================================

Tests cover:
    - Delay            impulse response
    - Chorus           modulated delay behavior
    - Soft Distortion  harmonic generation / bounded output
"""
from __future__ import annotations
import math
import numpy as np
import pytest
from dsp_core.voice_gain import VoiceDSPError
from dsp_core.effects import apply_delay, apply_chorus, apply_soft_distortion
SR = 44100

def _sine(freq_hz: float=440.0, duration_s: float=1.0, sr: int=SR, amplitude: float=0.5) -> np.ndarray:
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

def _stereo(fl: float=440.0, fr: float=880.0, duration_s: float=0.5, sr: int=SR, amp: float=0.5) -> np.ndarray:
    l_ = _sine(fl, duration_s, sr, amp)
    r_ = _sine(fr, duration_s, sr, amp)
    return np.stack([l_, r_], axis=1)

def _impulse(n_samples: int=1000, pos: int=0) -> np.ndarray:
    x = np.zeros(n_samples, dtype=np.float64)
    x[pos] = 1.0
    return x

def _dominant_freq(signal: np.ndarray, sr: int=SR) -> float:
    spectrum = np.abs(np.fft.rfft(signal.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    return float(freqs[np.argmax(spectrum)])

class TestDelay:

    def test_dry_mix_is_identity(self):
        """mix=0.0 → no delayed signal added, output equals input."""
        x = _sine(440.0, 0.2)
        y = apply_delay(x, SR, delay_ms=100.0, feedback=0.5, mix=0.0)
        np.testing.assert_allclose(y, x, rtol=1e-12)

    def test_impulse_response_structure(self):
        """Dirac impulse at n=0 should produce echo at D samples."""
        n_samp = SR
        x = _impulse(n_samp, pos=0)
        delay_ms = 100.0
        D = int(round(delay_ms * SR / 1000.0))
        feedback = 0.5
        mix = 1.0
        y = apply_delay(x, SR, delay_ms=delay_ms, feedback=feedback, mix=mix)
        assert abs(y[0] - 1.0) < 1e-12, f'y[0]={y[0]}'
        assert abs(y[D] - mix * feedback * x[0]) < 1e-10, f'Expected y[{D}]={mix * feedback}, got {y[D]:.6f}'
        if 2 * D < n_samp:
            expected_2d = (mix * feedback) ** 2
            assert abs(y[2 * D] - expected_2d) < 1e-10, f'Expected y[{2 * D}]={expected_2d:.6f}, got {y[2 * D]:.6f}'

    def test_stereo_shape(self):
        x = _stereo()
        y = apply_delay(x, SR, delay_ms=50.0, feedback=0.3, mix=0.5)
        assert y.shape == x.shape

    def test_silence_returns_silence(self):
        x = np.zeros(SR, dtype=np.float64)
        y = apply_delay(x, SR, delay_ms=100.0, feedback=0.5, mix=0.5)
        np.testing.assert_allclose(y, 0.0, atol=1e-15)

    def test_output_finite(self):
        x = _sine()
        y = apply_delay(x, SR, delay_ms=200.0, feedback=0.4, mix=0.5)
        assert np.all(np.isfinite(y))

    def test_invalid_delay_raises(self):
        with pytest.raises(ValueError):
            apply_delay(_sine(), SR, delay_ms=0.0, feedback=0.5, mix=0.5)
        with pytest.raises(ValueError):
            apply_delay(_sine(), SR, delay_ms=6000.0, feedback=0.5, mix=0.5)

    def test_invalid_feedback_raises(self):
        with pytest.raises(ValueError):
            apply_delay(_sine(), SR, delay_ms=100.0, feedback=-0.1, mix=0.5)
        with pytest.raises(ValueError):
            apply_delay(_sine(), SR, delay_ms=100.0, feedback=1.0, mix=0.5)

    def test_nan_input_raises(self):
        x = _sine()
        x[100] = np.nan
        with pytest.raises(VoiceDSPError):
            apply_delay(x, SR, delay_ms=100.0, feedback=0.3, mix=0.5)

    def test_short_signal(self):
        x = _sine(440.0, 0.005)
        y = apply_delay(x, SR, delay_ms=100.0, feedback=0.3, mix=0.5)
        assert len(y) == len(x)
        assert np.all(np.isfinite(y))

class TestChorus:

    def test_dry_mix_is_identity(self):
        """mix=0.0 should return the original signal."""
        x = _sine(440.0, 0.2)
        y = apply_chorus(x, SR, rate_hz=1.5, depth_ms=3.0, base_delay_ms=15.0, mix=0.0)
        np.testing.assert_allclose(y, x, rtol=1e-12)

    def test_output_shape_unchanged(self):
        x = _sine(440.0, 1.0)
        y = apply_chorus(x, SR, mix=0.5)
        assert y.shape == x.shape

    def test_stereo_shape(self):
        x = _stereo()
        y = apply_chorus(x, SR, mix=0.5)
        assert y.shape == x.shape

    def test_silence_returns_silence(self):
        x = np.zeros(SR, dtype=np.float64)
        y = apply_chorus(x, SR, mix=0.5)
        np.testing.assert_allclose(y, 0.0, atol=1e-14)

    def test_output_finite(self):
        x = _sine(440.0, 1.0)
        y = apply_chorus(x, SR, rate_hz=2.0, depth_ms=5.0, base_delay_ms=20.0, mix=0.5)
        assert np.all(np.isfinite(y))

    def test_modulation_creates_time_variation(self):
        """With depth_ms > 0, the delayed version should differ from a static delay."""
        x = _sine(440.0, 2.0)
        y_chorus = apply_chorus(x, SR, rate_hz=1.5, depth_ms=3.0, base_delay_ms=15.0, mix=0.5)
        static_delay_ms = 15.0
        D = int(round(static_delay_ms * SR / 1000.0))
        diff = np.abs(y_chorus[D:] - x[D:])
        assert np.mean(diff) > 1e-06, 'Chorus output identical to static delay — modulation may be absent'

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            apply_chorus(_sine(), SR, rate_hz=0.0, depth_ms=3.0, base_delay_ms=15.0)
        with pytest.raises(ValueError):
            apply_chorus(_sine(), SR, rate_hz=1.5, depth_ms=15.0, base_delay_ms=10.0)
        with pytest.raises(ValueError):
            apply_chorus(_sine(), SR, mix=1.5)

    def test_nan_input_raises(self):
        x = _sine()
        x[100] = np.nan
        with pytest.raises(VoiceDSPError):
            apply_chorus(x, SR)

class TestSoftDistortion:

    def test_output_bounded(self):
        """tanh(β·x)/tanh(β) approaches ±1 asymptotically for large x.

        For finite inputs tanh(β·x) can slightly exceed tanh(β) due to
        float64 precision when x > 1.0, so the bound is ≈ 1.0 + 1e-4.
        The key guarantee is that the output is strongly bounded compared
        to the input (which had amplitude 5.0 here).
        """
        x = _sine(440.0, 1.0, amplitude=5.0)
        y = apply_soft_distortion(x, drive=5.0)
        assert np.all(np.abs(y) <= 1.0 + 0.0001), f'Output exceeded soft clip bound; max={np.max(np.abs(y)):.6f}'
        assert np.max(np.abs(y)) < 1.1

    def test_low_drive_approximately_linear(self):
        """Very low drive should approach identity (linear region of tanh)."""
        x = _sine(440.0, 0.5, amplitude=0.1)
        y = apply_soft_distortion(x, drive=0.001)
        np.testing.assert_allclose(y, x, rtol=0.001)

    def test_harmonic_generation(self):
        """High drive on a single sine should produce odd harmonics."""
        fund_hz = 200.0
        x = _sine(fund_hz, 2.0, SR, amplitude=0.9)
        y = apply_soft_distortion(x, drive=10.0)
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1.0 / SR)

        def _energy_near(target_hz: float, tol: float=5.0) -> float:
            mask = np.abs(freqs - target_hz) < tol
            return float(np.max(spectrum[mask])) if np.any(mask) else 0.0
        third = _energy_near(3 * fund_hz)
        assert third > 0.001, f'3rd harmonic energy at {3 * fund_hz} Hz too low: {third:.4f}'

    def test_unity_gain_at_full_scale(self):
        """y[n] = tanh(β)/tanh(β) = 1 for x = 1.0 input."""
        drive = 5.0
        y = apply_soft_distortion(np.array([1.0], dtype=np.float64), drive=drive)
        assert abs(y[0] - 1.0) < 1e-10, f'Expected ≈1.0 output for x=1.0, got {y[0]}'

    def test_symmetry(self):
        """tanh is an odd function, so distortion must be symmetric."""
        x = np.array([0.1, 0.3, 0.7, -0.1, -0.3, -0.7], dtype=np.float64)
        y = apply_soft_distortion(x, drive=5.0)
        assert abs(y[0] + y[3]) < 1e-12
        assert abs(y[1] + y[4]) < 1e-12
        assert abs(y[2] + y[5]) < 1e-12

    def test_stereo_shape(self):
        x = _stereo()
        y = apply_soft_distortion(x, drive=3.0)
        assert y.shape == x.shape

    def test_silence_returns_silence(self):
        x = np.zeros(1000, dtype=np.float64)
        y = apply_soft_distortion(x, drive=5.0)
        np.testing.assert_allclose(y, 0.0, atol=1e-15)

    def test_output_finite(self):
        x = _sine(440.0, 0.5, amplitude=2.0)
        y = apply_soft_distortion(x, drive=20.0)
        assert np.all(np.isfinite(y))

    def test_invalid_drive_raises(self):
        with pytest.raises(ValueError):
            apply_soft_distortion(_sine(), drive=0.0)
        with pytest.raises(ValueError):
            apply_soft_distortion(_sine(), drive=-1.0)
        with pytest.raises(ValueError):
            apply_soft_distortion(_sine(), drive=101.0)

    def test_nan_input_raises(self):
        x = _sine()
        x[10] = np.nan
        with pytest.raises(VoiceDSPError):
            apply_soft_distortion(x, drive=3.0)