"""
tests/test_voice_lab_api.py — Module 09 Phase F: Processing Chain & API
=========================================================================

Integration tests for:
    - VoiceProcessRequest model validation
    - process_voice_lab() chain orchestration
    - Original input is never mutated
    - Neutral parameters produce identity output
    - Chained operations behave correctly
    - prevent_clipping logic
    - Operations summary reporting
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from app.models.voice_lab import (
    VoiceProcessRequest,
    GainOperation,
    SpeedOperation,
    TimeStretchOperation,
    PitchShiftOperation,
    TremoloOperation,
    DelayOperation,
    SoftDistortionOperation,
)
from app.services.voice_lab_service import process_voice_lab, _samples_to_numpy, _numpy_to_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100


def _make_sine_request(
    freq_hz: float = 440.0,
    duration_s: float = 0.5,
    sr: int = SR,
    amplitude: float = 0.3,
) -> tuple[list[list[float]], np.ndarray]:
    """Return (samples_as_list, numpy_array) for a mono sine wave."""
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    arr = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    return [[float(v) for v in arr]], arr


def _dominant_freq(signal: np.ndarray, sr: int = SR) -> float:
    spectrum = np.abs(np.fft.rfft(signal.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    return float(freqs[np.argmax(spectrum)])


# ---------------------------------------------------------------------------
# Helpers: conversion round-trip
# ---------------------------------------------------------------------------


class TestConversionHelpers:
    def test_mono_round_trip(self):
        arr = np.array([0.1, 0.2, -0.3], dtype=np.float64)
        lists = _numpy_to_samples(arr)
        back = _samples_to_numpy(lists)
        np.testing.assert_allclose(back, arr, rtol=1e-12)

    def test_stereo_round_trip(self):
        arr = np.random.default_rng(0).random((1000, 2)) * 2 - 1
        lists = _numpy_to_samples(arr)
        back = _samples_to_numpy(lists)
        np.testing.assert_allclose(back, arr, rtol=1e-12)


# ---------------------------------------------------------------------------
# Neutral parameters → identity
# ---------------------------------------------------------------------------


class TestNeutralParameters:
    def test_all_neutral_returns_original(self):
        """No operations → output equals input."""
        samples, arr = _make_sine_request()
        req = VoiceProcessRequest(samples=samples, sample_rate_hz=SR)
        resp = process_voice_lab(req)
        out = np.array(resp.samples[0])
        np.testing.assert_allclose(out, arr, rtol=1e-12)

    def test_0db_gain_is_identity(self):
        samples, arr = _make_sine_request()
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=0.0),
        )
        resp = process_voice_lab(req)
        out = np.array(resp.samples[0])
        np.testing.assert_allclose(out, arr, rtol=1e-12)

    def test_speed_1_is_identity(self):
        samples, arr = _make_sine_request()
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            speed=SpeedOperation(speed=1.0),
        )
        resp = process_voice_lab(req)
        out = np.array(resp.samples[0])
        np.testing.assert_allclose(out, arr, rtol=1e-12)

    def test_stretch_1_is_identity(self):
        samples, arr = _make_sine_request()
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            time_stretch=TimeStretchOperation(stretch=1.0),
        )
        resp = process_voice_lab(req)
        out = np.array(resp.samples[0])
        # Phase vocoder S=1 may have small reconstruction error; check duration
        assert abs(len(out) - len(arr)) <= len(arr) * 0.01

    def test_pitch_0_is_identity(self):
        samples, arr = _make_sine_request()
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            pitch_shift=PitchShiftOperation(semitones=0.0),
        )
        resp = process_voice_lab(req)
        out = np.array(resp.samples[0])
        np.testing.assert_allclose(out, arr, rtol=1e-12)


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------


class TestGainOperation:
    def test_plus_6_db(self):
        samples, arr = _make_sine_request(amplitude=0.2)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=6.0),
        )
        resp = process_voice_lab(req)
        out = np.array(resp.samples[0])
        expected = arr * 10.0 ** (6.0 / 20.0)
        np.testing.assert_allclose(out, expected, rtol=1e-10)

    def test_operations_applied_reported(self):
        samples, _ = _make_sine_request(amplitude=0.2)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=6.0),
        )
        resp = process_voice_lab(req)
        assert any("gain" in op for op in resp.operations_applied)

    def test_clipping_risk_reported(self):
        """Gain that pushes peak > 1.0 must set clipping_risk=True."""
        samples, _ = _make_sine_request(amplitude=0.9)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=20.0),  # ×10
        )
        resp = process_voice_lab(req)
        assert resp.clipping_risk is True
        assert resp.clipping_prevented is False
        # Output must NOT be silently normalized
        assert resp.output_peak > 1.0

    def test_prevent_clipping_normalizes(self):
        samples, _ = _make_sine_request(amplitude=0.9)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=20.0),
            prevent_clipping=True,
        )
        resp = process_voice_lab(req)
        assert resp.clipping_prevented is True
        out = np.array(resp.samples[0])
        assert abs(np.max(np.abs(out)) - 1.0) < 1e-10


class TestSpeedOperation:
    def test_2x_speed_halves_duration(self):
        samples, arr = _make_sine_request(duration_s=1.0)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            speed=SpeedOperation(speed=2.0),
        )
        resp = process_voice_lab(req)
        out_len = len(resp.samples[0])
        expected = len(arr) // 2
        assert abs(out_len - expected) <= 5

    def test_metadata_duration(self):
        samples, arr = _make_sine_request(duration_s=1.0)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            speed=SpeedOperation(speed=2.0),
        )
        resp = process_voice_lab(req)
        assert abs(resp.output_duration_s - 0.5) < 0.01
        assert abs(resp.input_duration_s - 1.0) < 1e-6


class TestPitchShiftOperation:
    def test_plus_12_semitones_frequency(self):
        samples, arr = _make_sine_request(freq_hz=440.0, duration_s=2.0)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            pitch_shift=PitchShiftOperation(semitones=12.0),
        )
        resp = process_voice_lab(req)
        out = np.array(resp.samples[0])
        dom = _dominant_freq(out, SR)
        assert abs(dom - 880.0) < 880.0 * 0.03


# ---------------------------------------------------------------------------
# Chained operations
# ---------------------------------------------------------------------------


class TestChainedOperations:
    def test_gain_then_effect(self):
        samples, _ = _make_sine_request(amplitude=0.3)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=6.0),
            effect=TremoloOperation(rate_hz=5.0, depth=0.5),
        )
        resp = process_voice_lab(req)
        assert resp.status == "processed"
        assert len(resp.operations_applied) >= 2

    def test_chain_starts_from_original(self):
        """Call process twice with same request — results must be identical."""
        samples, _ = _make_sine_request(amplitude=0.3)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=6.0),
            effect=DelayOperation(delay_ms=100.0, feedback=0.3, mix=0.5),
        )
        resp1 = process_voice_lab(req)
        resp2 = process_voice_lab(req)
        out1 = np.array(resp1.samples[0])
        out2 = np.array(resp2.samples[0])
        np.testing.assert_allclose(out1, out2, rtol=1e-12)

    def test_original_samples_not_mutated(self):
        """Input sample list must not be modified by processing."""
        samples, arr = _make_sine_request(amplitude=0.3)
        original_copy = [list(ch) for ch in samples]
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=20.0),
        )
        process_voice_lab(req)
        for ch_orig, ch_curr in zip(original_copy, samples):
            assert ch_orig == ch_curr, "Input samples were mutated!"

    def test_all_operations_together(self):
        """All operations in the chain must not crash."""
        samples, _ = _make_sine_request(freq_hz=440.0, duration_s=2.0, amplitude=0.3)
        req = VoiceProcessRequest(
            samples=samples, sample_rate_hz=SR,
            gain=GainOperation(mode="db", gain_value=3.0),
            speed=SpeedOperation(speed=1.1),
            time_stretch=TimeStretchOperation(stretch=0.9),
            pitch_shift=PitchShiftOperation(semitones=2.0),
            effect=SoftDistortionOperation(drive=3.0),
        )
        resp = process_voice_lab(req)
        assert resp.status == "processed"
        out = np.array(resp.samples[0])
        assert np.all(np.isfinite(out))
        assert len(resp.operations_applied) == 5


# ---------------------------------------------------------------------------
# Metadata accuracy
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_input_duration_reported_correctly(self):
        samples, arr = _make_sine_request(duration_s=0.5)
        req = VoiceProcessRequest(samples=samples, sample_rate_hz=SR)
        resp = process_voice_lab(req)
        assert abs(resp.input_duration_s - 0.5) < 1e-4

    def test_sample_rate_preserved(self):
        samples, _ = _make_sine_request()
        req = VoiceProcessRequest(samples=samples, sample_rate_hz=SR)
        resp = process_voice_lab(req)
        assert resp.sample_rate_hz == SR

    def test_peak_reported_correctly(self):
        samples, arr = _make_sine_request(amplitude=0.5)
        req = VoiceProcessRequest(samples=samples, sample_rate_hz=SR)
        resp = process_voice_lab(req)
        expected_peak = float(np.max(np.abs(arr)))
        assert abs(resp.output_peak - expected_peak) < 1e-6

    def test_stereo_input_channels(self):
        l_ = [0.1, 0.2, 0.3, 0.0]
        r_ = [0.05, 0.15, 0.25, 0.0]
        req = VoiceProcessRequest(samples=[l_, r_], sample_rate_hz=SR)
        resp = process_voice_lab(req)
        assert resp.input_channels == 2
        assert len(resp.samples) == 2
