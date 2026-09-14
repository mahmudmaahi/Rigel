"""
tests/test_filtering.py — Module 06: Audio Filtering Tests
============================================================

Comprehensive DSP behaviour tests for the Butterworth IIR filter.

Each test validates actual DSP behaviour using synthetic signals rather than
only checking that the code runs without error.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile as scipy_wavfile

from app.main import app
from dsp_core.filtering import (
    FilterDesignError,
    FilterSpec,
    apply_filter,
    compute_frequency_response,
    design_filter,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 44100  # default sample rate for tests


def make_mono_sine(freq_hz: float, duration_s: float = 1.0, fs: int = FS) -> np.ndarray:
    """Return a float32 mono sine wave with amplitude 0.5."""
    t = np.arange(int(fs * duration_s)) / fs
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack two mono signals into a (n_samples, 2) stereo array."""
    return np.column_stack([left, right]).astype(np.float32)


def sine_power(signal: np.ndarray) -> float:
    """Return mean-squared power of the signal."""
    if signal.ndim > 1:
        return float(np.mean(signal**2))
    return float(np.mean(signal**2))


def make_wav_bytes(samples: np.ndarray, fs: int = FS) -> bytes:
    """Encode samples to WAV bytes in memory."""
    buf = io.BytesIO()
    scipy_wavfile.write(buf, fs, samples)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# FilterSpec / design_filter — validation tests
# ---------------------------------------------------------------------------

class TestFilterSpecValidation:
    def test_lowpass_valid(self):
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=1000.0)
        result = design_filter(spec)
        assert result.sos.shape[1] == 6  # SOS rows always have 6 coefficients
        assert result.n_sections > 0

    def test_highpass_valid(self):
        spec = FilterSpec("highpass", "butterworth", 4, FS, cutoff_hz=5000.0)
        result = design_filter(spec)
        assert result.sos.shape[1] == 6

    def test_bandpass_valid(self):
        spec = FilterSpec("bandpass", "butterworth", 4, FS, low_hz=500.0, high_hz=3000.0)
        result = design_filter(spec)
        assert result.sos.shape[1] == 6

    def test_bandstop_valid(self):
        spec = FilterSpec("bandstop", "butterworth", 4, FS, low_hz=500.0, high_hz=3000.0)
        result = design_filter(spec)
        assert result.sos.shape[1] == 6

    def test_order_zero_rejected(self):
        spec = FilterSpec("lowpass", "butterworth", 0, FS, cutoff_hz=1000.0)
        with pytest.raises(FilterDesignError, match="≥ 1"):
            design_filter(spec)

    def test_order_too_high_rejected(self):
        spec = FilterSpec("lowpass", "butterworth", 21, FS, cutoff_hz=1000.0)
        with pytest.raises(FilterDesignError, match="≤ 20"):
            design_filter(spec)

    def test_cutoff_at_nyquist_rejected(self):
        nyquist = FS / 2
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=nyquist)
        with pytest.raises(FilterDesignError, match="Nyquist"):
            design_filter(spec)

    def test_cutoff_above_nyquist_rejected(self):
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=FS)
        with pytest.raises(FilterDesignError, match="Nyquist"):
            design_filter(spec)

    def test_cutoff_zero_rejected(self):
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=0.0)
        with pytest.raises(FilterDesignError, match="positive"):
            design_filter(spec)

    def test_cutoff_negative_rejected(self):
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=-500.0)
        with pytest.raises(FilterDesignError, match="positive"):
            design_filter(spec)

    def test_band_low_geq_high_rejected(self):
        spec = FilterSpec("bandpass", "butterworth", 4, FS, low_hz=3000.0, high_hz=1000.0)
        with pytest.raises(FilterDesignError, match="strictly less than"):
            design_filter(spec)

    def test_band_high_at_nyquist_rejected(self):
        nyquist = FS / 2
        spec = FilterSpec("bandpass", "butterworth", 4, FS, low_hz=1000.0, high_hz=nyquist)
        with pytest.raises(FilterDesignError, match="Nyquist"):
            design_filter(spec)

    def test_missing_cutoff_for_lowpass(self):
        spec = FilterSpec("lowpass", "butterworth", 4, FS)
        with pytest.raises(FilterDesignError, match="cutoff_hz"):
            design_filter(spec)

    def test_missing_band_freqs_for_bandpass(self):
        spec = FilterSpec("bandpass", "butterworth", 4, FS, cutoff_hz=1000.0)
        with pytest.raises(FilterDesignError, match="low_hz"):
            design_filter(spec)

    def test_unknown_family_rejected(self):
        spec = FilterSpec("lowpass", "unknown_family", 4, FS, cutoff_hz=1000.0)  # type: ignore[arg-type]
        with pytest.raises(FilterDesignError, match="filter family"):
            design_filter(spec)

    def test_invalid_sample_rate(self):
        spec = FilterSpec("lowpass", "butterworth", 4, 0, cutoff_hz=1000.0)
        with pytest.raises(FilterDesignError, match="Sample rate"):
            design_filter(spec)

    def test_multiple_sample_rates(self):
        for fs in [8000, 16000, 22050, 44100, 48000, 96000]:
            spec = FilterSpec("lowpass", "butterworth", 4, fs, cutoff_hz=fs / 4)
            result = design_filter(spec)
            assert result.sos.shape[1] == 6


# ---------------------------------------------------------------------------
# SOS structure tests
# ---------------------------------------------------------------------------

class TestSOSStructure:
    def test_sos_shape_order_1(self):
        """Order-1 Butterworth should have exactly 1 SOS section."""
        spec = FilterSpec("lowpass", "butterworth", 1, FS, cutoff_hz=1000.0)
        result = design_filter(spec)
        assert result.n_sections == 1

    def test_sos_shape_order_4(self):
        """Order-4 Butterworth → 2 SOS sections (each order-2)."""
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=1000.0)
        result = design_filter(spec)
        assert result.n_sections == 2

    def test_sos_shape_order_5(self):
        """Order-5 Butterworth → 3 SOS sections (one order-1, two order-2)."""
        spec = FilterSpec("lowpass", "butterworth", 5, FS, cutoff_hz=1000.0)
        result = design_filter(spec)
        assert result.n_sections == 3

    def test_sos_values_finite(self):
        spec = FilterSpec("lowpass", "butterworth", 6, FS, cutoff_hz=1000.0)
        result = design_filter(spec)
        assert np.all(np.isfinite(result.sos))


# ---------------------------------------------------------------------------
# DSP behaviour tests: lowpass
# ---------------------------------------------------------------------------

class TestLowpassBehavior:
    """Verify that a low-pass filter passes low frequencies and attenuates high ones."""

    CUTOFF = 2000.0  # Hz
    ORDER = 4

    @classmethod
    def setup_class(cls):
        spec = FilterSpec("lowpass", "butterworth", cls.ORDER, FS, cutoff_hz=cls.CUTOFF)
        cls.design = design_filter(spec)

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        return apply_filter(self.design, signal)

    def test_low_freq_passes(self):
        """A 200 Hz sine should pass through with little attenuation."""
        sig = make_mono_sine(200.0)
        filtered = self._filter(sig)
        # Passband should preserve most power (within 3 dB)
        ratio = sine_power(filtered) / sine_power(sig)
        assert ratio > 0.5, f"Low-frequency content was too attenuated: ratio={ratio:.3f}"

    def test_high_freq_attenuated(self):
        """A 10 kHz sine (5× above cutoff) should be strongly attenuated."""
        sig = make_mono_sine(10000.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / (sine_power(sig) + 1e-12)
        assert ratio < 0.1, f"High-frequency content was not sufficiently attenuated: ratio={ratio:.3f}"

    def test_mixed_signal(self):
        """Low component should be preserved; high component should be removed."""
        low = make_mono_sine(200.0)
        high = make_mono_sine(10000.0)
        mixed = (low + high).astype(np.float32)
        filtered = self._filter(mixed)
        # The filtered signal should be much closer to low than to high
        power_diff_from_low = np.mean((filtered - low) ** 2)
        power_diff_from_high = np.mean((filtered - high) ** 2)
        assert power_diff_from_low < power_diff_from_high

    def test_output_length_preserved(self):
        sig = make_mono_sine(200.0)
        filtered = self._filter(sig)
        assert len(filtered) == len(sig)

    def test_output_finite(self):
        sig = make_mono_sine(200.0)
        filtered = self._filter(sig)
        assert np.all(np.isfinite(filtered))

    def test_output_shape_mono(self):
        sig = make_mono_sine(200.0)
        assert sig.ndim == 1
        filtered = self._filter(sig)
        assert filtered.ndim == 1

    def test_dc_passed(self):
        """DC (0 Hz) should pass through a low-pass filter."""
        dc = np.full(FS, 0.5, dtype=np.float32)
        filtered = self._filter(dc)
        # After edge transient, the signal should be close to 0.5
        tail = filtered[FS // 4 :]  # skip the first quarter for transient
        assert np.abs(np.mean(tail) - 0.5) < 0.05


# ---------------------------------------------------------------------------
# DSP behaviour tests: highpass
# ---------------------------------------------------------------------------

class TestHighpassBehavior:
    CUTOFF = 2000.0
    ORDER = 4

    @classmethod
    def setup_class(cls):
        spec = FilterSpec("highpass", "butterworth", cls.ORDER, FS, cutoff_hz=cls.CUTOFF)
        cls.design = design_filter(spec)

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        return apply_filter(self.design, signal)

    def test_high_freq_passes(self):
        """A 10 kHz sine (far above cutoff) should pass through."""
        sig = make_mono_sine(10000.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / sine_power(sig)
        assert ratio > 0.5

    def test_low_freq_attenuated(self):
        """A 200 Hz sine (well below cutoff) should be attenuated."""
        sig = make_mono_sine(200.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / (sine_power(sig) + 1e-12)
        assert ratio < 0.1

    def test_dc_blocked(self):
        """DC should be blocked by a highpass filter."""
        dc = np.full(FS, 0.5, dtype=np.float32)
        filtered = self._filter(dc)
        tail = filtered[FS // 4 :]
        assert np.abs(np.mean(tail)) < 0.05

    def test_output_length_preserved(self):
        sig = make_mono_sine(5000.0)
        assert len(apply_filter(self.design, sig)) == len(sig)

    def test_output_finite(self):
        sig = make_mono_sine(5000.0)
        assert np.all(np.isfinite(apply_filter(self.design, sig)))


# ---------------------------------------------------------------------------
# DSP behaviour tests: bandpass
# ---------------------------------------------------------------------------

class TestBandpassBehavior:
    LOW = 1000.0
    HIGH = 5000.0
    ORDER = 4

    @classmethod
    def setup_class(cls):
        spec = FilterSpec("bandpass", "butterworth", cls.ORDER, FS, low_hz=cls.LOW, high_hz=cls.HIGH)
        cls.design = design_filter(spec)

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        return apply_filter(self.design, signal)

    def test_passband_freq_passes(self):
        """A 3 kHz sine (centre of band) should pass."""
        sig = make_mono_sine(3000.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / sine_power(sig)
        assert ratio > 0.25  # bandpass gain can vary; allow moderate attenuation

    def test_below_band_attenuated(self):
        """A 100 Hz sine (well below band) should be attenuated."""
        sig = make_mono_sine(100.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / (sine_power(sig) + 1e-12)
        assert ratio < 0.1

    def test_above_band_attenuated(self):
        """A 15 kHz sine (well above band) should be attenuated."""
        sig = make_mono_sine(15000.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / (sine_power(sig) + 1e-12)
        assert ratio < 0.1

    def test_output_length_preserved(self):
        sig = make_mono_sine(3000.0)
        assert len(self._filter(sig)) == len(sig)


# ---------------------------------------------------------------------------
# DSP behaviour tests: bandstop / notch
# ---------------------------------------------------------------------------

class TestBandstopBehavior:
    LOW = 1000.0
    HIGH = 5000.0
    ORDER = 4

    @classmethod
    def setup_class(cls):
        spec = FilterSpec("bandstop", "butterworth", cls.ORDER, FS, low_hz=cls.LOW, high_hz=cls.HIGH)
        cls.design = design_filter(spec)

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        return apply_filter(self.design, signal)

    def test_below_band_passes(self):
        """A 200 Hz sine (below stopband) should pass through."""
        sig = make_mono_sine(200.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / sine_power(sig)
        assert ratio > 0.5

    def test_above_band_passes(self):
        """A 15 kHz sine (above stopband) should pass through."""
        sig = make_mono_sine(15000.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / sine_power(sig)
        assert ratio > 0.5

    def test_in_band_attenuated(self):
        """A 2.5 kHz sine (centre of notch) should be attenuated."""
        sig = make_mono_sine(2500.0)
        filtered = self._filter(sig)
        ratio = sine_power(filtered) / (sine_power(sig) + 1e-12)
        assert ratio < 0.2


# ---------------------------------------------------------------------------
# Stereo processing tests
# ---------------------------------------------------------------------------

class TestStereoProcessing:
    CUTOFF = 2000.0
    ORDER = 4

    @classmethod
    def setup_class(cls):
        spec = FilterSpec("lowpass", "butterworth", cls.ORDER, FS, cutoff_hz=cls.CUTOFF)
        cls.design = design_filter(spec)

    def test_stereo_output_shape(self):
        left = make_mono_sine(200.0)
        right = make_mono_sine(5000.0)
        stereo = make_stereo(left, right)
        filtered = apply_filter(self.design, stereo)
        assert filtered.shape == stereo.shape

    def test_stereo_channels_independent(self):
        """Left channel (200 Hz low) should be preserved; right (5 kHz high) attenuated."""
        left = make_mono_sine(200.0)
        right = make_mono_sine(5000.0)
        stereo = make_stereo(left, right)
        filtered = apply_filter(self.design, stereo)

        left_out = filtered[:, 0]
        right_out = filtered[:, 1]

        # Left (low freq) mostly preserved
        left_ratio = sine_power(left_out) / sine_power(left)
        assert left_ratio > 0.5

        # Right (high freq) strongly attenuated
        right_ratio = sine_power(right_out) / (sine_power(right) + 1e-12)
        assert right_ratio < 0.1

    def test_stereo_output_finite(self):
        left = make_mono_sine(200.0)
        right = make_mono_sine(5000.0)
        filtered = apply_filter(self.design, make_stereo(left, right))
        assert np.all(np.isfinite(filtered))


# ---------------------------------------------------------------------------
# Short signal handling
# ---------------------------------------------------------------------------

class TestShortSignals:
    def test_signal_too_short_raises(self):
        """A very short signal should raise FilterDesignError (sosfiltfilt padding failure)."""
        spec = FilterSpec("lowpass", "butterworth", 8, FS, cutoff_hz=1000.0)
        design = design_filter(spec)
        # 5 samples is far too short for an 8-section filter
        tiny = np.array([0.1, -0.1, 0.2, -0.2, 0.1], dtype=np.float32)
        with pytest.raises(FilterDesignError, match="too short"):
            apply_filter(design, tiny)

    def test_sufficient_length_works(self):
        """A long enough signal should always filter without error."""
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=1000.0)
        design = design_filter(spec)
        # 1000 samples should be more than sufficient for a 2-section filter
        sig = np.random.randn(1000).astype(np.float32)
        filtered = apply_filter(design, sig)
        assert len(filtered) == len(sig)


# ---------------------------------------------------------------------------
# Frequency response tests
# ---------------------------------------------------------------------------

class TestFrequencyResponse:
    def test_response_length(self):
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=1000.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design, n_points=256)
        assert len(resp.frequencies_hz) == 256
        assert len(resp.magnitude_db) == 256

    def test_response_frequencies_ascending(self):
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=1000.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        freqs = resp.frequencies_hz
        assert all(freqs[i] <= freqs[i + 1] for i in range(len(freqs) - 1))

    def test_response_passband_near_zero_db(self):
        """DC frequency should be near 0 dB for a lowpass filter."""
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=1000.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        # First bin is DC (0 Hz) → magnitude should be near 0 dB for a lowpass
        assert resp.magnitude_db[0] > -3.0

    def test_response_above_cutoff_attenuated(self):
        """Frequencies well above cutoff should be strongly attenuated (< -20 dB)."""
        cutoff = 1000.0
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=cutoff)
        design = design_filter(spec)
        resp = compute_frequency_response(design, n_points=1024)
        # Find first bin at ~5x the cutoff
        freqs = np.array(resp.frequencies_hz)
        target = 5.0 * cutoff
        idx = int(np.argmin(np.abs(freqs - target)))
        assert resp.magnitude_db[idx] < -20.0

    def test_response_contains_cutoff_info(self):
        spec = FilterSpec("lowpass", "butterworth", 4, FS, cutoff_hz=1000.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert resp.cutoff_hz == 1000.0
        assert resp.filter_type == "lowpass"
        assert resp.order == 4

    def test_response_no_nan(self):
        spec = FilterSpec("bandpass", "butterworth", 4, FS, low_hz=500.0, high_hz=3000.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert all(np.isfinite(v) for v in resp.magnitude_db)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestFilterAPI:
    """Integration tests for the /api/audio/filter endpoint."""

    @staticmethod
    def _make_wav_upload(freq_hz: float = 1000.0, duration: float = 1.0) -> bytes:
        samples = make_mono_sine(freq_hz, duration)
        return make_wav_bytes(samples)

    def test_lowpass_endpoint(self):
        wav_bytes = self._make_wav_upload()
        resp = client.post(
            "/api/audio/filter",
            data={"filter_type": "lowpass", "family": "butterworth", "order": "4", "cutoff_hz": "3000"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    def test_highpass_endpoint(self):
        wav_bytes = self._make_wav_upload()
        resp = client.post(
            "/api/audio/filter",
            data={"filter_type": "highpass", "family": "butterworth", "order": "4", "cutoff_hz": "200"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_bandpass_endpoint(self):
        wav_bytes = self._make_wav_upload()
        resp = client.post(
            "/api/audio/filter",
            data={"filter_type": "bandpass", "family": "butterworth", "order": "4", "low_hz": "500", "high_hz": "3000"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_bandstop_endpoint(self):
        wav_bytes = self._make_wav_upload()
        resp = client.post(
            "/api/audio/filter",
            data={"filter_type": "bandstop", "family": "butterworth", "order": "4", "low_hz": "500", "high_hz": "3000"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_invalid_cutoff_rejected(self):
        wav_bytes = self._make_wav_upload()
        resp = client.post(
            "/api/audio/filter",
            data={"filter_type": "lowpass", "family": "butterworth", "order": "4", "cutoff_hz": "99999"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 422

    def test_no_file_rejected(self):
        resp = client.post(
            "/api/audio/filter",
            data={"filter_type": "lowpass", "family": "butterworth", "order": "4", "cutoff_hz": "1000"},
        )
        assert resp.status_code == 400

    def test_output_is_decodable_wav(self):
        """The returned stream must be a valid WAV file."""
        wav_bytes = self._make_wav_upload()
        resp = client.post(
            "/api/audio/filter",
            data={"filter_type": "lowpass", "family": "butterworth", "order": "4", "cutoff_hz": "3000"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        # Validate WAV header magic bytes
        content = resp.content
        assert content[:4] == b"RIFF"
        assert content[8:12] == b"WAVE"

    def test_filter_response_endpoint(self):
        resp = client.post(
            "/api/audio/filter/response",
            json={
                "filter_type": "lowpass",
                "family": "butterworth",
                "order": 4,
                "sample_rate_hz": 44100,
                "cutoff_hz": 1000.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "frequencies_hz" in body
        assert "magnitude_db" in body
        assert len(body["frequencies_hz"]) == len(body["magnitude_db"])
        assert body["filter_type"] == "lowpass"
        assert body["order"] == 4


# ---------------------------------------------------------------------------
# New filter families — DSP behaviour tests
# ---------------------------------------------------------------------------

class TestChebyshev1:
    """Chebyshev Type I: equiripple passband, monotone stopband."""

    def test_lowpass_attenuates_high(self):
        spec = FilterSpec("lowpass", "chebyshev1", 4, FS, cutoff_hz=2000.0, ripple_db=1.0)
        design = design_filter(spec)
        low = make_mono_sine(200.0)
        high = make_mono_sine(10000.0)
        out_low = apply_filter(design, low)
        out_high = apply_filter(design, high)
        assert sine_power(out_low) / sine_power(low) > 0.3
        assert sine_power(out_high) / (sine_power(high) + 1e-12) < 0.05

    def test_sos_structure(self):
        spec = FilterSpec("lowpass", "chebyshev1", 4, FS, cutoff_hz=2000.0, ripple_db=1.0)
        result = design_filter(spec)
        assert result.sos is not None
        assert result.b_fir is None
        assert not result.is_fir
        assert np.all(np.isfinite(result.sos))

    def test_ripple_db_zero_rejected(self):
        spec = FilterSpec("lowpass", "chebyshev1", 4, FS, cutoff_hz=2000.0, ripple_db=0.0)
        with pytest.raises(FilterDesignError, match="ripple_db"):
            design_filter(spec)

    def test_output_length_and_finite(self):
        spec = FilterSpec("lowpass", "chebyshev1", 4, FS, cutoff_hz=2000.0, ripple_db=1.0)
        design = design_filter(spec)
        sig = make_mono_sine(500.0)
        out = apply_filter(design, sig)
        assert len(out) == len(sig)
        assert np.all(np.isfinite(out))

    def test_frequency_response(self):
        spec = FilterSpec("lowpass", "chebyshev1", 4, FS, cutoff_hz=2000.0, ripple_db=1.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert resp.family == "chebyshev1"
        assert len(resp.frequencies_hz) > 0
        assert all(np.isfinite(v) for v in resp.magnitude_db)


class TestChebyshev2:
    """Chebyshev Type II: flat passband, equiripple stopband."""

    def test_lowpass_attenuates_high(self):
        spec = FilterSpec("lowpass", "chebyshev2", 4, FS, cutoff_hz=2000.0, attenuation_db=40.0)
        design = design_filter(spec)
        low = make_mono_sine(200.0)
        high = make_mono_sine(10000.0)
        out_low = apply_filter(design, low)
        out_high = apply_filter(design, high)
        assert sine_power(out_low) / sine_power(low) > 0.3
        assert sine_power(out_high) / (sine_power(high) + 1e-12) < 0.1

    def test_sos_structure(self):
        spec = FilterSpec("lowpass", "chebyshev2", 4, FS, cutoff_hz=2000.0, attenuation_db=40.0)
        result = design_filter(spec)
        assert result.sos is not None
        assert not result.is_fir
        assert np.all(np.isfinite(result.sos))

    def test_attenuation_db_zero_rejected(self):
        spec = FilterSpec("lowpass", "chebyshev2", 4, FS, cutoff_hz=2000.0, attenuation_db=0.0)
        with pytest.raises(FilterDesignError, match="attenuation_db"):
            design_filter(spec)

    def test_highpass(self):
        spec = FilterSpec("highpass", "chebyshev2", 4, FS, cutoff_hz=2000.0, attenuation_db=40.0)
        design = design_filter(spec)
        low = make_mono_sine(200.0)
        high = make_mono_sine(10000.0)
        assert sine_power(apply_filter(design, low)) / sine_power(low) < 0.2
        assert sine_power(apply_filter(design, high)) / sine_power(high) > 0.3

    def test_frequency_response(self):
        spec = FilterSpec("lowpass", "chebyshev2", 4, FS, cutoff_hz=2000.0, attenuation_db=40.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert resp.family == "chebyshev2"
        assert all(np.isfinite(v) for v in resp.magnitude_db)


class TestElliptic:
    """Elliptic: equiripple passband and stopband — sharpest classical IIR."""

    def test_lowpass_sharp_cutoff(self):
        """Elliptic should achieve strong attenuation very close to cutoff."""
        spec = FilterSpec("lowpass", "elliptic", 4, FS, cutoff_hz=2000.0,
                          ripple_db=1.0, attenuation_db=60.0)
        design = design_filter(spec)
        low = make_mono_sine(500.0)
        high = make_mono_sine(8000.0)
        out_low = apply_filter(design, low)
        out_high = apply_filter(design, high)
        assert sine_power(out_low) / sine_power(low) > 0.3
        assert sine_power(out_high) / (sine_power(high) + 1e-12) < 0.02

    def test_sos_structure(self):
        spec = FilterSpec("lowpass", "elliptic", 4, FS, cutoff_hz=2000.0,
                          ripple_db=1.0, attenuation_db=60.0)
        result = design_filter(spec)
        assert result.sos is not None
        assert not result.is_fir
        assert np.all(np.isfinite(result.sos))

    def test_frequency_response(self):
        spec = FilterSpec("lowpass", "elliptic", 4, FS, cutoff_hz=2000.0,
                          ripple_db=1.0, attenuation_db=60.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert resp.family == "elliptic"
        assert all(np.isfinite(v) for v in resp.magnitude_db)

    def test_output_finite(self):
        spec = FilterSpec("bandpass", "elliptic", 4, FS, low_hz=500.0, high_hz=3000.0,
                          ripple_db=1.0, attenuation_db=40.0)
        design = design_filter(spec)
        sig = make_mono_sine(1500.0)
        out = apply_filter(design, sig)
        assert np.all(np.isfinite(out))


class TestBessel:
    """Bessel: maximally flat group delay, poorer magnitude selectivity."""

    def test_lowpass_creates_sos(self):
        spec = FilterSpec("lowpass", "bessel", 4, FS, cutoff_hz=2000.0)
        result = design_filter(spec)
        assert result.sos is not None
        assert not result.is_fir
        assert np.all(np.isfinite(result.sos))

    def test_lowpass_passes_low(self):
        spec = FilterSpec("lowpass", "bessel", 4, FS, cutoff_hz=2000.0)
        design = design_filter(spec)
        low = make_mono_sine(200.0)
        out = apply_filter(design, low)
        assert sine_power(out) / sine_power(low) > 0.3

    def test_highpass_basic(self):
        spec = FilterSpec("highpass", "bessel", 4, FS, cutoff_hz=2000.0)
        design = design_filter(spec)
        high = make_mono_sine(10000.0)
        out = apply_filter(design, high)
        assert sine_power(out) / sine_power(high) > 0.2

    def test_output_length_and_finite(self):
        spec = FilterSpec("lowpass", "bessel", 4, FS, cutoff_hz=2000.0)
        design = design_filter(spec)
        sig = make_mono_sine(500.0)
        out = apply_filter(design, sig)
        assert len(out) == len(sig)
        assert np.all(np.isfinite(out))

    def test_frequency_response(self):
        spec = FilterSpec("lowpass", "bessel", 4, FS, cutoff_hz=2000.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert resp.family == "bessel"
        assert all(np.isfinite(v) for v in resp.magnitude_db)


class TestFIRWindow:
    """FIR windowed filter tests — always stable, linear phase."""

    def test_lowpass_creates_fir(self):
        spec = FilterSpec("lowpass", "fir_window", 64, FS, cutoff_hz=2000.0,
                          fir_window="hamming")
        result = design_filter(spec)
        assert result.is_fir
        assert result.b_fir is not None
        assert result.sos is None

    def test_lowpass_attenuates_high(self):
        spec = FilterSpec("lowpass", "fir_window", 64, FS, cutoff_hz=2000.0)
        design = design_filter(spec)
        low = make_mono_sine(200.0)
        high = make_mono_sine(10000.0)
        out_low = apply_filter(design, low)
        out_high = apply_filter(design, high)
        assert sine_power(out_low) / sine_power(low) > 0.3
        assert sine_power(out_high) / (sine_power(high) + 1e-12) < 0.05

    def test_highpass(self):
        spec = FilterSpec("highpass", "fir_window", 64, FS, cutoff_hz=2000.0)
        design = design_filter(spec)
        low = make_mono_sine(200.0)
        high = make_mono_sine(10000.0)
        assert sine_power(apply_filter(design, low)) / sine_power(low) < 0.1
        assert sine_power(apply_filter(design, high)) / sine_power(high) > 0.5

    def test_bandpass(self):
        spec = FilterSpec("bandpass", "fir_window", 64, FS, low_hz=500.0, high_hz=3000.0)
        design = design_filter(spec)
        mid = make_mono_sine(1500.0)
        out = apply_filter(design, mid)
        assert sine_power(out) / sine_power(mid) > 0.2

    def test_bandstop(self):
        spec = FilterSpec("bandstop", "fir_window", 64, FS, low_hz=500.0, high_hz=3000.0)
        design = design_filter(spec)
        mid = make_mono_sine(1500.0)
        below = make_mono_sine(100.0)
        out_mid = apply_filter(design, mid)
        out_below = apply_filter(design, below)
        assert sine_power(out_mid) / (sine_power(mid) + 1e-12) < 0.2
        assert sine_power(out_below) / sine_power(below) > 0.5

    @pytest.mark.parametrize("window", ["hamming", "hann", "blackman", "bartlett"])
    def test_all_windows_work(self, window: str):
        spec = FilterSpec("lowpass", "fir_window", 32, FS, cutoff_hz=2000.0,
                          fir_window=window)
        design = design_filter(spec)
        sig = make_mono_sine(500.0)
        out = apply_filter(design, sig)
        assert np.all(np.isfinite(out))

    def test_invalid_window_rejected(self):
        spec = FilterSpec("lowpass", "fir_window", 32, FS, cutoff_hz=2000.0,
                          fir_window="totally_fake_window")
        with pytest.raises(FilterDesignError, match="window"):
            design_filter(spec)

    def test_output_length_and_finite(self):
        spec = FilterSpec("lowpass", "fir_window", 64, FS, cutoff_hz=2000.0)
        design = design_filter(spec)
        sig = make_mono_sine(500.0)
        out = apply_filter(design, sig)
        assert len(out) == len(sig)
        assert np.all(np.isfinite(out))

    def test_frequency_response(self):
        spec = FilterSpec("lowpass", "fir_window", 64, FS, cutoff_hz=2000.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert resp.family == "fir_window"
        assert all(np.isfinite(v) for v in resp.magnitude_db)


class TestFIRRemez:
    """Parks-McClellan (remez) equiripple FIR tests."""

    def test_lowpass_creates_fir(self):
        spec = FilterSpec("lowpass", "fir_remez", 64, FS, cutoff_hz=2000.0,
                          transition_bandwidth_hz=400.0)
        result = design_filter(spec)
        assert result.is_fir
        assert result.b_fir is not None

    def test_lowpass_attenuates_high(self):
        spec = FilterSpec("lowpass", "fir_remez", 64, FS, cutoff_hz=2000.0,
                          transition_bandwidth_hz=400.0)
        design = design_filter(spec)
        low = make_mono_sine(200.0)
        high = make_mono_sine(10000.0)
        assert sine_power(apply_filter(design, high)) / (sine_power(high) + 1e-12) < 0.1
        assert sine_power(apply_filter(design, low)) / sine_power(low) > 0.3

    def test_highpass(self):
        spec = FilterSpec("highpass", "fir_remez", 64, FS, cutoff_hz=2000.0,
                          transition_bandwidth_hz=400.0)
        design = design_filter(spec)
        low = make_mono_sine(200.0)
        high = make_mono_sine(10000.0)
        assert sine_power(apply_filter(design, low)) / sine_power(low) < 0.1
        assert sine_power(apply_filter(design, high)) / sine_power(high) > 0.3

    def test_invalid_transition_bandwidth(self):
        """Too-wide transition band that clips into DC should be rejected."""
        spec = FilterSpec("lowpass", "fir_remez", 64, FS, cutoff_hz=100.0,
                          transition_bandwidth_hz=300.0)  # would go below 0 Hz
        with pytest.raises(FilterDesignError):
            design_filter(spec)

    def test_frequency_response(self):
        spec = FilterSpec("lowpass", "fir_remez", 64, FS, cutoff_hz=2000.0,
                          transition_bandwidth_hz=400.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert resp.family == "fir_remez"
        assert all(np.isfinite(v) for v in resp.magnitude_db)

    def test_output_finite(self):
        spec = FilterSpec("lowpass", "fir_remez", 64, FS, cutoff_hz=2000.0,
                          transition_bandwidth_hz=400.0)
        design = design_filter(spec)
        sig = make_mono_sine(500.0)
        out = apply_filter(design, sig)
        assert np.all(np.isfinite(out))
        assert len(out) == len(sig)


class TestPeakingEQ:
    """Parametric EQ biquad — boost/cut at centre frequency."""

    def test_creates_single_sos(self):
        spec = FilterSpec("peaking", "butterworth", 1, FS, center_hz=1000.0,
                          gain_db=6.0, q_factor=1.0)
        result = design_filter(spec)
        assert result.sos is not None
        assert result.n_sections == 1
        assert result.sos.shape == (1, 6)

    def test_boost_increases_power_at_centre(self):
        """A +12 dB boost at 1 kHz should increase the power of a 1 kHz sine."""
        spec = FilterSpec("peaking", "butterworth", 1, FS, center_hz=1000.0,
                          gain_db=12.0, q_factor=2.0)
        design = design_filter(spec)
        sig = make_mono_sine(1000.0)
        out = apply_filter(design, sig)
        assert sine_power(out) > sine_power(sig) * 1.5

    def test_cut_reduces_power_at_centre(self):
        """A -12 dB cut at 1 kHz should reduce the power of a 1 kHz sine."""
        spec = FilterSpec("peaking", "butterworth", 1, FS, center_hz=1000.0,
                          gain_db=-12.0, q_factor=2.0)
        design = design_filter(spec)
        sig = make_mono_sine(1000.0)
        out = apply_filter(design, sig)
        assert sine_power(out) < sine_power(sig) * 0.7

    def test_zero_gain_unity_passthrough(self):
        """0 dB gain should leave the signal essentially unchanged."""
        spec = FilterSpec("peaking", "butterworth", 1, FS, center_hz=1000.0,
                          gain_db=0.0, q_factor=1.0)
        design = design_filter(spec)
        sig = make_mono_sine(1000.0)
        out = apply_filter(design, sig)
        ratio = sine_power(out) / sine_power(sig)
        assert 0.95 < ratio < 1.05

    def test_center_hz_required(self):
        spec = FilterSpec("peaking", "butterworth", 1, FS, gain_db=6.0)
        with pytest.raises(FilterDesignError, match="center_hz"):
            design_filter(spec)

    def test_output_finite(self):
        spec = FilterSpec("peaking", "butterworth", 1, FS, center_hz=2000.0,
                          gain_db=6.0, q_factor=1.5)
        design = design_filter(spec)
        sig = make_mono_sine(2000.0)
        out = apply_filter(design, sig)
        assert np.all(np.isfinite(out))

    def test_frequency_response(self):
        spec = FilterSpec("peaking", "butterworth", 1, FS, center_hz=1000.0,
                          gain_db=6.0, q_factor=1.0)
        design = design_filter(spec)
        resp = compute_frequency_response(design)
        assert resp.center_hz == 1000.0
        assert all(np.isfinite(v) for v in resp.magnitude_db)
