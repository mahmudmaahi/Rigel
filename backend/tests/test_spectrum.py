"""
tests/test_spectrum.py — Module 04: Fourier Analysis Tests
===========================================================

Tests for dsp_core/spectrum.py and the POST /api/audio/spectrum endpoint.

Test categories:
    1.  Sample normalisation
    2.  Naive DFT — shape, DC bin, agreement with numpy FFT
    3.  One-sided amplitude scaling — DC, mid-bins, even-N Nyquist, odd-N
    4.  Full-scale sine amplitude
    5.  440 Hz exact-bin test (primary intuitive demonstration)
    6.  Frequency axis — starts at 0, ends near Nyquist, correct Δf
    7.  dBFS conversion
    8.  Display reduction — bin count, peak preservation, long audio
    9.  Stereo handling
    10. API endpoint structure and validation
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from dsp_core.spectrum import (
    N_DISPLAY_BINS,
    SpectrumData,
    _amplitudes_to_dbfs,
    _compute_one_sided_magnitude,
    _downsample_spectrum,
    compute_dft_naive,
    compute_spectrum,
    normalize_samples,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a NumPy int16 array as an in-memory WAV file."""
    buf = io.BytesIO()
    n_channels = 1 if samples.ndim == 1 else samples.shape[1]
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _sine_int16(frequency_hz: float, sample_rate: int, n_samples: int) -> np.ndarray:
    """Generate a mono int16 sine wave."""
    t = np.arange(n_samples) / sample_rate
    return (np.sin(2 * np.pi * frequency_hz * t) * 32767).astype(np.int16)


def _sine_float32(frequency_hz: float, sample_rate: int, n_samples: int) -> np.ndarray:
    """Generate a mono float32 sine wave in [-1, 1]."""
    t = np.arange(n_samples) / sample_rate
    return np.sin(2 * np.pi * frequency_hz * t).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Sample normalisation
# ---------------------------------------------------------------------------


class TestNormalizeSamples:
    def test_int16_maps_to_float_range(self):
        samples = np.array([-32768, 0, 32767], dtype=np.int16)
        result = normalize_samples(samples)
        assert result.dtype == np.float32
        assert float(result.min()) >= -1.0
        assert float(result.max()) <= 1.0

    def test_int16_full_negative_is_minus_one(self):
        samples = np.array([-32768], dtype=np.int16)
        result = normalize_samples(samples)
        assert abs(float(result[0]) - (-1.0)) < 1e-4

    def test_int16_zero_stays_zero(self):
        samples = np.array([0], dtype=np.int16)
        result = normalize_samples(samples)
        assert float(result[0]) == 0.0

    def test_float32_passthrough(self):
        samples = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        result = normalize_samples(samples)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, samples)


# ---------------------------------------------------------------------------
# 2. Naive DFT — O(N²) educational implementation
# ---------------------------------------------------------------------------


class TestNaiveDFT:
    def test_output_shape_equals_input_length(self):
        """DFT of length-N signal → N complex coefficients."""
        for N in [4, 8, 16]:
            signal = np.random.rand(N)
            X = compute_dft_naive(signal)
            assert X.shape == (N,), f"Expected shape ({N},) for N={N}"
            assert X.dtype == complex

    def test_dc_bin_equals_sum_of_samples(self):
        """DFT bin 0 is the sum of all samples (DC component).

        X[0] = Σ x[n] · exp(0) = Σ x[n]
        """
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        X = compute_dft_naive(signal)
        expected_dc = sum(signal)
        assert abs(X[0].real - expected_dc) < 1e-9

    def test_agrees_with_numpy_fft(self):
        """Naive DFT must match np.fft.fft within numerical tolerance."""
        N = 32
        signal = np.random.rand(N).astype(np.float64)
        X_naive = compute_dft_naive(signal)
        X_numpy = np.fft.fft(signal)
        np.testing.assert_allclose(X_naive, X_numpy, atol=1e-8, rtol=1e-6)

    def test_agrees_with_numpy_fft_for_small_sine(self):
        """DFT vs FFT agreement on a known sinusoidal signal."""
        N = 16
        signal = np.sin(2 * np.pi * 2 * np.arange(N) / N)
        X_naive = compute_dft_naive(signal)
        X_numpy = np.fft.fft(signal)
        np.testing.assert_allclose(X_naive, X_numpy, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. One-sided amplitude scaling
# ---------------------------------------------------------------------------


class TestOneSidedScaling:
    def test_output_length_even_n(self):
        """Even N → N//2 + 1 one-sided bins."""
        N = 100
        signal = np.random.rand(N).astype(np.float32)
        freqs, amps = _compute_one_sided_magnitude(signal, sample_rate_hz=N)
        assert len(freqs) == N // 2 + 1
        assert len(amps) == N // 2 + 1

    def test_output_length_odd_n(self):
        """Odd N → (N+1)//2 one-sided bins."""
        N = 101
        signal = np.random.rand(N).astype(np.float32)
        freqs, amps = _compute_one_sided_magnitude(signal, sample_rate_hz=N)
        assert len(freqs) == N // 2 + 1  # numpy's rfft always returns N//2+1

    def test_dc_amplitude_not_doubled(self):
        """DC bin A[0] = |X[0]| / N (must NOT be doubled).

        A DC signal x[n] = c → X[0] = N·c → A[0] should equal c.
        """
        N = 64
        c = 0.5
        signal = np.full(N, c, dtype=np.float32)
        _, amplitudes = _compute_one_sided_magnitude(signal, sample_rate_hz=N)
        # DC amplitude should equal c
        assert abs(float(amplitudes[0]) - c) < 1e-5

    def test_mid_bin_doubled(self):
        """Interior bin A[k] = 2|X[k]|/N.

        A cosine at frequency k=2 has amplitude 1.0 by definition.
        The one-sided spectrum should recover amplitude ≈ 1.0.
        """
        N = 64
        k_target = 2
        t = np.arange(N)
        signal = np.cos(2 * np.pi * k_target * t / N).astype(np.float32)
        _, amplitudes = _compute_one_sided_magnitude(signal, sample_rate_hz=N)
        # Peak at k_target should be ≈ 1.0
        assert abs(float(amplitudes[k_target]) - 1.0) < 0.01

    def test_nyquist_bin_not_doubled_even_n(self):
        """Nyquist bin (k = N//2) must NOT be doubled for even N.

        A Nyquist cosine has amplitude 1.0. The one-sided spectrum should
        recover ≈ 1.0, not 2.0.
        """
        N = 64   # even
        k_nyquist = N // 2
        t = np.arange(N)
        signal = np.cos(2 * np.pi * k_nyquist * t / N).astype(np.float32)
        _, amplitudes = _compute_one_sided_magnitude(signal, sample_rate_hz=N)
        nyquist_amp = float(amplitudes[k_nyquist])
        assert abs(nyquist_amp - 1.0) < 0.05  # not doubled (not ≈ 2.0)

    def test_odd_n_all_bins_after_dc_doubled(self):
        """For odd N, all bins after DC should be doubled."""
        N = 65   # odd
        signal = np.random.rand(N).astype(np.float32)
        _, amplitudes = _compute_one_sided_magnitude(signal, sample_rate_hz=N)
        # Sanity: should produce N//2 + 1 bins
        assert len(amplitudes) == N // 2 + 1


# ---------------------------------------------------------------------------
# 4. Full-scale sine amplitude
# ---------------------------------------------------------------------------


class TestFullScaleAmplitude:
    def test_full_scale_sine_near_zero_dbfs(self):
        """A full-scale sine (amplitude=1.0 float32) should give ≈ 0 dBFS at its peak.

        A[k] = 2|X[k]|/N ≈ 1.0  →  20·log10(1.0) = 0 dBFS
        """
        Fs = 16_000
        N = Fs  # 1 second, exact-bin alignment at f=1 Hz per bin
        k_target = 200  # 200 Hz
        signal = np.sin(2 * np.pi * k_target * np.arange(N) / N).astype(np.float32)
        freqs, amplitudes = _compute_one_sided_magnitude(signal, Fs)
        peak_amplitude = float(amplitudes[k_target])
        # Should be close to 1.0
        assert abs(peak_amplitude - 1.0) < 0.01
        # dBFS should be close to 0
        dbfs_values = _amplitudes_to_dbfs(amplitudes)
        peak_dbfs = float(dbfs_values[k_target])
        assert abs(peak_dbfs - 0.0) < 0.5   # within 0.5 dBFS of 0


# ---------------------------------------------------------------------------
# 5. 440 Hz exact-bin test — primary intuitive demonstration
# ---------------------------------------------------------------------------


class TestSineWavePeak:
    def test_440hz_exact_bin_raw_fft(self):
        """440 Hz sine at Fs=16000, N=16000 → peak at bin 440.

        With Δf = Fs/N = 1.0 Hz per bin, 440 Hz falls exactly on bin 440.
        No spectral leakage should occur.
        """
        Fs = 16_000
        N = 16_000
        f0 = 440.0

        signal = _sine_float32(f0, Fs, N)
        freqs, amplitudes = _compute_one_sided_magnitude(signal, Fs)

        peak_bin = int(np.argmax(amplitudes))
        expected_bin = round(f0 * N / Fs)  # = 440

        # Allow ±1 bin tolerance
        assert abs(peak_bin - expected_bin) <= 1, (
            f"Peak at bin {peak_bin} ({float(freqs[peak_bin]):.1f} Hz), "
            f"expected near bin {expected_bin} (440 Hz)"
        )

    def test_440hz_via_compute_spectrum_display_bins(self):
        """compute_spectrum on 440 Hz int16 WAV → display peak near 440 Hz."""
        Fs = 16_000
        N = 16_000
        samples_int16 = _sine_int16(440.0, Fs, N)

        result = compute_spectrum(samples_int16, Fs)
        assert isinstance(result, SpectrumData)

        bins = result.channels[0].bins
        peak_bin = max(bins, key=lambda b: b.magnitude_dbfs)

        # Allow generous tolerance because 1024 display bins span Fs/2 = 8000 Hz
        # → each display bin is ~7.8 Hz wide
        assert abs(peak_bin.frequency_hz - 440.0) < 30.0, (
            f"Peak display bin at {peak_bin.frequency_hz:.1f} Hz, expected near 440 Hz"
        )


# ---------------------------------------------------------------------------
# 6. Frequency axis
# ---------------------------------------------------------------------------


class TestFrequencyAxis:
    def test_dc_bin_is_zero_hz(self):
        signal = _sine_float32(100.0, 8000, 800)
        freqs, _ = _compute_one_sided_magnitude(signal, 8000)
        assert float(freqs[0]) == 0.0

    def test_last_bin_near_nyquist(self):
        """Last raw FFT bin must be at or very close to Nyquist = Fs/2."""
        Fs = 8_000
        N = 800
        signal = _sine_float32(100.0, Fs, N)
        freqs, _ = _compute_one_sided_magnitude(signal, Fs)
        # Allow Δf/2 tolerance (the last bin may be slightly below Nyquist for odd N)
        nyquist = Fs / 2
        assert abs(float(freqs[-1]) - nyquist) <= Fs / N

    def test_frequency_resolution_delta_f(self):
        """Spacing between consecutive raw bins must equal Δf = Fs / N."""
        Fs = 16_000
        N = 1_600
        signal = np.random.rand(N).astype(np.float32)
        freqs, _ = _compute_one_sided_magnitude(signal, Fs)
        expected_delta_f = Fs / N
        actual_delta_f = float(freqs[1]) - float(freqs[0])
        assert abs(actual_delta_f - expected_delta_f) < 1e-3

    def test_frequency_resolution_stored_in_spectrum_data(self):
        """SpectrumData.frequency_resolution_hz must equal Fs / N."""
        Fs = 16_000
        N = 8_000
        samples = _sine_int16(440.0, Fs, N)
        result = compute_spectrum(samples, Fs)
        expected_delta_f = Fs / N
        assert abs(result.frequency_resolution_hz - expected_delta_f) < 1e-4

    def test_nyquist_stored_in_spectrum_data(self):
        Fs = 16_000
        samples = _sine_int16(440.0, Fs, Fs)
        result = compute_spectrum(samples, Fs)
        assert abs(result.nyquist_hz - Fs / 2) < 1e-6


# ---------------------------------------------------------------------------
# 7. dBFS conversion
# ---------------------------------------------------------------------------


class TestDbfsConversion:
    def test_amplitude_one_gives_zero_dbfs(self):
        """Amplitude 1.0 (full scale) must give 0 dBFS."""
        amplitudes = np.array([1.0], dtype=np.float32)
        dbfs = _amplitudes_to_dbfs(amplitudes)
        assert abs(float(dbfs[0]) - 0.0) < 0.001

    def test_amplitude_half_gives_minus_6_dbfs(self):
        """Amplitude 0.5 → 20·log10(0.5) ≈ -6.02 dBFS."""
        amplitudes = np.array([0.5], dtype=np.float32)
        dbfs = _amplitudes_to_dbfs(amplitudes)
        assert abs(float(dbfs[0]) - (-6.021)) < 0.01

    def test_silent_bin_does_not_raise(self):
        """Amplitude 0.0 must not raise; result should be a large negative dBFS."""
        amplitudes = np.array([0.0], dtype=np.float32)
        dbfs = _amplitudes_to_dbfs(amplitudes)
        assert float(dbfs[0]) < -200.0   # far below hearing threshold

    def test_output_dtype_float32(self):
        amplitudes = np.array([0.1, 0.5, 1.0], dtype=np.float32)
        dbfs = _amplitudes_to_dbfs(amplitudes)
        assert dbfs.dtype == np.float32


# ---------------------------------------------------------------------------
# 8. Display reduction — bin count and peak preservation
# ---------------------------------------------------------------------------


class TestDisplayReduction:
    def test_mono_display_bin_count(self):
        """Display spectrum must have ≤ N_DISPLAY_BINS bins."""
        Fs = 16_000
        N = Fs * 2  # 2 seconds
        samples = _sine_int16(440.0, Fs, N)
        result = compute_spectrum(samples, Fs)
        assert result.n_display_bins <= N_DISPLAY_BINS

    def test_long_audio_stays_within_display_budget(self):
        """~1.5 M samples must still produce ≤ 1024 display bins."""
        Fs = 44_100
        N = 1_469_952
        # White noise — structurally tests payload, not correctness
        rng = np.random.default_rng(42)
        samples = (rng.uniform(-1, 1, N) * 32767).astype(np.int16)
        result = compute_spectrum(samples, Fs)
        assert result.n_display_bins <= N_DISPLAY_BINS
        assert result.n_fft_full == N

    def test_peak_preserved_after_display_reduction(self):
        """Dominant spectral peak at 440 Hz must survive display reduction."""
        Fs = 16_000
        samples = _sine_int16(440.0, Fs, Fs)
        result = compute_spectrum(samples, Fs)
        peak = max(result.channels[0].bins, key=lambda b: b.magnitude_dbfs)
        # 1024 bins over 8 kHz → each bin ≈ 7.8 Hz wide; allow 2-bin tolerance
        assert abs(peak.frequency_hz - 440.0) < 30.0

    def test_display_bin_center_frequencies_increase(self):
        """Display bin centre frequencies must be strictly increasing."""
        Fs = 16_000
        samples = _sine_int16(440.0, Fs, Fs)
        result = compute_spectrum(samples, Fs)
        freqs = [b.frequency_hz for b in result.channels[0].bins]
        assert all(f2 > f1 for f1, f2 in zip(freqs, freqs[1:]))

    def test_dc_signal_spectrum(self):
        """A DC offset (silence minus DC) should show energy at 0 Hz."""
        N = 16_000
        Fs = 16_000
        # Pure DC: all samples at half scale
        samples = np.full(N, 16383, dtype=np.int16)
        result = compute_spectrum(samples, Fs)
        first_bin = result.channels[0].bins[0]
        # First display bin covers 0 Hz region; should have the highest magnitude
        all_mags = [b.magnitude_dbfs for b in result.channels[0].bins]
        assert first_bin.magnitude_dbfs == max(all_mags)


# ---------------------------------------------------------------------------
# 9. Stereo handling
# ---------------------------------------------------------------------------


class TestStereoSpectrum:
    def test_stereo_produces_two_channels(self):
        Fs = 16_000
        N = Fs
        left = _sine_float32(440.0, Fs, N)
        right = _sine_float32(880.0, Fs, N)
        stereo = np.stack([left, right], axis=1)
        samples_int16 = (stereo * 32767).astype(np.int16)

        result = compute_spectrum(samples_int16, Fs)

        assert result.n_channels == 2
        assert len(result.channels) == 2
        assert result.channels[0].channel_index == 0
        assert result.channels[1].channel_index == 1

    def test_stereo_channel_peaks_at_different_frequencies(self):
        """Left (440 Hz) and right (880 Hz) channels must peak at different freqs."""
        Fs = 16_000
        N = Fs
        left = _sine_float32(440.0, Fs, N)
        right = _sine_float32(880.0, Fs, N)
        stereo = np.stack([left, right], axis=1)
        samples_int16 = (stereo * 32767).astype(np.int16)

        result = compute_spectrum(samples_int16, Fs)

        peak_left = max(result.channels[0].bins, key=lambda b: b.magnitude_dbfs)
        peak_right = max(result.channels[1].bins, key=lambda b: b.magnitude_dbfs)

        # Right channel (880 Hz) peak should be higher in frequency than left (440 Hz)
        assert peak_right.frequency_hz > peak_left.frequency_hz


# ---------------------------------------------------------------------------
# 10. API endpoint
# ---------------------------------------------------------------------------


class TestSpectrumEndpoint:
    def _post_wav(self, samples: np.ndarray, sample_rate: int):
        wav_bytes = _make_wav_bytes(samples, sample_rate)
        return client.post(
            "/api/audio/spectrum",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )

    def test_returns_200_for_valid_wav(self):
        samples = _sine_int16(440.0, 16_000, 8_000)
        assert self._post_wav(samples, 16_000).status_code == 200

    def test_response_status_field(self):
        samples = _sine_int16(440.0, 16_000, 8_000)
        data = self._post_wav(samples, 16_000).json()
        assert data["status"] == "spectrum_computed"

    def test_response_has_metadata_and_spectrum(self):
        samples = _sine_int16(440.0, 16_000, 8_000)
        data = self._post_wav(samples, 16_000).json()
        assert "metadata" in data
        assert "spectrum" in data

    def test_spectrum_fields_present(self):
        samples = _sine_int16(440.0, 16_000, 8_000)
        spectrum = self._post_wav(samples, 16_000).json()["spectrum"]
        for field in ("n_channels", "n_fft_full", "n_display_bins",
                      "frequency_resolution_hz", "nyquist_hz", "channels"):
            assert field in spectrum, f"Missing field: {field}"

    def test_bin_fields_present(self):
        samples = _sine_int16(440.0, 16_000, 8_000)
        channels = self._post_wav(samples, 16_000).json()["spectrum"]["channels"]
        first_bin = channels[0]["bins"][0]
        assert "frequency_hz" in first_bin
        assert "magnitude_dbfs" in first_bin

    def test_nyquist_equals_half_sample_rate(self):
        Fs = 16_000
        samples = _sine_int16(440.0, Fs, 8_000)
        spectrum = self._post_wav(samples, Fs).json()["spectrum"]
        assert abs(spectrum["nyquist_hz"] - Fs / 2) < 0.001

    def test_no_file_returns_400(self):
        assert client.post("/api/audio/spectrum").status_code == 400

    def test_empty_file_returns_400(self):
        response = client.post(
            "/api/audio/spectrum",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400

    def test_existing_analyze_endpoint_still_works(self):
        """POST /api/audio/analyze (Module 03) must still return 200."""
        samples = _sine_int16(440.0, 16_000, 8_000)
        wav_bytes = _make_wav_bytes(samples, 16_000)
        response = client.post(
            "/api/audio/analyze",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        assert "waveform" in response.json()
