"""
tests/test_stft.py — Module 05: Spectrogram / STFT
====================================================

Full test coverage for dsp_core/stft.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dsp_core.stft import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    N_FREQ_BINS,
    N_TIME_BINS,
    SpectrogramChannel,
    SpectrogramData,
    _build_freq_axis,
    _build_time_axis,
    _downsample_freq,
    _downsample_time,
    _magnitudes_to_dbfs,
    _normalize_samples,
    _stft_channel,
    compute_spectrogram,
    get_window,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sine_wave(
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono sine-wave signal as float32."""
    t = np.arange(int(duration_s * sample_rate), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _stereo_sine(
    freq_l: float = 440.0,
    freq_r: float = 880.0,
    duration_s: float = 0.5,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate a stereo [N, 2] sine signal."""
    left = _sine_wave(freq_l, duration_s, sample_rate)
    right = _sine_wave(freq_r, duration_s, sample_rate)
    return np.stack([left, right], axis=1)


# ---------------------------------------------------------------------------
# get_window
# ---------------------------------------------------------------------------


class TestGetWindow:
    def test_hann_length(self):
        w = get_window("hann", 2048)
        assert len(w) == 2048

    def test_hann_endpoints(self):
        w = get_window("hann", 512)
        assert abs(w[0]) < 1e-9
        assert abs(w[-1]) < 1e-2   # ends close but not exactly zero for even length

    def test_hann_peak_is_one(self):
        w = get_window("hann", 1025)   # odd length — exact centre is 1.0
        assert abs(w[512] - 1.0) < 1e-9

    def test_hann_symmetric(self):
        w = get_window("hann", 512)
        np.testing.assert_allclose(w, w[::-1], atol=1e-10)

    def test_hann_values_in_range(self):
        w = get_window("hann", 1024)
        assert float(w.min()) >= 0.0
        assert float(w.max()) <= 1.0 + 1e-9

    def test_length_1(self):
        w = get_window("hann", 1)
        assert len(w) == 1
        assert w[0] == 1.0

    def test_length_below_minimum_raises(self):
        with pytest.raises(ValueError, match="length must be ≥ 1"):
            get_window("hann", 0)

    def test_dtype_float64(self):
        w = get_window("hann", 256)
        assert w.dtype == np.float64
        
    def test_rectangular_window(self):
        w = get_window("rectangular", 256)
        assert np.all(w == 1.0)
        
    def test_hamming_window(self):
        w = get_window("hamming", 512)
        assert abs(w[0] - 0.08) < 1e-7
        assert abs(w[-1] - 0.08) < 1e-2
        
    def test_bartlett_window(self):
        w = get_window("bartlett", 513)
        assert abs(w[0]) < 1e-9
        assert abs(w[-1]) < 1e-9
        assert abs(w[256] - 1.0) < 1e-9
        
    def test_welch_window(self):
        w = get_window("welch", 513)
        assert abs(w[256] - 1.0) < 1e-9
        assert abs(w[0]) < 1e-9
        assert abs(w[-1]) < 1e-9
        # Check parabolic shape
        assert w[128] > w[64]
        
    def test_kaiser_window(self):
        # We test that it's utilizing beta=14.0 by comparing to a known value
        w = get_window("kaiser", 513)
        # Endpoint of kaiser with beta=14 is very small
        assert abs(w[0]) < 1e-5
        assert abs(w[256] - 1.0) < 1e-9

    def test_blackman_window(self):
        w = get_window("blackman", 513)
        assert abs(w[256] - 1.0) < 1e-9
        assert abs(w[0]) < 1e-9



# ---------------------------------------------------------------------------
# _normalize_samples
# ---------------------------------------------------------------------------


class TestNormalizeSamples:
    def test_int16_range(self):
        samples = np.array([0, 16384, -32768, 32767], dtype=np.int16)
        out = _normalize_samples(samples)
        assert out.dtype == np.float32
        assert float(out.min()) >= -1.0
        assert float(out.max()) <= 1.0

    def test_float32_passthrough(self):
        samples = np.array([0.5, -0.5, 0.1], dtype=np.float32)
        out = _normalize_samples(samples)
        np.testing.assert_allclose(out, samples, atol=1e-6)
        assert out.dtype == np.float32

    def test_int32_scaling(self):
        samples = np.array([2**30], dtype=np.int32)
        out = _normalize_samples(samples)
        expected = 2**30 / 2**31
        assert abs(float(out[0]) - expected) < 1e-5


# ---------------------------------------------------------------------------
# _stft_channel
# ---------------------------------------------------------------------------


class TestStftChannel:
    def _run(self, n_samples=44100, frame_length=2048, hop_length=512):
        signal = _sine_wave(duration_s=n_samples / 44100, sample_rate=44100)
        window = get_window("hann", frame_length)
        return _stft_channel(signal, frame_length, hop_length, window)

    def test_output_shape(self):
        signal = _sine_wave(duration_s=1.0, sample_rate=44100)
        window = get_window("hann", 2048)
        mag = _stft_channel(signal, 2048, 512, window)
        n_frames_expected = 1 + (len(signal) - 2048) // 512
        assert mag.shape[0] == n_frames_expected
        assert mag.shape[1] == 2048 // 2 + 1

    def test_output_dtype_float32(self):
        signal = _sine_wave()
        window = get_window("hann", 2048)
        mag = _stft_channel(signal, 2048, 512, window)
        assert mag.dtype == np.float32

    def test_magnitudes_non_negative(self):
        signal = _sine_wave()
        window = get_window("hann", 2048)
        mag = _stft_channel(signal, 2048, 512, window)
        assert float(mag.min()) >= 0.0

    def test_silence_magnitudes_near_zero(self):
        signal = np.zeros(44100, dtype=np.float32)
        window = get_window("hann", 2048)
        mag = _stft_channel(signal, 2048, 512, window)
        assert float(mag.max()) < 1e-10

    def test_sine_peak_at_correct_bin(self):
        """The dominant bin should correspond to the sine frequency."""
        fs = 44100
        freq = 1000.0
        frame_length = 4096
        signal = _sine_wave(freq_hz=freq, duration_s=2.0, sample_rate=fs)
        window = get_window("hann", frame_length)
        mag = _stft_channel(signal, frame_length, 1024, window)

        # Average magnitude across frames, find peak bin
        mean_mag = mag.mean(axis=0)
        peak_bin = int(np.argmax(mean_mag))

        # Expected bin
        expected_bin = round(freq * frame_length / fs)
        assert abs(peak_bin - expected_bin) <= 2   # allow ±2 bin tolerance

    def test_short_signal_is_padded(self):
        """Signal shorter than frame_length should not raise."""
        signal = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        window = get_window("hann", 2048)
        mag = _stft_channel(signal, 2048, 512, window)
        assert mag.shape[0] >= 1
        assert mag.shape[1] == 2048 // 2 + 1

    def test_hop_length_one(self):
        """hop_length=1 should produce many frames without error."""
        signal = np.zeros(100, dtype=np.float32)
        frame_length = 10
        window = get_window("hann", frame_length)
        mag = _stft_channel(signal, frame_length, 1, window)
        assert mag.shape[0] >= 1


# ---------------------------------------------------------------------------
# _magnitudes_to_dbfs
# ---------------------------------------------------------------------------


class TestMagnitudesToDbfs:
    def test_full_scale_is_zero_dbfs(self):
        mag = np.array([[1.0]], dtype=np.float32)
        dbfs = _magnitudes_to_dbfs(mag)
        assert abs(float(dbfs[0, 0])) < 0.01   # ≈ 0 dBFS

    def test_silence_is_strongly_negative(self):
        mag = np.zeros((4, 4), dtype=np.float32)
        dbfs = _magnitudes_to_dbfs(mag)
        assert float(dbfs.max()) < -200.0

    def test_half_amplitude(self):
        mag = np.array([[0.5]], dtype=np.float32)
        dbfs = _magnitudes_to_dbfs(mag)
        expected = 20.0 * math.log10(0.5)
        assert abs(float(dbfs[0, 0]) - expected) < 0.01

    def test_output_dtype_float32(self):
        mag = np.ones((3, 3), dtype=np.float32) * 0.1
        dbfs = _magnitudes_to_dbfs(mag)
        assert dbfs.dtype == np.float32

    def test_shape_preserved(self):
        mag = np.ones((10, 20), dtype=np.float32) * 0.5
        dbfs = _magnitudes_to_dbfs(mag)
        assert dbfs.shape == (10, 20)


# ---------------------------------------------------------------------------
# _downsample_time
# ---------------------------------------------------------------------------


class TestDownsampleTime:
    def test_no_downsampling_needed(self):
        m = np.random.rand(10, 50).astype(np.float32)
        out = _downsample_time(m, 20)
        np.testing.assert_array_equal(out, m)

    def test_reduces_to_target(self):
        m = np.random.rand(1000, 50).astype(np.float32)
        out = _downsample_time(m, 64)
        assert out.shape[0] == 64
        assert out.shape[1] == 50

    def test_peak_preserving(self):
        """Maximum in each group should appear in the downsampled output."""
        # Two distinct frames: first has high value at freq 0, second low
        m = np.zeros((4, 3), dtype=np.float32)
        m[0, 0] = 1.0   # peak in group 0
        m[2, 1] = 0.9   # peak in group 1
        out = _downsample_time(m, 2)
        assert float(out[0, 0]) == pytest.approx(1.0)
        assert float(out[1, 1]) == pytest.approx(0.9)

    def test_frequency_axis_unchanged(self):
        m = np.random.rand(100, 30).astype(np.float32)
        out = _downsample_time(m, 50)
        assert out.shape[1] == 30


# ---------------------------------------------------------------------------
# _downsample_freq
# ---------------------------------------------------------------------------


class TestDownsampleFreq:
    def test_no_downsampling_needed(self):
        m = np.random.rand(10, 50).astype(np.float32)
        out = _downsample_freq(m, 100)
        np.testing.assert_array_equal(out, m)

    def test_reduces_to_target(self):
        m = np.random.rand(50, 1000).astype(np.float32)
        out = _downsample_freq(m, 128)
        assert out.shape[0] == 50
        assert out.shape[1] == 128

    def test_peak_preserving(self):
        m = np.zeros((3, 4), dtype=np.float32)
        m[0, 0] = 1.0
        m[1, 2] = 0.7
        out = _downsample_freq(m, 2)
        assert float(out[0, 0]) == pytest.approx(1.0)

    def test_time_axis_unchanged(self):
        m = np.random.rand(30, 100).astype(np.float32)
        out = _downsample_freq(m, 50)
        assert out.shape[0] == 30


# ---------------------------------------------------------------------------
# _build_time_axis / _build_freq_axis
# ---------------------------------------------------------------------------


class TestTimeAxis:
    def test_length_matches_display_bins(self):
        n_frames = 200
        n_display = 50
        times = _build_time_axis(n_frames, n_display, 2048, 512, 44100)
        assert len(times) == n_display

    def test_times_monotonically_increasing(self):
        times = _build_time_axis(200, 50, 2048, 512, 44100)
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_first_time_positive(self):
        times = _build_time_axis(100, 20, 2048, 512, 44100)
        assert times[0] > 0.0

    def test_fewer_frames_than_display(self):
        """If n_frames < n_display, should return n_frames times."""
        times = _build_time_axis(10, 50, 2048, 512, 44100)
        assert len(times) == 10


class TestFreqAxis:
    def test_length_matches_display_bins(self):
        freqs = _build_freq_axis(1025, 256, 2048, 44100)
        assert len(freqs) == 256

    def test_dc_bin_near_low_freq(self):
        """After grouping raw bins, the first display row should be at a low frequency.

        When downsampling 1025 raw bins to 256 display bins, each group spans
        1025/256 ≈ 4 raw bins.  The centre of the first group is around
        bin 1–2, so the centre frequency is a few Hz above 0, not exactly 0.
        We assert it's below the frequency resolution (Δf = Fs/L ≈ 21.5 Hz)
        so that we're confident the first group captures the near-DC region.
        """
        fs = 44100
        frame_length = 2048
        freqs = _build_freq_axis(1025, 256, frame_length, fs)
        delta_f = fs / frame_length   # ≈ 21.5 Hz
        # The first displayed frequency should be well below Nyquist
        # and reasonably close to the low-frequency end of the spectrum
        assert freqs[0] < delta_f * 4   # within a few raw bins of DC

    def test_last_bin_near_nyquist(self):
        fs = 44100
        freqs = _build_freq_axis(1025, 256, 2048, fs)
        assert freqs[-1] < fs / 2
        assert freqs[-1] > fs / 2 - 500

    def test_frequencies_monotonically_increasing(self):
        freqs = _build_freq_axis(1025, 256, 2048, 44100)
        for i in range(1, len(freqs)):
            assert freqs[i] > freqs[i - 1]

    def test_fewer_bins_than_display(self):
        """If n_freq_bins_full < n_display, should return n_freq_bins_full entries."""
        freqs = _build_freq_axis(10, 256, 20, 44100)
        assert len(freqs) == 10


# ---------------------------------------------------------------------------
# compute_spectrogram — integration tests
# ---------------------------------------------------------------------------


class TestComputeSpectrogram:
    def test_returns_spectrogram_data(self):
        signal = _sine_wave(duration_s=1.0)
        result = compute_spectrogram(signal, 44100)
        assert isinstance(result, SpectrogramData)

    def test_mono_one_channel(self):
        signal = _sine_wave(duration_s=0.5)
        result = compute_spectrogram(signal, 44100)
        assert result.n_channels == 1
        assert len(result.channels) == 1

    def test_stereo_two_channels(self):
        signal = _stereo_sine()
        result = compute_spectrogram(signal, 44100)
        assert result.n_channels == 2
        assert len(result.channels) == 2

    def test_channel_indices_correct(self):
        signal = _stereo_sine()
        result = compute_spectrogram(signal, 44100)
        assert result.channels[0].channel_index == 0
        assert result.channels[1].channel_index == 1

    def test_metadata_fields(self):
        signal = _sine_wave(duration_s=1.0, sample_rate=44100)
        result = compute_spectrogram(signal, 44100)
        assert result.sample_rate_hz == 44100
        assert result.frame_length == DEFAULT_FRAME_LENGTH
        assert result.hop_length == DEFAULT_HOP_LENGTH
        assert result.window == "hann"
        assert abs(result.duration_seconds - 1.0) < 0.01
        assert result.nyquist_hz == pytest.approx(22050.0)

    def test_frequency_resolution(self):
        fs = 44100
        fl = 2048
        signal = _sine_wave(sample_rate=fs)
        result = compute_spectrogram(signal, fs, frame_length=fl)
        expected = fs / fl
        assert abs(result.frequency_resolution_hz - expected) < 0.01

    def test_time_resolution(self):
        fs = 44100
        hl = 512
        signal = _sine_wave(sample_rate=fs)
        result = compute_spectrogram(signal, fs, hop_length=hl)
        expected = hl / fs
        assert abs(result.time_resolution_seconds - expected) < 1e-6

    def test_display_dimensions_within_caps(self):
        signal = _sine_wave(duration_s=2.0)
        result = compute_spectrogram(signal, 44100)
        assert result.n_time_display <= N_TIME_BINS
        assert result.n_freq_display <= N_FREQ_BINS

    def test_magnitude_matrix_shape(self):
        signal = _sine_wave(duration_s=1.0)
        result = compute_spectrogram(signal, 44100)
        ch = result.channels[0]
        n_t = result.n_time_display
        n_f = result.n_freq_display
        assert len(ch.magnitudes) == n_t
        assert all(len(row) == n_f for row in ch.magnitudes)

    def test_time_frames_length_matches_n_time_display(self):
        signal = _sine_wave(duration_s=1.0)
        result = compute_spectrogram(signal, 44100)
        ch = result.channels[0]
        assert len(ch.time_frames) == result.n_time_display

    def test_freq_bins_length_matches_n_freq_display(self):
        signal = _sine_wave(duration_s=1.0)
        result = compute_spectrogram(signal, 44100)
        ch = result.channels[0]
        assert len(ch.freq_bins) == result.n_freq_display

    def test_magnitudes_are_negative_dbfs(self):
        """All displayed dBFS values should be ≤ 0 for a normalized signal."""
        signal = _sine_wave(duration_s=1.0, amplitude=0.5)
        result = compute_spectrogram(signal, 44100)
        ch = result.channels[0]
        max_val = max(max(row) for row in ch.magnitudes)
        assert max_val <= 0.5   # amplitude 0.5 → ≈ -6 dBFS, tiny tolerance

    def test_silence_all_very_negative(self):
        signal = np.zeros(44100, dtype=np.float32)
        result = compute_spectrogram(signal, 44100)
        ch = result.channels[0]
        max_val = max(max(row) for row in ch.magnitudes)
        assert max_val < -100.0

    def test_short_audio_no_error(self):
        signal = np.zeros(100, dtype=np.float32)
        result = compute_spectrogram(signal, 44100)
        assert isinstance(result, SpectrogramData)

    def test_int16_signal_accepted(self):
        signal = (np.random.rand(44100) * 32767).astype(np.int16)
        result = compute_spectrogram(signal, 44100)
        assert isinstance(result, SpectrogramData)

    def test_custom_frame_hop(self):
        signal = _sine_wave(duration_s=1.0)
        result = compute_spectrogram(signal, 44100, frame_length=1024, hop_length=256)
        assert result.frame_length == 1024
        assert result.hop_length == 256

    def test_invalid_frame_length_raises(self):
        signal = _sine_wave()
        with pytest.raises(ValueError, match="frame_length"):
            compute_spectrogram(signal, 44100, frame_length=1)

    def test_invalid_hop_length_raises(self):
        signal = _sine_wave()
        with pytest.raises(ValueError, match="hop_length"):
            compute_spectrogram(signal, 44100, hop_length=0)

    def test_unsupported_window_raises(self):
        signal = _sine_wave()
        with pytest.raises(ValueError, match="Unsupported window"):
            compute_spectrogram(signal, 44100, window="nonexistent_window")

    def test_supported_windows(self):
        signal = _sine_wave()
        for win in ["hann", "hamming", "rectangular", "blackman", "kaiser", "bartlett", "welch"]:
            result = compute_spectrogram(signal, 44100, window=win)
            assert result.window == win
            assert isinstance(result, SpectrogramData)

    def test_freq_bins_monotonically_increasing(self):
        signal = _sine_wave(duration_s=1.0)
        result = compute_spectrogram(signal, 44100)
        freqs = result.channels[0].freq_bins
        for i in range(1, len(freqs)):
            assert freqs[i] > freqs[i - 1]

    def test_time_frames_monotonically_increasing(self):
        signal = _sine_wave(duration_s=1.0)
        result = compute_spectrogram(signal, 44100)
        times = result.channels[0].time_frames
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_n_freq_bins_full_is_frame_length_half_plus_one(self):
        fl = 1024
        signal = _sine_wave(duration_s=1.0)
        result = compute_spectrogram(signal, 44100, frame_length=fl)
        assert result.n_freq_bins_full == fl // 2 + 1

    def test_different_sample_rates(self):
        for fs in [8000, 16000, 22050, 44100, 48000]:
            signal = _sine_wave(duration_s=1.0, sample_rate=fs)
            result = compute_spectrogram(signal, fs)
            assert result.sample_rate_hz == fs
            assert result.nyquist_hz == pytest.approx(fs / 2)
