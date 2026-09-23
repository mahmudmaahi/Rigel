"""
Unit tests for Module 03 — Waveform Visualization.

Tests cover:
- DSP core: time mapping, downsampling, min/max peak preservation, mono/stereo,
  short audio, long audio, edge cases.
- API layer: /api/audio/analyze response structure and correctness.
"""

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from app.main import create_app
from dsp_core.waveform import (
    DEFAULT_N_BINS,
    WaveformBin,
    compute_waveform,
)


client = TestClient(create_app())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_bytes(samples: np.ndarray, sample_rate: int = 8000) -> bytes:
    buffer = BytesIO()
    wavfile.write(buffer, sample_rate, samples)
    return buffer.getvalue()


def _make_mono(n_samples: int, sample_rate: int = 8000) -> np.ndarray:
    """Return a simple incrementing int16 mono signal."""
    return np.arange(n_samples, dtype=np.int16)


def _make_stereo(n_samples: int, sample_rate: int = 8000) -> np.ndarray:
    """Return a two-channel int16 stereo signal."""
    left = np.arange(n_samples, dtype=np.int16)
    right = np.arange(n_samples, 0, -1, dtype=np.int16)
    return np.column_stack((left, right))


# ---------------------------------------------------------------------------
# DSP core — compute_waveform
# ---------------------------------------------------------------------------


class TestComputeWaveformMono:
    def test_returns_one_channel_for_mono(self) -> None:
        samples = _make_mono(4000)
        result = compute_waveform(samples, sample_rate_hz=8000)
        assert result.n_channels == 1
        assert len(result.channels) == 1
        assert result.channels[0].channel_index == 0

    def test_bin_count_capped_at_n_bins(self) -> None:
        samples = _make_mono(16000)
        result = compute_waveform(samples, sample_rate_hz=8000, n_bins=500)
        assert result.n_bins == 500
        assert len(result.channels[0].bins) == 500

    def test_bin_count_clamped_for_short_audio(self) -> None:
        """If audio has fewer samples than n_bins, one bin per sample."""
        samples = _make_mono(10)
        result = compute_waveform(samples, sample_rate_hz=8000, n_bins=1500)
        assert result.n_bins == 10
        assert len(result.channels[0].bins) == 10

    def test_duration_seconds_correct(self) -> None:
        n = 8000
        result = compute_waveform(_make_mono(n), sample_rate_hz=8000)
        assert result.duration_seconds == pytest.approx(1.0)

    def test_sample_rate_preserved(self) -> None:
        result = compute_waveform(_make_mono(4000), sample_rate_hz=44100)
        assert result.sample_rate_hz == 44100

    def test_first_bin_time_is_zero(self) -> None:
        result = compute_waveform(_make_mono(8000), sample_rate_hz=8000)
        assert result.channels[0].bins[0].time_seconds == pytest.approx(0.0)

    def test_bin_times_are_non_decreasing(self) -> None:
        result = compute_waveform(_make_mono(16000), sample_rate_hz=8000)
        times = [b.time_seconds for b in result.channels[0].bins]
        for a, b in zip(times, times[1:]):
            assert b >= a

    def test_last_bin_time_less_than_duration(self) -> None:
        result = compute_waveform(_make_mono(8000), sample_rate_hz=8000)
        last_t = result.channels[0].bins[-1].time_seconds
        assert last_t < result.duration_seconds

    def test_min_amplitude_leq_max_amplitude(self) -> None:
        result = compute_waveform(_make_mono(8000), sample_rate_hz=8000)
        for bin_ in result.channels[0].bins:
            assert bin_.min_amplitude <= bin_.max_amplitude


class TestComputeWaveformStereo:
    def test_returns_two_channels_for_stereo(self) -> None:
        samples = _make_stereo(8000)
        result = compute_waveform(samples, sample_rate_hz=8000)
        assert result.n_channels == 2
        assert len(result.channels) == 2

    def test_channel_indices_are_correct(self) -> None:
        samples = _make_stereo(8000)
        result = compute_waveform(samples, sample_rate_hz=8000)
        assert result.channels[0].channel_index == 0
        assert result.channels[1].channel_index == 1

    def test_stereo_channels_are_independent(self) -> None:
        """Left and right channels must differ for an asymmetric stereo signal."""
        samples = _make_stereo(8000)
        result = compute_waveform(samples, sample_rate_hz=8000)
        ch0_maxes = [b.max_amplitude for b in result.channels[0].bins]
        ch1_maxes = [b.max_amplitude for b in result.channels[1].bins]
        # Channels are mirror-images, so they must differ.
        assert ch0_maxes != ch1_maxes

    def test_stereo_bin_counts_match(self) -> None:
        samples = _make_stereo(8000)
        result = compute_waveform(samples, sample_rate_hz=8000)
        assert len(result.channels[0].bins) == len(result.channels[1].bins)


class TestPeakPreservation:
    def test_peak_preserved_in_bin(self) -> None:
        """A single large spike must appear in the bin that contains it."""
        samples = np.zeros(8000, dtype=np.int16)
        samples[4000] = 30000  # spike exactly in the middle
        result = compute_waveform(samples, sample_rate_hz=8000, n_bins=100)
        all_maxes = [b.max_amplitude for b in result.channels[0].bins]
        assert max(all_maxes) == pytest.approx(30000.0 / 32768.0)

    def test_negative_peak_preserved(self) -> None:
        samples = np.zeros(8000, dtype=np.int16)
        samples[100] = -25000
        result = compute_waveform(samples, sample_rate_hz=8000, n_bins=100)
        all_mins = [b.min_amplitude for b in result.channels[0].bins]
        assert min(all_mins) == pytest.approx(-25000.0 / 32768.0)

    def test_constant_signal_min_equals_max(self) -> None:
        samples = np.full(8000, 1000, dtype=np.int16)
        result = compute_waveform(samples, sample_rate_hz=8000)
        for bin_ in result.channels[0].bins:
            assert bin_.min_amplitude == pytest.approx(1000.0 / 32768.0)
            assert bin_.max_amplitude == pytest.approx(1000.0 / 32768.0)


class TestTimeMapping:
    def test_bin_start_time_matches_expected_sample_position(self) -> None:
        """
        For a 1-second mono signal at 10 Hz with 10 bins (1 bin = 1 sample),
        each bin's time should equal sample_index / sample_rate.
        """
        sample_rate = 10
        n_samples = 10
        samples = np.arange(n_samples, dtype=np.float32)
        result = compute_waveform(samples, sample_rate_hz=sample_rate, n_bins=n_samples)
        bins = result.channels[0].bins
        for i, bin_ in enumerate(bins):
            expected_t = i / sample_rate
            assert bin_.time_seconds == pytest.approx(expected_t, abs=1e-9)

    def test_single_bin(self) -> None:
        samples = _make_mono(8000)
        result = compute_waveform(samples, sample_rate_hz=8000, n_bins=1)
        assert result.n_bins == 1
        bins = result.channels[0].bins
        assert len(bins) == 1
        assert bins[0].time_seconds == pytest.approx(0.0)


class TestEdgeCases:
    def test_empty_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_waveform(np.array([], dtype=np.int16), sample_rate_hz=8000)

    def test_invalid_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="sample rate"):
            compute_waveform(_make_mono(100), sample_rate_hz=0)

    def test_negative_n_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            compute_waveform(_make_mono(100), sample_rate_hz=8000, n_bins=0)

    def test_single_sample_audio(self) -> None:
        samples = np.array([42], dtype=np.int16)
        result = compute_waveform(samples, sample_rate_hz=8000, n_bins=1)
        assert result.n_bins == 1
        bins = result.channels[0].bins
        assert bins[0].min_amplitude == pytest.approx(42.0 / 32768.0)
        assert bins[0].max_amplitude == pytest.approx(42.0 / 32768.0)

    def test_float32_audio(self) -> None:
        """float32 samples (normalised −1…+1) should work without errors."""
        samples = np.linspace(-1.0, 1.0, 8000, dtype=np.float32)
        result = compute_waveform(samples, sample_rate_hz=8000)
        assert result.n_bins > 0


class TestLongAudio:
    def test_long_audio_produces_default_bins(self) -> None:
        """A 10-minute signal should still produce DEFAULT_N_BINS bins."""
        n_samples = 44100 * 600  # 10 min at 44.1 kHz
        # Use linspace to avoid allocating a huge integer array.
        samples = np.zeros(n_samples, dtype=np.int16)
        result = compute_waveform(samples, sample_rate_hz=44100)
        assert result.n_bins == DEFAULT_N_BINS


# ---------------------------------------------------------------------------
# API layer — POST /api/audio/analyze
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    def test_mono_wav_returns_analyzed_status(self) -> None:
        samples = _make_mono(8000)
        response = client.post(
            "/api/audio/analyze",
            files={"file": ("tone.wav", _wav_bytes(samples, 8000), "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "analyzed"

    def test_analyze_response_contains_metadata(self) -> None:
        samples = _make_mono(8000)
        response = client.post(
            "/api/audio/analyze",
            files={"file": ("tone.wav", _wav_bytes(samples, 8000), "audio/wav")},
        )
        metadata = response.json()["metadata"]
        assert metadata["filename"] == "tone.wav"
        assert metadata["sample_rate_hz"] == 8000
        assert metadata["channels"] == 1
        assert metadata["duration_seconds"] == pytest.approx(1.0)

    def test_analyze_response_contains_waveform(self) -> None:
        samples = _make_mono(8000)
        response = client.post(
            "/api/audio/analyze",
            files={"file": ("tone.wav", _wav_bytes(samples, 8000), "audio/wav")},
        )
        waveform = response.json()["waveform"]
        assert waveform["n_channels"] == 1
        assert waveform["n_bins"] > 0
        assert len(waveform["channels"]) == 1
        assert len(waveform["channels"][0]["bins"]) == waveform["n_bins"]

    def test_analyze_waveform_bin_structure(self) -> None:
        samples = _make_mono(8000)
        response = client.post(
            "/api/audio/analyze",
            files={"file": ("tone.wav", _wav_bytes(samples, 8000), "audio/wav")},
        )
        first_bin = response.json()["waveform"]["channels"][0]["bins"][0]
        assert "time_seconds" in first_bin
        assert "min_amplitude" in first_bin
        assert "max_amplitude" in first_bin
        assert first_bin["time_seconds"] == pytest.approx(0.0)

    def test_analyze_stereo_wav(self) -> None:
        samples = _make_stereo(8000)
        response = client.post(
            "/api/audio/analyze",
            files={"file": ("stereo.wav", _wav_bytes(samples, 8000), "audio/wav")},
        )
        assert response.status_code == 200
        waveform = response.json()["waveform"]
        assert waveform["n_channels"] == 2
        assert len(waveform["channels"]) == 2

    def test_analyze_missing_file_returns_400(self) -> None:
        response = client.post("/api/audio/analyze")
        assert response.status_code == 400

    def test_analyze_unsupported_type_returns_415(self) -> None:
        response = client.post(
            "/api/audio/analyze",
            files={"file": ("notes.txt", b"not audio", "text/plain")},
        )
        assert response.status_code == 415

    def test_analyze_malformed_wav_returns_400(self) -> None:
        response = client.post(
            "/api/audio/analyze",
            files={"file": ("broken.wav", b"not really a wav", "audio/wav")},
        )
        assert response.status_code == 400

    def test_module02_upload_endpoint_still_works(self) -> None:
        """The /audio/upload endpoint from Module 02 must remain functional."""
        samples = _make_mono(4000)
        response = client.post(
            "/api/audio/upload",
            files={"file": ("tone.wav", _wav_bytes(samples, 8000), "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "loaded"
        # Module 02 response should NOT contain a 'waveform' key.
        assert "waveform" not in response.json()
