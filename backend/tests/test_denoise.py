"""
tests/test_denoise.py — Module 07: Noise Removal / Speech Enhancement
======================================================================

Phase 1 tests cover the processing STFT/ISTFT infrastructure.
All tests are DSP-correctness tests, not merely execution tests.

Test categories:
    1. _hann_window: shape, dtype, endpoint, peak, symmetry.
    2. stft_process: shape, dtype, phase content, short/long signals,
       stereo rejection, parameter validation.
    3. istft_process: shape, dtype, parameter validation.
    4. Round-trip reconstruction: the critical invariant
           istft_process(stft_process(x)) ≈ x
       tested across many signal lengths, sample rates, frequencies,
       and for silence and boundary edge cases.
    5. Boundary arithmetic: verify that the OLA normalisation buffer
       is non-zero at every sample position produced by stft_process
       + istft_process for the default parameters.
    6. Frame count / padding consistency.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dsp_core.denoise import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    MIN_SIGNAL_LENGTH,
    _POWER_FLOOR,
    _hann_window,
    istft_process,
    stft_process,
)

# ---------------------------------------------------------------------------
# Reconstruction tolerance
# The STFT → ISTFT round-trip produces near-perfect reconstruction.
# "Near-perfect" is defined as max|x - y| < RTOL for all tested lengths.
# This reflects accumulated floating-point arithmetic (typically ~1e-14).
# ---------------------------------------------------------------------------

RTOL_MAX = 1e-7     # strict per-sample absolute tolerance for allclose
RTOL_PEAK = 1e-10   # tolerance for the worst-case single sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Return a float64 mono sine-wave array in [-1, 1]."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    return amplitude * np.sin(2.0 * math.pi * freq_hz * t)


def _white_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Return float64 white noise in [-1, 1]."""
    return rng.uniform(-1.0, 1.0, size=n_samples)


# ---------------------------------------------------------------------------
# 1. _hann_window
# ---------------------------------------------------------------------------


class TestHannWindow:
    def test_length(self):
        w = _hann_window(2048)
        assert len(w) == 2048

    def test_dtype(self):
        w = _hann_window(512)
        assert w.dtype == np.float64

    def test_first_sample_is_zero(self):
        """w[0] must be exactly 0 for the symmetric Hann definition."""
        w = _hann_window(1024)
        assert abs(w[0]) < 1e-15

    def test_last_sample_is_zero(self):
        """w[L-1] must be exactly 0 for the symmetric Hann definition."""
        w = _hann_window(1024)
        assert abs(w[-1]) < 1e-15

    def test_peak_is_one(self):
        """For odd length, the centre sample is exactly 1.0."""
        w = _hann_window(2049)   # odd — centre at index 1024
        assert abs(w[1024] - 1.0) < 1e-12

    def test_symmetry(self):
        """Hann window must be symmetric: w[n] == w[L-1-n]."""
        w = _hann_window(512)
        assert np.allclose(w, w[::-1], atol=1e-12)

    def test_all_nonnegative(self):
        w = _hann_window(2048)
        assert np.all(w >= 0.0)

    def test_length_one(self):
        w = _hann_window(1)
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 1e-15

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            _hann_window(0)


# ---------------------------------------------------------------------------
# 2. stft_process
# ---------------------------------------------------------------------------


class TestStftProcess:
    # ── shape & dtype ──────────────────────────────────────────────────────

    def test_output_dtype_complex(self):
        x = _sine(duration_s=0.5)
        S = stft_process(x, frame_length=1024, hop_length=256)
        assert np.iscomplexobj(S)

    def test_output_ndim(self):
        x = _sine(duration_s=0.5)
        S = stft_process(x, frame_length=1024, hop_length=256)
        assert S.ndim == 2

    def test_frequency_axis_size(self):
        L = 1024
        x = _sine(duration_s=0.5)
        S = stft_process(x, frame_length=L, hop_length=L // 4)
        assert S.shape[1] == L // 2 + 1

    def test_default_hop_is_quarter_frame(self):
        L = 2048
        x = _sine(duration_s=1.0)
        S = stft_process(x, frame_length=L)
        # With hop = L // 4 the number of frames should match our padding.
        H = L // 4
        pad_start = L - H
        pad_end = L
        n_padded = len(x) + pad_start + pad_end
        expected_frames = 1 + (n_padded - L) // H
        assert S.shape[0] == expected_frames

    # ── phase content ──────────────────────────────────────────────────────

    def test_complex_values_have_nonzero_imaginary(self):
        """A real sine has a non-trivial imaginary part in the STFT."""
        x = _sine(freq_hz=440.0, duration_s=0.2)
        S = stft_process(x, frame_length=1024, hop_length=256)
        # Not all imaginary parts should be exactly zero.
        assert not np.allclose(S.imag, 0.0, atol=1e-10)

    def test_silence_has_near_zero_magnitude(self):
        """Pure silence should give an STFT with magnitude near zero."""
        x = np.zeros(44100, dtype=np.float64)
        S = stft_process(x, frame_length=1024, hop_length=256)
        assert np.max(np.abs(S)) < 1e-12

    # ── short signals ──────────────────────────────────────────────────────

    def test_single_sample_signal(self):
        """A signal shorter than one frame must still produce at least one frame."""
        x = np.array([0.5], dtype=np.float64)
        S = stft_process(x, frame_length=64, hop_length=16)
        assert S.ndim == 2
        assert S.shape[0] >= 1
        assert S.shape[1] == 64 // 2 + 1

    def test_signal_exactly_one_frame_long(self):
        L = 512
        x = np.zeros(L, dtype=np.float64)
        x[L // 2] = 1.0
        S = stft_process(x, frame_length=L, hop_length=L // 4)
        assert S.ndim == 2

    # ── parameter validation ───────────────────────────────────────────────

    def test_frame_length_too_small_raises(self):
        with pytest.raises(ValueError, match="frame_length"):
            stft_process(np.zeros(100), frame_length=1)

    def test_hop_length_zero_raises(self):
        with pytest.raises(ValueError, match="hop_length"):
            stft_process(np.zeros(100), frame_length=64, hop_length=0)

    def test_multidimensional_signal_raises(self):
        """stft_process requires a 1-D signal; stereo must be split before calling."""
        x = np.zeros((1000, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="1-D"):
            stft_process(x, frame_length=256, hop_length=64)


# ---------------------------------------------------------------------------
# 3. istft_process — input validation
# ---------------------------------------------------------------------------


class TestIstftProcessValidation:
    def test_frame_length_too_small_raises(self):
        stft = np.ones((10, 33), dtype=np.complex128)
        with pytest.raises(ValueError, match="frame_length"):
            istft_process(stft, frame_length=1)

    def test_hop_length_zero_raises(self):
        stft = np.ones((10, 33), dtype=np.complex128)
        with pytest.raises(ValueError, match="hop_length"):
            istft_process(stft, frame_length=64, hop_length=0)

    def test_wrong_freq_bins_raises(self):
        """n_freq bins in stft must equal L // 2 + 1."""
        L = 64
        stft = np.ones((10, 20), dtype=np.complex128)   # 20 ≠ 33
        with pytest.raises(ValueError, match="frequency axis"):
            istft_process(stft, frame_length=L, hop_length=L // 4)

    def test_1d_stft_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            istft_process(np.ones(10, dtype=np.complex128))


# ---------------------------------------------------------------------------
# 4. Round-trip reconstruction
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """
    Verify that istft_process(stft_process(x)) ≈ x within floating-point
    precision for a broad range of signal lengths.

    The tolerance is:
        max|x - y| < RTOL_PEAK
        np.allclose(x, y, atol=RTOL_MAX)

    These tolerances reflect accumulated double-precision floating-point
    arithmetic, not an algorithm defect.  Perfect exact reconstruction
    is not mathematically achievable in finite-precision arithmetic.
    """

    def _roundtrip(
        self, signal: np.ndarray, L: int = 1024, H: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if H is None:
            H = L // 4
        S = stft_process(signal, frame_length=L, hop_length=H)
        y = istft_process(S, frame_length=L, hop_length=H, original_length=len(signal))
        return signal, y

    # ── silence ────────────────────────────────────────────────────────────

    def test_silence_roundtrip(self):
        x = np.zeros(44100, dtype=np.float64)
        x, y = self._roundtrip(x)
        assert np.allclose(y, 0.0, atol=RTOL_MAX)

    # ── lengths that are exact multiples of hop ────────────────────────────

    @pytest.mark.parametrize("n", [512, 1024, 2048, 4096, 8192])
    def test_exact_multiple_lengths(self, n: int):
        x = _sine(duration_s=n / 44100, sample_rate=44100)[:n]
        x, y = self._roundtrip(x, L=256, H=64)
        assert np.max(np.abs(x - y)) < RTOL_PEAK, (
            f"n={n}: peak error {np.max(np.abs(x - y)):.3e}"
        )

    # ── lengths that are NOT exact multiples of hop ────────────────────────

    @pytest.mark.parametrize("n", [1, 7, 63, 100, 500, 1001, 3333, 22050, 44099])
    def test_off_multiple_lengths(self, n: int):
        rng = np.random.default_rng(seed=n)
        x = _white_noise(n, rng)
        x, y = self._roundtrip(x, L=256, H=64)
        assert np.allclose(x, y, atol=RTOL_MAX), (
            f"n={n}: max error {np.max(np.abs(x - y)):.3e}"
        )

    # ── prime-length signals ───────────────────────────────────────────────

    @pytest.mark.parametrize("n", [997, 1009, 2003, 4001])
    def test_prime_lengths(self, n: int):
        rng = np.random.default_rng(seed=42)
        x = _white_noise(n, rng)
        x, y = self._roundtrip(x, L=512, H=128)
        assert np.allclose(x, y, atol=RTOL_MAX)

    # ── different frame lengths and hops ──────────────────────────────────

    @pytest.mark.parametrize("L,H", [
        (256, 64),
        (512, 128),
        (1024, 256),
        (2048, 512),
    ])
    def test_different_frame_hop_combinations(self, L: int, H: int):
        x = _sine(freq_hz=1000.0, duration_s=0.5)
        x, y = self._roundtrip(x, L=L, H=H)
        assert np.allclose(x, y, atol=RTOL_MAX), (
            f"L={L}, H={H}: max error {np.max(np.abs(x - y)):.3e}"
        )

    # ── different signal types ─────────────────────────────────────────────

    def test_sine_roundtrip(self):
        x = _sine(freq_hz=440.0, duration_s=1.0)
        x, y = self._roundtrip(x, L=2048, H=512)
        assert np.allclose(x, y, atol=RTOL_MAX)

    def test_white_noise_roundtrip(self):
        rng = np.random.default_rng(seed=7)
        x = _white_noise(44100, rng)
        x, y = self._roundtrip(x, L=2048, H=512)
        assert np.allclose(x, y, atol=RTOL_MAX)

    def test_mixed_sine_roundtrip(self):
        """Signal with multiple frequency components."""
        x = _sine(100.0) * 0.3 + _sine(1000.0) * 0.3 + _sine(5000.0) * 0.4
        x = x / np.max(np.abs(x) + 1e-12)
        x, y = self._roundtrip(x, L=2048, H=512)
        assert np.allclose(x, y, atol=RTOL_MAX)

    # ── very short signals ─────────────────────────────────────────────────

    def test_single_sample_roundtrip(self):
        x = np.array([0.7], dtype=np.float64)
        x_orig = x.copy()
        S = stft_process(x, frame_length=64, hop_length=16)
        y = istft_process(S, frame_length=64, hop_length=16, original_length=1)
        assert len(y) == 1
        assert abs(y[0] - x_orig[0]) < RTOL_MAX

    def test_two_sample_roundtrip(self):
        x = np.array([0.5, -0.5], dtype=np.float64)
        S = stft_process(x, frame_length=64, hop_length=16)
        y = istft_process(S, frame_length=64, hop_length=16, original_length=2)
        assert np.allclose(x, y, atol=RTOL_MAX)

    # ── long signal ────────────────────────────────────────────────────────

    def test_long_signal_roundtrip(self):
        """5 seconds at 44.1 kHz = 220 500 samples."""
        rng = np.random.default_rng(seed=99)
        x = _white_noise(220500, rng)
        x, y = self._roundtrip(x, L=2048, H=512)
        assert np.allclose(x, y, atol=RTOL_MAX), (
            f"max error: {np.max(np.abs(x - y)):.3e}"
        )

    # ── sample rates ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("fs", [8000, 16000, 22050, 44100, 48000])
    def test_different_sample_rates_roundtrip(self, fs: int):
        """The STFT is sample-count agnostic; sample rate does not affect reconstruction."""
        n = int(0.3 * fs)
        rng = np.random.default_rng(seed=fs)
        x = _white_noise(n, rng)
        x, y = self._roundtrip(x, L=512, H=128)
        assert np.allclose(x, y, atol=RTOL_MAX)

    # ── output length trimming ─────────────────────────────────────────────

    def test_output_length_matches_original(self):
        for n in [1, 100, 1000, 44100, 44099, 44101]:
            rng = np.random.default_rng(seed=n)
            x = _white_noise(n, rng)
            S = stft_process(x, frame_length=256, hop_length=64)
            y = istft_process(S, frame_length=256, hop_length=64, original_length=n)
            assert len(y) == n, f"Expected length {n}, got {len(y)}"

    def test_output_length_without_original_length(self):
        """Without original_length, istft_process returns a signal of determined length."""
        x = _sine(duration_s=0.5)
        S = stft_process(x, frame_length=512, hop_length=128)
        y = istft_process(S, frame_length=512, hop_length=128)
        # Result should cover at least the original signal
        assert len(y) >= len(x)


# ---------------------------------------------------------------------------
# 5. Boundary OLA normalisation
# ---------------------------------------------------------------------------


class TestOLANormalisation:
    """
    Verify that the element-wise OLA normalisation buffer never contains
    a zero inside the reconstruction range.  A zero at any position would
    produce a divide-by-zero or a missed normalisation, causing a DC offset
    or amplitude spike.
    """

    @pytest.mark.parametrize("n", [1, 64, 512, 1001, 44100])
    def test_norm_buffer_nonzero_everywhere(self, n: int):
        """
        To access the internal norm buffer we verify the reconstruction:
        if any position had norm == 0, the output would be 0 there even
        if the input was non-zero.  We use an all-ones signal as a probe.
        """
        L = 256
        H = L // 4
        x = np.ones(n, dtype=np.float64)
        S = stft_process(x, frame_length=L, hop_length=H)
        y = istft_process(S, frame_length=L, hop_length=H, original_length=n)
        # For an all-ones signal every reconstructed sample should be ≈ 1.
        # A zero in the norm buffer would produce a near-zero output sample.
        assert np.all(y > 0.1), (
            f"n={n}: some reconstructed samples near zero, "
            f"suggesting OLA normalisation failure. min(y)={np.min(y):.4f}"
        )

    def test_window_product_sum_interior(self):
        """
        For Hann window with 75 % overlap (H = L // 4), the sum of
        analysis × synthesis window products at interior samples should be
        approximately 1.5 (the COLA constant for this configuration).
        We verify this by checking the accumulated norm buffer directly
        from a hand-constructed OLA step.
        """
        L = 1024
        H = L // 4
        n_frames = 16
        window = np.hanning(L)
        win2 = window * window

        total_len = (n_frames - 1) * H + L
        norm = np.zeros(total_len)
        for m in range(n_frames):
            norm[m * H : m * H + L] += win2

        # Interior samples are those covered by exactly 4 frames.
        # The centre of the interior region is well away from boundaries.
        interior_start = L
        interior_end = total_len - L
        interior_norm = norm[interior_start:interior_end]
        # COLA constant for Hann + 75 % overlap = 1.5.
        # np.hanning uses the symmetric (non-periodic) definition, so the
        # squared-window sum converges to ≈1.4985 rather than exactly 1.5
        # for interior samples.  We verify near-COLA (within 0.5 % of 1.5).
        assert np.allclose(interior_norm, 1.5, atol=5e-3), (
            f"Interior COLA constant expected ≈1.5, "
            f"got range [{interior_norm.min():.6f}, {interior_norm.max():.6f}]"
        )
        # Also verify no interior sample deviates more than 0.5 % from the mean.
        mean_norm = interior_norm.mean()
        assert np.all(np.abs(interior_norm - mean_norm) < 0.005 * mean_norm), (
            "Interior norm is not uniform — OLA accumulation has a defect."
        )


# ---------------------------------------------------------------------------
# 6. Frame count and padding consistency
# ---------------------------------------------------------------------------


class TestFrameCountPadding:
    """
    Verify that the frame count produced by stft_process is consistent
    with the padding convention documented in the module docstring.
    """

    @pytest.mark.parametrize("n,L,H", [
        (1000, 256, 64),
        (44100, 2048, 512),
        (44099, 2048, 512),
        (44101, 2048, 512),
        (512, 512, 128),
        (1, 64, 16),
    ])
    def test_frame_count_formula(self, n: int, L: int, H: int):
        x = np.zeros(n, dtype=np.float64)
        S = stft_process(x, frame_length=L, hop_length=H)
        pad_start = L - H
        pad_end = L
        n_padded = n + pad_start + pad_end
        expected_frames = 1 + (n_padded - L) // H
        assert S.shape[0] == expected_frames, (
            f"n={n}, L={L}, H={H}: "
            f"expected {expected_frames} frames, got {S.shape[0]}"
        )

    def test_more_frames_for_longer_signal(self):
        L, H = 256, 64
        x_short = np.zeros(1000)
        x_long = np.zeros(2000)
        S_short = stft_process(x_short, frame_length=L, hop_length=H)
        S_long = stft_process(x_long, frame_length=L, hop_length=H)
        assert S_long.shape[0] > S_short.shape[0]
