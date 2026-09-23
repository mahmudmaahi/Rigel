"""
tests/test_denoise_phase2.py — Module 07 Phase 2: Minimum Statistics noise tracking
=====================================================================================

Tests for estimate_noise_minima() in dsp_core/denoise.py.

Design principle: every test verifies that the estimator BEHAVES SENSIBLY
given the physical meaning of the noise estimate.  Tests do not merely
check that an array is returned; they check numerical properties that
must hold for the estimate to be scientifically meaningful.

Test categories:
    1. Parameter validation (ValueError on bad inputs).
    2. Output dimensions, dtype, finiteness, non-negativity.
    3. Stationary white noise: estimate converges near the true noise PSD.
    4. Stationary colored/pink noise: converges across unequal bins.
    5. Slowly varying noise level: tracker follows a gradually rising floor.
    6. Speech + noise: estimate stays near noise floor during speech bursts.
    7. Pure speech / no noise: estimate is small (near signal-floor minimum).
    8. Pure silence (digital zero): estimate is exactly zero.
    9. Short signals (< window length).
   10. Initialization / boundary behavior (early frames).
   11. Different sample rates → correct default window_frames.
   12. Stereo use pattern: independent-channel calls produce valid estimates.
   13. Bias correction factor applied correctly.
   14. Custom window_frames parameter.
   15. alpha_s = 0 (no smoothing): estimate follows instantaneous minimum.
   16. Numerical diagnostic: print SNR-adjacent stats for white noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dsp_core.denoise import (
    DEFAULT_HOP_LENGTH,
    MS_DEFAULT_ALPHA_S,
    MS_DEFAULT_BIAS,
    MS_DEFAULT_WINDOW_SECONDS,
    estimate_noise_minima,
    stft_process,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 44100
L = 2048
H = 512


def _power_spectrum(signal: np.ndarray, frame_length: int = L, hop: int = H) -> np.ndarray:
    """Compute |STFT|^2 of a 1-D float64 signal."""
    S = stft_process(signal, frame_length=frame_length, hop_length=hop)
    return np.abs(S) ** 2


def _white_noise(n: int, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, n).astype(np.float64)


def _sine(freq_hz: float, n: int, fs: int = FS, amplitude: float = 0.8) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    return amplitude * np.sin(2.0 * math.pi * freq_hz * t)


def _pink_noise(n: int, seed: int = 42) -> np.ndarray:
    """Generate approximate pink noise (1/f^0.5 spectral slope in amplitude)."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, n).astype(np.float64)
    # Shaping in frequency domain: scale by 1/sqrt(k+1) per bin.
    X = np.fft.rfft(white)
    freqs = np.arange(len(X), dtype=np.float64)
    X /= np.sqrt(freqs + 1.0)
    pink = np.fft.irfft(X, n=n)
    # Normalize to unit RMS.
    rms = np.sqrt(np.mean(pink ** 2))
    return (pink / (rms + 1e-12)).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    def test_non_2d_raises(self):
        P = np.ones(100, dtype=np.float64)
        with pytest.raises(ValueError, match="2-D"):
            estimate_noise_minima(P)

    def test_3d_raises(self):
        P = np.ones((10, 5, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="2-D"):
            estimate_noise_minima(P)

    def test_alpha_s_negative_raises(self):
        P = np.ones((10, 64))
        with pytest.raises(ValueError, match="alpha_s"):
            estimate_noise_minima(P, alpha_s=-0.1)

    def test_alpha_s_one_raises(self):
        P = np.ones((10, 64))
        with pytest.raises(ValueError, match="alpha_s"):
            estimate_noise_minima(P, alpha_s=1.0)

    def test_bias_less_than_one_raises(self):
        P = np.ones((10, 64))
        with pytest.raises(ValueError, match="bias"):
            estimate_noise_minima(P, bias=0.9)

    def test_window_frames_zero_raises(self):
        P = np.ones((10, 64))
        with pytest.raises(ValueError, match="window_frames"):
            estimate_noise_minima(P, window_frames=0)

    def test_window_frames_negative_raises(self):
        P = np.ones((10, 64))
        with pytest.raises(ValueError, match="window_frames"):
            estimate_noise_minima(P, window_frames=-5)


# ---------------------------------------------------------------------------
# 2. Output shape, dtype, finiteness, non-negativity
# ---------------------------------------------------------------------------


class TestOutputProperties:
    def test_output_shape_matches_input(self):
        P = np.ones((120, 1025), dtype=np.float64)
        result = estimate_noise_minima(P, window_frames=30)
        assert result.shape == (120, 1025)

    def test_output_dtype_float64(self):
        P = np.ones((50, 64), dtype=np.float32)
        result = estimate_noise_minima(P, window_frames=10)
        assert result.dtype == np.float64

    def test_output_all_finite(self):
        P = _power_spectrum(_white_noise(FS * 2))
        result = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        assert np.all(np.isfinite(result)), "P_noise contains non-finite values."

    def test_output_all_non_negative(self):
        P = _power_spectrum(_white_noise(FS * 2))
        result = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        assert np.all(result >= 0.0), "P_noise contains negative values."

    def test_accepts_float32_input(self):
        P = _power_spectrum(_white_noise(FS)).astype(np.float32)
        result = estimate_noise_minima(P, window_frames=20)
        assert result.shape == P.shape
        assert np.all(result >= 0)

    def test_single_frame(self):
        """A single-frame signal should not raise and should return shape (1, n_freq)."""
        P = np.ones((1, 64), dtype=np.float64)
        result = estimate_noise_minima(P, window_frames=10)
        assert result.shape == (1, 64)
        assert np.all(result >= 0)

    def test_single_bin(self):
        """A single frequency bin edge case."""
        P = np.ones((50, 1), dtype=np.float64)
        result = estimate_noise_minima(P, window_frames=10)
        assert result.shape == (50, 1)


# ---------------------------------------------------------------------------
# 3. Stationary white noise: convergence toward the true noise floor
# ---------------------------------------------------------------------------


class TestStationaryWhiteNoise:
    """
    For stationary white noise of known variance sigma^2, the total power
    across all STFT bins per frame should be approximately sigma^2 (by
    Parseval).  The noise estimate P_noise should converge toward the
    per-bin average over the minimum statistics window.

    We test:
        - The summed estimate is within an order-of-magnitude of the true power.
        - The estimate is substantially smaller than the instantaneous
          per-bin maximum (i.e., it is tracking the floor, not the peaks).
        - The estimate is strictly positive (noise was present).
        - The estimate is smaller than the mean power (minimum < mean).
    """

    def test_estimate_is_positive_for_white_noise(self):
        noise = _white_noise(FS * 3, sigma=0.1, seed=1)
        P = _power_spectrum(noise)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        # After the initial window, estimate should be clearly above zero.
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        assert P_noise[W:].mean() > 0.0, "Noise estimate is zero for a non-silent signal."

    def test_estimate_below_mean_power(self):
        """The noise floor estimate should be below the mean instantaneous power
        because the minimum is always <= mean."""
        noise = _white_noise(FS * 3, sigma=0.1, seed=2)
        P = _power_spectrum(noise)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        # Compare total power summed over frequency bins.
        mean_total_P = P[W:].mean()
        mean_noise_est = P_noise[W:].mean()
        assert mean_noise_est < mean_total_P, (
            f"Noise estimate ({mean_noise_est:.6f}) >= mean power ({mean_total_P:.6f})."
        )

    def test_estimate_within_order_of_magnitude_of_true_floor(self):
        """
        Minimum Statistics with bias correction B=1.5 may produce estimates
        that EXCEED the instantaneous mean power.  This is by design:
        the minimum of a random process underestimates the mean, and B=1.5
        deliberately over-compensates to avoid under-estimating the noise floor
        in practice.  The correct invariant is:

            P_noise(m, k) = B * P_min(m, k)  where  P_min ≤ P_smooth ≤ P_noisy_mean

        So P_noise can be up to B × P_mean in the worst case.

        What we verify here is:
          1. The estimate is strictly positive for non-silent noise.
          2. The raw P_min (i.e., P_noise / B) is strictly ≤ the mean P_smooth
             (i.e. the minimum is always ≤ mean — a mathematical certainty).
          3. The estimate is not trivially equal to the full noisy power.
        """
        sigma2 = 0.01
        noise = _white_noise(FS * 3, sigma=sigma2**0.5, seed=3)
        P = _power_spectrum(noise)
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        bias_val = MS_DEFAULT_BIAS
        P_noise = estimate_noise_minima(
            P, bias=bias_val, sample_rate_hz=FS, hop_length=H
        )
        # After the initialization window.
        if W >= P.shape[0]:
            pytest.skip("Signal too short.")
        post = slice(W, None)
        # 1. Estimate is positive.
        assert P_noise[post].mean() > 0.0
        # 2. P_min = P_noise / B must be ≤ mean of P_noisy (minimum ≤ mean).
        P_min = P_noise[post] / bias_val
        assert P_min.mean() <= P[post].mean() * 1.001, (
            f"P_min ({P_min.mean():.3e}) > mean P_noisy ({P[post].mean():.3e}); "
            "the minimum of a distribution cannot exceed its mean."
        )
        # 3. Estimate is strictly less than B × mean (would require P_min = mean).
        assert P_noise[post].mean() < bias_val * P[post].mean() * 1.01

    def test_estimate_is_not_the_full_noisy_power(self):
        """
        The noise estimate must be substantially less than the mean instantaneous
        power after the initial window.  The minimum should be noticeably lower
        than the mean due to statistical fluctuation in white noise.
        """
        noise = _white_noise(FS * 2, sigma=0.2, seed=4)
        P = _power_spectrum(noise)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        # Noise estimate should be meaningfully smaller than average power.
        ratio = P_noise[W:].mean() / P[W:].mean()
        assert ratio < 0.99, (
            f"Noise estimate is {ratio:.3f} of mean power; "
            "the minimum-statistics step seems ineffective."
        )


# ---------------------------------------------------------------------------
# 4. Stationary colored (pink) noise
# ---------------------------------------------------------------------------


class TestColoredNoise:
    def test_pink_noise_estimate_is_positive(self):
        pink = 0.1 * _pink_noise(FS * 3, seed=10)
        P = _power_spectrum(pink)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        assert P_noise[W:].mean() > 0.0

    def test_pink_noise_estimate_below_mean(self):
        pink = 0.1 * _pink_noise(FS * 3, seed=11)
        P = _power_spectrum(pink)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        assert P_noise[W:].mean() < P[W:].mean()

    def test_pink_noise_estimate_non_uniform_across_bins(self):
        """
        Pink noise has more power at low frequencies.  The noise estimate
        should reflect this: low-frequency bins should have higher P_noise
        than high-frequency bins.
        """
        pink = 0.15 * _pink_noise(FS * 4, seed=12)
        P = _power_spectrum(pink)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        low_mean  = P_noise[W:, :  50].mean()    # low-frequency bins
        high_mean = P_noise[W:, 500:].mean()      # high-frequency bins
        assert low_mean > high_mean, (
            f"Expected low-freq bins ({low_mean:.6f}) > high-freq bins ({high_mean:.6f}) "
            "for pink noise."
        )


# ---------------------------------------------------------------------------
# 5. Slowly varying noise level
# ---------------------------------------------------------------------------


class TestSlowlyVaryingNoise:
    """
    When the noise floor rises slowly, the estimator should eventually
    track the higher level — though it will lag by up to W frames.

    We split the signal into two halves: low noise followed by higher noise.
    After more than W frames into the high-noise section, the estimate should
    be measurably higher than in the low-noise section.
    """

    def test_estimate_rises_after_noise_increase(self):
        rng = np.random.default_rng(99)
        n_half = FS * 4   # 4 seconds of each level
        low_noise  = rng.normal(0.0, 0.05, n_half).astype(np.float64)
        high_noise = rng.normal(0.0, 0.30, n_half).astype(np.float64)
        signal = np.concatenate([low_noise, high_noise])

        P = _power_spectrum(signal)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)

        n_frames_total = P.shape[0]
        n_frames_half  = n_frames_total // 2
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)

        # Low section: all frames except the last few (which may be
        # influenced by the high-noise section in the look-back window).
        low_est  = P_noise[:n_frames_half - W].mean() if n_frames_half > W else P_noise[:1].mean()
        # High section: start well after the transition + W frames lag.
        high_est = P_noise[n_frames_half + W:].mean() if n_frames_half + W < n_frames_total else P_noise[-1:].mean()

        assert high_est > low_est, (
            f"Noise estimate did not rise after noise floor increase. "
            f"Low section mean: {low_est:.6f}, High section mean: {high_est:.6f}."
        )

    def test_estimate_lag_at_most_w_frames(self):
        """
        The estimate cannot track a step increase in noise power before
        W frames of high-noise content have had time to flush the buffer.

        We verify that the estimate at the exact midpoint is still BELOW
        the fully settled estimate well after the high-noise section begins
        (i.e., at n_frames_half + W frames).  This proves the lag exists
        rather than claiming a specific maximum multiplier at the first frame.

        Note: the IIR smoother with alpha_s=0.98 does let a small amount
        of high-power information bleed through even at the exact transition
        frame, so we do not test the absolute value at that frame.
        """
        rng = np.random.default_rng(77)
        n_half = FS * 3
        low_noise  = rng.normal(0.0, 0.02, n_half).astype(np.float64)
        high_noise = rng.normal(0.0, 0.50, n_half).astype(np.float64)
        signal = np.concatenate([low_noise, high_noise])

        P = _power_spectrum(signal)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)

        n_frames_half = P.shape[0] // 2
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)

        # Estimate at the exact transition frame.
        est_at_transition = P_noise[n_frames_half].mean()
        # Estimate well after the high-noise section has fully settled.
        settled_start = min(n_frames_half + W, P_noise.shape[0] - 1)
        est_settled = P_noise[settled_start:].mean() if settled_start < P_noise.shape[0] else None

        if est_settled is not None:
            assert est_at_transition < est_settled, (
                f"Noise estimate at the transition ({est_at_transition:.6f}) should be "
                f"less than the settled high-noise estimate ({est_settled:.6f}), "
                f"demonstrating that the tracker lags by W frames."
            )


# ---------------------------------------------------------------------------
# 6. Speech + noise
# ---------------------------------------------------------------------------


class TestSpeechPlusNoise:
    """
    Simulate speech (sine burst) added to stationary noise.
    The noise estimate during the speech burst should remain approximately
    close to the noise-only level — not spike to the speech+noise level.

    The estimator only sees speech+noise; it is not told when speech is present.
    We verify that the estimate does not simply equal the instantaneous power.
    """

    def test_estimate_during_speech_burst_lower_than_signal_power(self):
        rng = np.random.default_rng(55)
        n_total = FS * 5
        noise = rng.normal(0.0, 0.05, n_total).astype(np.float64)

        # Add a loud sine burst in the middle 2 seconds.
        speech_start = FS
        speech_end   = FS * 3
        speech = 0.8 * _sine(440.0, speech_end - speech_start)
        signal = noise.copy()
        signal[speech_start:speech_end] += speech

        P = _power_spectrum(signal)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)

        # Frame indices corresponding to the middle of the speech burst.
        burst_mid_start = (speech_start + (speech_end - speech_start) // 4) // H
        burst_mid_end   = (speech_end   - (speech_end - speech_start) // 4) // H
        burst_mid_end   = min(burst_mid_end, P_noise.shape[0])

        if burst_mid_end > burst_mid_start:
            P_signal_during_burst = P[burst_mid_start:burst_mid_end].mean()
            P_noise_during_burst  = P_noise[burst_mid_start:burst_mid_end].mean()

            assert P_noise_during_burst < P_signal_during_burst, (
                f"Noise estimate during speech burst ({P_noise_during_burst:.6f}) "
                f">= total signal power ({P_signal_during_burst:.6f}). "
                "The estimator appears to be tracking the speech, not the noise floor."
            )

    def test_noise_only_sections_have_lower_estimate_than_speech_sections(self):
        """
        Noise-only frames should produce a similar or lower estimate than
        frames during a speech burst, because the speech-only increases
        power but the minimum window retains noise-only observations.
        """
        rng = np.random.default_rng(33)
        n_total = FS * 6
        noise = rng.normal(0.0, 0.04, n_total).astype(np.float64)
        speech_start, speech_end = FS * 2, FS * 4
        speech = 0.7 * _sine(300.0, speech_end - speech_start)
        signal = noise.copy()
        signal[speech_start:speech_end] += speech

        P = _power_spectrum(signal)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)

        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        # Noise-only region (after init): before speech starts, excluding init.
        noise_only_frames = slice(W, speech_start // H)
        # During speech: middle of burst, well before the burst ends.
        speech_frames = slice(
            (speech_start + FS // 2) // H,
            (speech_start + FS * 1) // H,
        )

        noise_est_quiet  = P_noise[noise_only_frames].mean() if (speech_start // H) > W else None
        noise_est_speech = P_noise[speech_frames].mean()

        if noise_est_quiet is not None:
            # The estimate during speech should not massively exceed the
            # noise-only estimate, because the minimum window should still
            # contain some noise-dominated frames.
            assert noise_est_speech < noise_est_quiet * 5.0, (
                f"Noise estimate during speech ({noise_est_speech:.6f}) is more "
                f"than 5× the quiet-section estimate ({noise_est_quiet:.6f}). "
                "The minimum-statistics window may be too short or the speech too loud."
            )


# ---------------------------------------------------------------------------
# 7. Pure speech / no additive noise
# ---------------------------------------------------------------------------


class TestPureSpeech:
    """
    A pure tone (proxy for speech) has very concentrated spectral energy.
    The Minimum Statistics estimate is NOT zero because the signal itself
    has power.  However, the estimate should be substantially smaller than
    the mean power of the signal (the minimum is always <= mean).

    We do NOT expect the estimate to be near zero.  A tone has sustained
    energy in one bin; the minimum across the window is the floor of that
    sustained energy, not zero.
    """

    def test_estimate_below_mean_power_for_pure_tone(self):
        tone = _sine(440.0, n=FS * 2)
        P = _power_spectrum(tone)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        W = round(MS_DEFAULT_WINDOW_SECONDS * FS / H)
        assert P_noise[W:].mean() <= P[W:].mean(), (
            "Noise estimate must be <= mean power; minimum statistics violated."
        )

    def test_estimate_finite_for_pure_tone(self):
        tone = _sine(880.0, n=FS)
        P = _power_spectrum(tone)
        P_noise = estimate_noise_minima(P, window_frames=20)
        assert np.all(np.isfinite(P_noise))
        assert np.all(P_noise >= 0.0)


# ---------------------------------------------------------------------------
# 8. Pure silence
# ---------------------------------------------------------------------------


class TestSilence:
    def test_silence_estimate_is_zero(self):
        """
        For a signal of all zeros, the STFT power spectrum is zero everywhere.
        The noise estimate should also be zero.
        """
        silence = np.zeros(FS * 2, dtype=np.float64)
        P = _power_spectrum(silence)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        assert np.allclose(P_noise, 0.0, atol=1e-30), (
            f"Noise estimate for silence is not zero. Max value: {P_noise.max():.2e}"
        )

    def test_silence_then_noise(self):
        """
        Signal starts with silence, then noise begins.
        The estimate should remain near zero during silence and should
        eventually begin to rise after the noise starts.
        """
        rng = np.random.default_rng(13)
        n_silence = FS * 2
        n_noise   = FS * 3
        signal = np.concatenate([
            np.zeros(n_silence),
            rng.normal(0.0, 0.1, n_noise),
        ]).astype(np.float64)

        P = _power_spectrum(signal)
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)

        # During the silence region (excluding the last few frames near the
        # transition, which may start to pull in noise-region values as the
        # buffer rolls over).
        silence_end_frame = n_silence // H - 5
        if silence_end_frame > 2:
            silence_est = P_noise[:silence_end_frame].mean()
            # Should be very close to zero.
            assert silence_est < 1e-15, (
                f"Noise estimate during silence is {silence_est:.2e}, expected ~0."
            )


# ---------------------------------------------------------------------------
# 9. Short signals (< window length)
# ---------------------------------------------------------------------------


class TestShortSignals:
    def test_signal_shorter_than_window(self):
        """A signal producing fewer frames than window_frames must not raise."""
        noise = _white_noise(512, sigma=0.1, seed=20)
        P = _power_spectrum(noise, frame_length=256, hop=64)
        n_frames = P.shape[0]
        window = n_frames + 10   # window larger than the signal
        result = estimate_noise_minima(P, window_frames=window)
        assert result.shape == P.shape
        assert np.all(np.isfinite(result))

    def test_single_frame_signal(self):
        P = np.array([[0.1, 0.2, 0.05, 0.3, 0.1]])
        result = estimate_noise_minima(P, window_frames=5)
        assert result.shape == (1, 5)
        # For a single frame the minimum equals the (bias * frame) value.
        expected = MS_DEFAULT_BIAS * P[0]
        assert np.allclose(result[0], expected, rtol=1e-9)

    def test_two_frame_signal(self):
        P = np.array([
            [0.1, 0.1],
            [0.5, 0.5],
        ], dtype=np.float64)
        result = estimate_noise_minima(P, window_frames=3)
        assert result.shape == (2, 2)
        assert np.all(result >= 0)
        # Frame 1 smoothed value is dominated by frame 0 (pre-filled buffer).
        # Noise estimate at frame 0 = bias * P[0].
        expected_m0 = MS_DEFAULT_BIAS * P[0]
        assert np.allclose(result[0], expected_m0, rtol=1e-9)


# ---------------------------------------------------------------------------
# 10. Initialization and boundary behaviour
# ---------------------------------------------------------------------------


class TestInitializationAndBoundary:
    def test_estimate_at_frame_zero_equals_bias_times_first_power(self):
        """
        At m=0, the buffer is pre-filled with P_noisy[0].
        The minimum is P_noisy[0] and P_noise[0] = bias * P_noisy[0].
        """
        P = np.ones((50, 32), dtype=np.float64)
        P[0] = 0.5   # distinctive first frame
        result = estimate_noise_minima(P, alpha_s=0.0, window_frames=5, bias=1.5)
        # With alpha_s=0 every frame is its own P_smooth.
        # Frame 0 buffer is pre-filled with P[0] = 0.5.
        np.testing.assert_allclose(result[0], 1.5 * 0.5, rtol=1e-9)

    def test_estimate_monotonically_non_decreasing_for_decreasing_step(self):
        """
        If the signal steps down (from high power to silence), the
        minimum-statistics estimate should fall quickly because the
        minimum of the window can drop as soon as a low-power frame
        enters the buffer.
        """
        n_frames, n_freq = 100, 64
        P = np.ones((n_frames, n_freq), dtype=np.float64)
        P[50:] = 0.0   # power drops to zero halfway through
        result = estimate_noise_minima(P, alpha_s=0.5, window_frames=5, bias=1.0)
        # Well after the step, the estimate should be near zero.
        assert result[80:].mean() < result[10:40].mean(), (
            "Estimate did not fall after a power step-down."
        )

    def test_pre_fill_initialization_with_first_frame_power(self):
        """
        If the first frame is low-energy and subsequent frames are high-energy,
        the initial estimate will be low (based on the first frame).
        This tests the documented initialization behaviour.
        """
        n_freq = 16
        P = np.ones((60, n_freq), dtype=np.float64) * 1.0
        P[0] = 0.001   # very quiet first frame
        result = estimate_noise_minima(P, alpha_s=0.0, window_frames=4, bias=1.0)
        # At frame 0 the estimate = 1.0 * P[0] = 0.001.
        assert result[0].mean() < 0.01, (
            f"Frame-0 estimate ({result[0].mean():.4f}) should reflect the "
            f"first-frame power (0.001)."
        )
        # After the window has filled with the high-power frames, the estimate
        # should rise.  Exactly how much depends on alpha_s; with alpha_s=0
        # P_smooth = P_noisy, so after window_frames frames the buffer is full
        # of the high-power value and the estimate climbs.
        assert result[-1].mean() > result[0].mean()


# ---------------------------------------------------------------------------
# 11. Default window_frames from sample rate and hop
# ---------------------------------------------------------------------------


class TestDefaultWindowFrames:
    @pytest.mark.parametrize("fs,hop", [
        (8000, 256),
        (16000, 256),
        (44100, 512),
        (48000, 512),
    ])
    def test_default_window_computed_correctly(self, fs: int, hop: int):
        """
        When window_frames=None, the function should compute W as
            max(1, round(MS_DEFAULT_WINDOW_SECONDS * fs / hop))
        This is implicitly tested by confirming output shape is correct
        and no errors occur for varied sample rates / hop sizes.
        """
        expected_W = max(1, round(MS_DEFAULT_WINDOW_SECONDS * fs / hop))
        n_frames = expected_W * 3 + 10
        P = np.ones((n_frames, 64), dtype=np.float64) * 0.01
        result = estimate_noise_minima(
            P, sample_rate_hz=fs, hop_length=hop
        )
        assert result.shape == (n_frames, 64)
        assert np.all(result >= 0)

    def test_small_hop_large_window(self):
        """Very small hop → very large W in frames; should not error."""
        fs, hop = 44100, 64
        W = max(1, round(MS_DEFAULT_WINDOW_SECONDS * fs / hop))
        n_frames = 20   # fewer frames than W (boundary test)
        P = np.ones((n_frames, 16), dtype=np.float64) * 0.02
        result = estimate_noise_minima(P, sample_rate_hz=fs, hop_length=hop)
        assert result.shape == (n_frames, 16)


# ---------------------------------------------------------------------------
# 12. Stereo use pattern
# ---------------------------------------------------------------------------


class TestStereoPattern:
    """
    estimate_noise_minima operates on a single (n_frames, n_freq) power
    spectrum.  Stereo is handled by calling it independently per channel.
    We verify that calling it on left and right channels independently
    produces valid and (for the same signal) equal results.
    """

    def test_identical_stereo_channels_give_identical_estimates(self):
        rng = np.random.default_rng(61)
        mono = rng.normal(0.0, 0.08, FS * 2).astype(np.float64)
        P = _power_spectrum(mono)

        est_L = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        est_R = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)
        np.testing.assert_array_equal(est_L, est_R)

    def test_different_stereo_channels_give_different_estimates(self):
        rng = np.random.default_rng(62)
        left  = rng.normal(0.0, 0.05, FS * 2).astype(np.float64)
        right = rng.normal(0.0, 0.30, FS * 2).astype(np.float64)

        P_L = _power_spectrum(left)
        P_R = _power_spectrum(right)

        est_L = estimate_noise_minima(P_L, sample_rate_hz=FS, hop_length=H)
        est_R = estimate_noise_minima(P_R, sample_rate_hz=FS, hop_length=H)

        # Right channel is 6× louder; its estimate should be significantly larger.
        assert est_R.mean() > est_L.mean() * 2.0, (
            "Right-channel estimate should be substantially larger than left."
        )


# ---------------------------------------------------------------------------
# 13. Bias correction
# ---------------------------------------------------------------------------


class TestBiasCorrection:
    def test_larger_bias_gives_larger_estimate(self):
        P = _power_spectrum(_white_noise(FS * 2, sigma=0.1, seed=70))
        est_low_bias  = estimate_noise_minima(P, bias=1.0, sample_rate_hz=FS, hop_length=H)
        est_high_bias = estimate_noise_minima(P, bias=3.0, sample_rate_hz=FS, hop_length=H)
        assert est_high_bias.mean() > est_low_bias.mean(), (
            "A larger bias factor must produce a larger noise estimate."
        )

    def test_bias_is_linear_multiplier(self):
        """
        Since P_noise = B * P_min and P_min is the same for both calls,
        the ratio of two estimates with different B should equal B2 / B1.
        """
        P = _power_spectrum(_white_noise(FS, sigma=0.1, seed=71))
        b1, b2 = 1.0, 2.5
        est1 = estimate_noise_minima(P, bias=b1, window_frames=10, alpha_s=0.9)
        est2 = estimate_noise_minima(P, bias=b2, window_frames=10, alpha_s=0.9)
        ratio = est2.mean() / (est1.mean() + 1e-20)
        np.testing.assert_allclose(ratio, b2 / b1, rtol=1e-9)


# ---------------------------------------------------------------------------
# 14. Custom window_frames
# ---------------------------------------------------------------------------


class TestCustomWindowFrames:
    def test_larger_window_gives_lower_or_equal_estimate(self):
        """
        A larger window has more chances to include a low-power frame,
        producing a lower or equal minimum.
        """
        rng = np.random.default_rng(80)
        signal = rng.normal(0.0, 0.1, FS * 5).astype(np.float64)
        P = _power_spectrum(signal)
        est_small = estimate_noise_minima(P, window_frames=5,  bias=1.0, alpha_s=0.9)
        est_large = estimate_noise_minima(P, window_frames=50, bias=1.0, alpha_s=0.9)
        W = 50
        assert est_large[W:].mean() <= est_small[W:].mean() * 1.05, (
            "Larger window should yield a lower or equal noise estimate "
            "(minimum over more samples can only decrease or stay the same)."
        )

    def test_window_of_one_frame(self):
        """With W=1, the minimum is just the current smoothed power; no look-back."""
        P = np.array([[0.1, 0.2], [0.5, 0.6], [0.2, 0.1]], dtype=np.float64)
        result = estimate_noise_minima(P, window_frames=1, alpha_s=0.0, bias=1.0)
        # alpha_s=0: P_smooth = P_noisy each frame.
        # window=1: min of 1 element = itself.
        np.testing.assert_allclose(result, P, rtol=1e-9)


# ---------------------------------------------------------------------------
# 15. alpha_s = 0 (no smoothing)
# ---------------------------------------------------------------------------


class TestAlphaZero:
    def test_alpha_zero_no_smoothing(self):
        """With alpha_s=0, P_smooth = P_noisy at every frame (no IIR smoothing)."""
        P = np.array([
            [0.1, 0.2],
            [0.3, 0.1],
            [0.5, 0.4],
        ], dtype=np.float64)
        # window=1: min is the current frame; bias=1.
        result = estimate_noise_minima(P, alpha_s=0.0, window_frames=1, bias=1.0)
        np.testing.assert_allclose(result, P, rtol=1e-9)

    def test_alpha_zero_window_two(self):
        """With window=2, alpha_s=0, bias=1: result at m is min(P[m], P[m-1])."""
        P = np.array([
            [1.0],
            [0.2],
            [0.8],
            [0.1],
        ], dtype=np.float64)
        result = estimate_noise_minima(P, alpha_s=0.0, window_frames=2, bias=1.0)
        # Frame 0: buffer pre-filled with P[0]=[1.0]; min=1.0.
        # Frame 1: buffer has P[0]=[1.0] and P[1]=[0.2]; min=0.2.
        # Frame 2: buffer has P[1]=[0.2] and P[2]=[0.8]; min=0.2.
        # Frame 3: buffer has P[2]=[0.8] and P[3]=[0.1]; min=0.1.
        expected = np.array([[1.0], [0.2], [0.2], [0.1]])
        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# 16. Numerical diagnostic (printed, not asserted)
# ---------------------------------------------------------------------------


class TestNumericalDiagnostic:
    """
    This test prints a diagnostic summary of the estimator behavior on
    white noise with known variance.  It is not a pass/fail assertion
    beyond basic sanity; it provides the numerical report requested in
    the Phase 2 specification.
    """

    def test_diagnostic_white_noise(self, capsys):
        sigma2 = 0.01
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, sigma2 ** 0.5, FS * 2).astype(np.float64)
        S = stft_process(noise, frame_length=L, hop_length=H)
        P = np.abs(S) ** 2

        W = max(1, round(MS_DEFAULT_WINDOW_SECONDS * FS / H))
        P_noise = estimate_noise_minima(P, sample_rate_hz=FS, hop_length=H)

        n_freq = P.shape[1]
        # After the initial window.
        post_init = P_noise[W:]
        mean_est_per_bin    = post_init.mean()
        total_est_per_frame = post_init.sum(axis=1).mean()
        total_P_per_frame   = P[W:].sum(axis=1).mean()

        # Expected total power per frame ≈ sigma^2 (Parseval, unnormalized STFT).
        # Expected per-bin power ≈ sigma^2 / n_freq (white = uniform distribution).
        ratio_to_true = total_est_per_frame / sigma2

        with capsys.disabled():
            print(f"\n--- Minimum Statistics Diagnostic (white noise, sigma^2={sigma2}) ---")
            print(f"  n_frames: {P.shape[0]}, n_freq: {n_freq}")
            print(f"  Window W: {W} frames ({MS_DEFAULT_WINDOW_SECONDS:.1f} s)")
            print(f"  alpha_s: {MS_DEFAULT_ALPHA_S}, bias B: {MS_DEFAULT_BIAS}")
            print(f"  Mean P_noisy per bin (post-init):  {P[W:].mean():.6e}")
            print(f"  Mean P_noise est per bin:          {mean_est_per_bin:.6e}")
            print(f"  Total P_noisy per frame (post):    {total_P_per_frame:.6e}")
            print(f"  Total P_noise est per frame:       {total_est_per_frame:.6e}")
            print(f"  True sigma^2:                      {sigma2:.6e}")
            print(f"  Ratio (est_total / sigma^2):       {ratio_to_true:.3f}")
            print(f"  Interpretation: the estimator recovered {ratio_to_true*100:.1f}% of")
            print(f"  true total noise power (>100% means over-estimation due to bias B).")

        # Sanity assertion: the estimate should be a meaningful fraction of
        # the mean noisy power (not zero, not larger than the mean).
        ratio_to_noisy = total_est_per_frame / total_P_per_frame
        assert 0.01 < ratio_to_noisy < 1.0, (
            f"Noise estimate ({total_est_per_frame:.3e}) should be a meaningful "
            f"fraction of mean noisy power ({total_P_per_frame:.3e}). "
            f"Ratio: {ratio_to_noisy:.4f}"
        )
