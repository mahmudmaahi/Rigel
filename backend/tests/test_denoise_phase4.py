import math
import numpy as np
import pytest

from dsp_core.denoise import (
    apply_wiener_filter_dd,
    denoise_wiener_dd,
    stft_process,
    istft_process
)

# ---------------------------------------------------------------------------
# Test signals and parameters
# ---------------------------------------------------------------------------

FS = 44100
L = 2048
H = 512

def _white_noise(n: int, sigma: float = 0.1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, sigma, size=n)

def _pink_noise(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, size=n)
    S = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    S /= np.sqrt(f)
    return np.fft.irfft(S, n=n)

def _sine(freq: float, n: int, sample_rate: float = 44100.0, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * freq * t)

def _snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    var_c = float(np.mean(clean**2))
    var_n = float(np.mean(noise**2))
    if var_c < 1e-12 or var_n < 1e-12:
        return 0.0
    return 10.0 * math.log10(var_c / var_n)

# ---------------------------------------------------------------------------
# Core Wiener Filter Tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    def test_alpha_dd_below_zero_raises(self):
        stft = np.zeros((10, 1025), dtype=np.complex128)
        psd = np.zeros((10, 1025), dtype=np.float64)
        with pytest.raises(ValueError, match="alpha_dd"):
            apply_wiener_filter_dd(stft, psd, alpha_dd=-0.1)

    def test_alpha_dd_above_one_raises(self):
        stft = np.zeros((10, 1025), dtype=np.complex128)
        psd = np.zeros((10, 1025), dtype=np.float64)
        with pytest.raises(ValueError, match="alpha_dd"):
            apply_wiener_filter_dd(stft, psd, alpha_dd=1.1)

    def test_shape_mismatch_raises(self):
        stft = np.zeros((10, 1025), dtype=np.complex128)
        psd = np.zeros((11, 1025), dtype=np.float64)
        with pytest.raises(ValueError, match="shape mismatch"):
            apply_wiener_filter_dd(stft, psd)


class TestMathematicalBounds:
    def test_gain_strictly_bounded(self):
        # We need realistic frames to exercise the recursion
        n_frames, n_freq = 10, 129
        rng = np.random.default_rng(100)
        stft = rng.normal(0, 1, size=(n_frames, n_freq)) + 1j * rng.normal(0, 1, size=(n_frames, n_freq))
        noise_psd = rng.uniform(0.1, 2.0, size=(n_frames, n_freq))
        
        enhanced = apply_wiener_filter_dd(stft, noise_psd, alpha_dd=0.98)
        
        # Gain is |Y| / |X|
        gain = np.abs(enhanced) / (np.abs(stft) + 1e-20)
        
        # Verify 0 <= G <= 1 mathematically
        assert np.all(gain >= -1e-15)  # allow floating point jitter
        assert np.all(gain <= 1.0 + 1e-15)
        
        # Verify outputs are finite
        assert np.all(np.isfinite(enhanced))

    def test_previous_enhanced_frame_is_used(self):
        """
        CRITICAL DD DETAIL test: verify that the DD recursion uses the
        previous ENHANCED frame |Y(m-1,k)|^2, not the noisy |X(m-1,k)|^2.
        We can test this by providing a huge noisy frame at m=0, but setting
        alpha_dd=0 (so it has no memory) or alpha_dd=1 (so it strictly relies on Y).
        Wait, alpha_dd=1 means term1 = |Y(m-1,k)|^2 / P_N.
        If we compare the output at m=1 for two implementations (one using X, one using Y),
        they will differ if G_0 != 1.
        """
        n_frames, n_freq = 2, 5
        # Frame 0: huge signal, small noise => high SNR => G_0 ≈ 1, but let's make it such that G_0 < 1.
        stft = np.ones((n_frames, n_freq), dtype=np.complex128)
        noise_psd = np.ones((n_frames, n_freq), dtype=np.float64) * 2.0
        
        # Frame 0: P_X = 1, P_N = 2 => gamma = 0.5.
        # xi_0 = max(0.5 - 1, 0) = 0.
        # G_0 = 0 / 1 = 0.
        # Y_0 = 0.
        
        # Frame 1: P_X = 1, P_N = 2 => gamma = 0.5
        # If it uses Y_prev: xi_1 = alpha_dd * (0 / 2) + (1 - alpha_dd) * max(0.5 - 1, 0) = 0. G_1 = 0.
        # If it used X_prev: xi_1 = alpha_dd * (1 / 2) + 0 = 0.5 * alpha_dd. G_1 > 0.
        
        enhanced = apply_wiener_filter_dd(stft, noise_psd, alpha_dd=0.98)
        # We expect G_1 == 0 because Y_0 == 0.
        assert np.all(np.abs(enhanced[1]) < 1e-10), "Implementation must use previous ENHANCED frame, not NOISY frame!"


class TestEdgeCases:
    def test_zero_input(self):
        n = 4410
        signal = np.zeros(n)
        enhanced = denoise_wiener_dd(signal)
        assert np.all(np.isfinite(enhanced))
        assert np.all(np.abs(enhanced) < 1e-7)

    def test_silence_input(self):
        n = 4410
        signal = np.ones(n) * 1e-12
        enhanced = denoise_wiener_dd(signal)
        assert np.all(np.isfinite(enhanced))
        assert np.all(np.abs(enhanced) < 1e-7)

    def test_noise_free_sine(self):
        n = 4410
        signal = _sine(440.0, n, amplitude=0.5)
        enhanced = denoise_wiener_dd(signal)
        assert np.all(np.isfinite(enhanced))
        # Depending on noise estimate bias, some attenuation may occur,
        # but output should not be NaN/Inf.

    def test_single_frame(self):
        # 2048 samples exactly = 1 frame
        signal = np.ones(L)
        enhanced = denoise_wiener_dd(signal, frame_length=L, hop_length=H)
        assert len(enhanced) == L
        assert np.all(np.isfinite(enhanced))

    def test_very_short_signal(self):
        # Shorter than one frame
        signal = np.ones(10)
        enhanced = denoise_wiener_dd(signal, frame_length=L, hop_length=H)
        assert len(enhanced) == 10
        assert np.all(np.isfinite(enhanced))


class TestPipelineConstraints:
    def test_stereo_preserved(self):
        n = FS
        signal = np.zeros((n, 2))
        signal[:, 0] = _sine(440.0, n)
        signal[:, 1] = _sine(880.0, n)
        enhanced = denoise_wiener_dd(signal)
        assert enhanced.shape == signal.shape
        assert not np.allclose(enhanced[:, 0], enhanced[:, 1])

    def test_output_length_matches(self):
        for n in [1000, 44099, 44100, 44101]:
            signal = np.zeros(n)
            enhanced = denoise_wiener_dd(signal)
            assert len(enhanced) == n

    def test_phase_preservation(self):
        # G is real, so angle should be strictly identical
        rng = np.random.default_rng(200)
        stft = rng.normal(0, 1, size=(5, 129)) + 1j * rng.normal(0, 1, size=(5, 129))
        noise_psd = np.ones((5, 129))
        
        enhanced = apply_wiener_filter_dd(stft, noise_psd)
        
        angle_in = np.angle(stft)
        angle_out = np.angle(enhanced)
        
        # Where magnitude is > 0, angle must match
        mag_out = np.abs(enhanced)
        mask = mag_out > 1e-10
        np.testing.assert_allclose(angle_in[mask], angle_out[mask], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Signal Preservation and Noise Attenuation Tests (A/B/C)
# ---------------------------------------------------------------------------

class TestSignalPreservationSNR:
    """
    Identical test philosophy to Phase 3.
    """
    def _get_metrics(
        self, clean: np.ndarray, mixed: np.ndarray, enhanced: np.ndarray, active_mask: np.ndarray
    ):
        in_snr = _snr_db(clean, mixed - clean)
        out_snr = _snr_db(clean, enhanced - clean)
        dSNR = out_snr - in_snr
        
        corr = float(np.corrcoef(clean, enhanced)[0, 1])
        
        rms_clean = np.sqrt(np.mean(clean[active_mask]**2))
        rms_enh = np.sqrt(np.mean(enhanced[active_mask]**2))
        sig_ratio = rms_enh / (rms_clean + 1e-12)
        
        inactive_mask = ~active_mask
        rms_noise = np.sqrt(np.mean(mixed[inactive_mask]**2))
        rms_resid = np.sqrt(np.mean(enhanced[inactive_mask]**2))
        noise_attn_db = 20 * math.log10(rms_resid / (rms_noise + 1e-12))
        
        return in_snr, out_snr, dSNR, corr, sig_ratio, noise_attn_db

    def test_pulsed_sine_preservation_test_A(self):
        n = FS * 5
        clean = np.zeros(n, dtype=np.float64)
        sine = _sine(440.0, n, amplitude=0.4)
        clean[:FS] = sine[:FS]
        clean[int(2.5*FS):] = sine[int(2.5*FS):]
        
        active_mask = np.zeros(n, dtype=bool)
        active_mask[:FS] = True
        active_mask[int(2.5*FS):] = True
        
        noise = _white_noise(n, sigma=0.05, seed=200)
        mixed = clean + noise
        
        enhanced = denoise_wiener_dd(mixed, sample_rate_hz=FS, frame_length=L, hop_length=H, alpha_dd=0.98).astype(np.float64)
        
        in_snr, out_snr, dSNR, corr, sig_ratio, noise_attn = self._get_metrics(
            clean, mixed, enhanced[:n], active_mask
        )
        
        # Note: Wiener DD with alpha_dd=0.98 causes significant signal attenuation initially
        # because the memory assumes speech is absent until gamma overcomes it.
        # We assert reasonable bounds.
        assert corr > 0.80, f"Waveform correlation too low: {corr:.3f}"
        assert sig_ratio > 0.40, f"Signal severely attenuated: ratio={sig_ratio:.3f}"
        assert noise_attn < -2.0, f"Noise not attenuated: {noise_attn:.1f} dB"

    def test_multitone_broadband_preservation_test_B(self):
        n = FS * 5
        clean = np.zeros(n, dtype=np.float64)
        sine = (_sine(300.0, n, amplitude=0.2) + 
                _sine(700.0, n, amplitude=0.13) + 
                _sine(1500.0, n, amplitude=0.1) + 
                _sine(3000.0, n, amplitude=0.1))
        clean[:FS] = sine[:FS]
        clean[int(2.5*FS):] = sine[int(2.5*FS):]
        
        active_mask = np.zeros(n, dtype=bool)
        active_mask[:FS] = True
        active_mask[int(2.5*FS):] = True
        
        noise = _white_noise(n, sigma=0.05, seed=201)
        mixed = clean + noise
        
        enhanced = denoise_wiener_dd(mixed, sample_rate_hz=FS, frame_length=L, hop_length=H, alpha_dd=0.98).astype(np.float64)
        
        in_snr, out_snr, dSNR, corr, sig_ratio, noise_attn = self._get_metrics(
            clean, mixed, enhanced[:n], active_mask
        )
        
        assert corr > 0.80, f"Waveform correlation too low: {corr:.3f}"
        assert sig_ratio > 0.40, f"Signal severely attenuated: ratio={sig_ratio:.3f}"
        assert noise_attn < -2.0, f"Noise not attenuated: {noise_attn:.1f} dB"

    def test_overlapping_frequency_limitation_test_C(self):
        n = FS * 5
        clean = np.zeros(n, dtype=np.float64)
        sine = _sine(1000.0, n, amplitude=0.4)
        clean[:FS] = sine[:FS]
        clean[int(2.5*FS):] = sine[int(2.5*FS):]
        
        active_mask = np.zeros(n, dtype=bool)
        active_mask[:FS] = True
        active_mask[int(2.5*FS):] = True
        
        white = _white_noise(n, sigma=0.25, seed=202)
        S_noise = stft_process(white, frame_length=L, hop_length=H)
        freqs = np.fft.rfftfreq(L, 1/FS)
        S_noise[:, (freqs < 900) | (freqs > 1100)] = 0.0
        noise = istft_process(S_noise, frame_length=L, hop_length=H, original_length=n)
        
        mixed = clean + noise
        
        enhanced = denoise_wiener_dd(mixed, sample_rate_hz=FS, frame_length=L, hop_length=H, alpha_dd=0.98).astype(np.float64)
        
        in_snr, out_snr, dSNR, corr, sig_ratio, noise_attn = self._get_metrics(
            clean, mixed, enhanced[:n], active_mask
        )
        
        assert corr > 0.65, f"Overlapping correlation too low: {corr:.3f}"
        assert sig_ratio > 0.25, f"Overlapping signal severely attenuated: ratio={sig_ratio:.3f}"
        assert noise_attn < -3.0, f"Overlapping noise not attenuated: {noise_attn:.1f} dB"
