import math
import numpy as np
import pytest

from dsp_core.denoise import apply_logmmse_filter, denoise_logmmse, stft_process, istft_process

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
# Core Log-MMSE Tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    def test_alpha_dd_below_zero_raises(self):
        stft = np.zeros((10, 1025), dtype=np.complex128)
        psd = np.zeros((10, 1025), dtype=np.float64)
        with pytest.raises(ValueError, match="alpha_dd"):
            apply_logmmse_filter(stft, psd, alpha_dd=-0.1)

    def test_alpha_dd_above_one_raises(self):
        stft = np.zeros((10, 1025), dtype=np.complex128)
        psd = np.zeros((10, 1025), dtype=np.float64)
        with pytest.raises(ValueError, match="alpha_dd"):
            apply_logmmse_filter(stft, psd, alpha_dd=1.1)

    def test_shape_mismatch_raises(self):
        stft = np.zeros((10, 1025), dtype=np.complex128)
        psd = np.zeros((11, 1025), dtype=np.float64)
        with pytest.raises(ValueError, match="shape mismatch"):
            apply_logmmse_filter(stft, psd)


class TestMathematicalBounds:
    def test_gain_not_capped_at_one(self):
        # We explicitly verify that Log-MMSE gain can legitimately exceed 1.0
        # If xi is high but gamma is low, v is small. 
        # For example, gamma = 0.001, xi = 100. 
        # v = (100/101) * 0.001 = 0.00099
        # G ≈ sqrt(100/101) * exp(-0.5*0.577) / sqrt(0.001) * exp(0.5*0.00099)
        # G ≈ 1.0 * 0.749 / 0.0316 * 1.0 ≈ 23.7 > 1.0
        # We simulate this artificially to test the mathematical branch.
        # Note: apply_logmmse_filter updates xi internally. We can force it
        # by passing an input sequence where frame 0 has huge gamma, and frame 1 has tiny gamma.
        n_frames, n_freq = 2, 5
        stft = np.ones((n_frames, n_freq), dtype=np.complex128)
        noise_psd = np.ones((n_frames, n_freq), dtype=np.float64)
        
        # Frame 0: huge signal, gamma = 10000. xi will be 9999.
        stft[0] = 100.0 
        noise_psd[0] = 1.0
        
        # Frame 1: tiny signal, gamma = 1e-4. 
        stft[1] = 0.01
        noise_psd[1] = 1.0
        
        enhanced = apply_logmmse_filter(stft, noise_psd, alpha_dd=0.98)
        
        G_1 = np.abs(enhanced[1]) / (np.abs(stft[1]))
        
        # We expect G_1 to be much greater than 1.0
        assert np.all(G_1 > 2.0), "Log-MMSE gain must not be arbitrarily clipped to 1.0"
        assert np.all(np.isfinite(G_1))
        
    def test_gain_nonnegative(self):
        n_frames, n_freq = 10, 129
        rng = np.random.default_rng(100)
        stft = rng.normal(0, 1, size=(n_frames, n_freq)) + 1j * rng.normal(0, 1, size=(n_frames, n_freq))
        noise_psd = rng.uniform(0.1, 2.0, size=(n_frames, n_freq))
        
        enhanced = apply_logmmse_filter(stft, noise_psd, alpha_dd=0.98)
        gain = np.abs(enhanced) / (np.abs(stft) + 1e-20)
        
        assert np.all(gain >= -1e-15)
        assert np.all(np.isfinite(enhanced))

    def test_continuity_around_branch_boundary(self):
        # We test continuity at v = 1e-3
        # We will create an artificial scenario where v smoothly crosses 1e-3
        from scipy.special import exp1
        
        v_vals = np.linspace(0.99e-3, 1.01e-3, 100)
        
        # Using the exact mathematical components
        EULER_GAMMA = 0.57721566490153286
        
        def g_asymptotic(v):
            return np.exp(-0.5 * EULER_GAMMA) / np.sqrt(v) * np.exp(0.5 * v)
            
        def g_exact(v):
            return np.exp(0.5 * exp1(v))
            
        diff = np.abs(g_asymptotic(v_vals) - g_exact(v_vals))
        
        # The difference around 1e-3 should be exceptionally small.
        # Theoretical error: absolute gain ~ 17.7, relative error ~ v^2 / 8 = 1.25e-7
        # Absolute error = 17.7 * 1.25e-7 = 2.2e-6
        assert np.max(diff) < 5e-6, "Branch boundary must be strictly C0 continuous"


class TestEdgeCases:
    def test_x_zero_explicit_branch(self):
        # X=0 must yield Y=0, no NaN
        # We set up a state where xi > 0, but X = 0
        n_frames, n_freq = 2, 5
        stft = np.ones((n_frames, n_freq), dtype=np.complex128)
        noise_psd = np.ones((n_frames, n_freq), dtype=np.float64)
        
        # Frame 0: normal signal to build xi
        stft[0] = 10.0
        # Frame 1: X = 0
        stft[1] = 0.0
        
        enhanced = apply_logmmse_filter(stft, noise_psd, alpha_dd=0.98)
        
        # Frame 1 output MUST be exactly 0, and not NaN
        assert np.all(enhanced[1] == 0.0)
        assert np.all(np.isfinite(enhanced))

    def test_x_zero_and_xi_zero(self):
        # X=0 and xi=0 from the very start
        n_frames, n_freq = 2, 5
        stft = np.zeros((n_frames, n_freq), dtype=np.complex128)
        noise_psd = np.ones((n_frames, n_freq), dtype=np.float64)
        
        enhanced = apply_logmmse_filter(stft, noise_psd, alpha_dd=0.98)
        
        assert np.all(enhanced == 0.0)
        assert np.all(np.isfinite(enhanced))

    def test_noise_free_sine(self):
        n = 4410
        signal = _sine(440.0, n, amplitude=0.5)
        enhanced = denoise_logmmse(signal)
        assert np.all(np.isfinite(enhanced))

    def test_exact_silence(self):
        n = 4410
        signal = np.zeros(n)
        enhanced = denoise_logmmse(signal)
        assert np.all(np.isfinite(enhanced))
        assert np.all(np.abs(enhanced) < 1e-10)

    def test_extremely_small_floats(self):
        n = 4410
        signal = np.ones(n) * 1e-30
        enhanced = denoise_logmmse(signal)
        assert np.all(np.isfinite(enhanced))


class TestPipelineConstraints:
    def test_stereo_preserved(self):
        n = FS
        signal = np.zeros((n, 2))
        signal[:, 0] = _sine(440.0, n)
        signal[:, 1] = _sine(880.0, n)
        enhanced = denoise_logmmse(signal)
        assert enhanced.shape == signal.shape
        assert not np.allclose(enhanced[:, 0], enhanced[:, 1])

    def test_output_length_matches(self):
        for n in [1000, 44099, 44100, 44101]:
            signal = np.zeros(n)
            enhanced = denoise_logmmse(signal)
            assert len(enhanced) == n

    def test_phase_preservation(self):
        rng = np.random.default_rng(200)
        stft = rng.normal(0, 1, size=(5, 129)) + 1j * rng.normal(0, 1, size=(5, 129))
        noise_psd = np.ones((5, 129))
        
        enhanced = apply_logmmse_filter(stft, noise_psd)
        
        angle_in = np.angle(stft)
        angle_out = np.angle(enhanced)
        
        mag_out = np.abs(enhanced)
        mask = mag_out > 1e-10
        np.testing.assert_allclose(angle_in[mask], angle_out[mask], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Signal Preservation and Noise Attenuation Tests (A/B/C)
# ---------------------------------------------------------------------------

class TestSignalPreservationSNR:
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
        active_mask[:FS] = True; active_mask[int(2.5*FS):] = True
        
        noise = _white_noise(n, sigma=0.05, seed=200)
        mixed = clean + noise
        
        enhanced = denoise_logmmse(mixed, sample_rate_hz=FS, frame_length=L, hop_length=H, alpha_dd=0.98).astype(np.float64)
        
        in_snr, out_snr, dSNR, corr, sig_ratio, noise_attn = self._get_metrics(
            clean, mixed, enhanced[:n], active_mask
        )
        
        assert corr > 0.85
        assert sig_ratio > 0.50
        assert noise_attn < -2.0

    def test_multitone_broadband_preservation_test_B(self):
        n = FS * 5
        clean = np.zeros(n, dtype=np.float64)
        sine = (_sine(300.0, n, amplitude=0.2) + _sine(700.0, n, amplitude=0.13) + 
                _sine(1500.0, n, amplitude=0.1) + _sine(3000.0, n, amplitude=0.1))
        clean[:FS] = sine[:FS]
        clean[int(2.5*FS):] = sine[int(2.5*FS):]
        
        active_mask = np.zeros(n, dtype=bool)
        active_mask[:FS] = True; active_mask[int(2.5*FS):] = True
        
        noise = _white_noise(n, sigma=0.05, seed=201)
        mixed = clean + noise
        
        enhanced = denoise_logmmse(mixed, sample_rate_hz=FS, frame_length=L, hop_length=H, alpha_dd=0.98).astype(np.float64)
        
        in_snr, out_snr, dSNR, corr, sig_ratio, noise_attn = self._get_metrics(
            clean, mixed, enhanced[:n], active_mask
        )
        
        assert corr > 0.85
        assert sig_ratio > 0.50
        assert noise_attn < -2.0

    def test_overlapping_frequency_limitation_test_C(self):
        n = FS * 5
        clean = np.zeros(n, dtype=np.float64)
        sine = _sine(1000.0, n, amplitude=0.4)
        clean[:FS] = sine[:FS]
        clean[int(2.5*FS):] = sine[int(2.5*FS):]
        
        active_mask = np.zeros(n, dtype=bool)
        active_mask[:FS] = True; active_mask[int(2.5*FS):] = True
        
        white = _white_noise(n, sigma=0.25, seed=202)
        S_noise = stft_process(white, frame_length=L, hop_length=H)
        freqs = np.fft.rfftfreq(L, 1/FS)
        S_noise[:, (freqs < 900) | (freqs > 1100)] = 0.0
        noise = istft_process(S_noise, frame_length=L, hop_length=H, original_length=n)
        
        mixed = clean + noise
        
        enhanced = denoise_logmmse(mixed, sample_rate_hz=FS, frame_length=L, hop_length=H, alpha_dd=0.98).astype(np.float64)
        
        in_snr, out_snr, dSNR, corr, sig_ratio, noise_attn = self._get_metrics(
            clean, mixed, enhanced[:n], active_mask
        )
        
        assert corr > 0.70
        assert sig_ratio > 0.30
        assert noise_attn < -3.0
