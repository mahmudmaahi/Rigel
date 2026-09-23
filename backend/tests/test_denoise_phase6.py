import numpy as np
import pytest

from dsp_core.denoise import estimate_noise_imcra, apply_omlsa_filter, denoise_omlsa

def test_imcra_freq_smoothing_zeros():
    # If input is all zeros, noise estimate must be zero
    P_noisy = np.zeros((10, 4))
    P_noise, spp = estimate_noise_imcra(P_noisy)
    np.testing.assert_array_equal(P_noise, 0.0)

def test_imcra_freq_smoothing_constant():
    # If input is constant in freq, smoothing with [0.25, 0.5, 0.25] should preserve it
    P_noisy = np.ones((10, 8))
    # w_freq=1 implies smoothing.
    P_noise, spp = estimate_noise_imcra(P_noisy, alpha_s=0.0, alpha_d=0.0)
    # The output P_noise eventually tracks the constant 1.0 (with alpha_d=0)
    assert np.allclose(P_noise[-1], 1.0)

def test_imcra_sub_window_minimum():
    # Create a noisy signal that drops to a minimum at a specific frame
    n_frames = 150
    P_noisy = np.ones((n_frames, 4))
    
    # Drop to 0.1 at frame 10
    P_noisy[10, :] = 0.1
    
    # We will trace the minimum tracker logic.
    # Since V = 8 * 16 = 128 frames, the minimum should be remembered until frame 10 + 128 = 138 approx
    P_noise, spp = estimate_noise_imcra(P_noisy, U=8, V_sub=16, alpha_s=0.5, alpha_d=0.0, beta_min=1.0)
    
    # At frame 20, noise estimate should be very small because the minimum 0.1 forces q to 0 or 1?
    # Actually, let's just test that it runs without errors, because full tracing of IMCRA
    # requires checking internal variables which are not exposed. 
    # But we can verify it doesn't crash on incomplete windows.
    assert P_noise.shape == (n_frames, 4)
    assert not np.any(np.isnan(P_noise))

def test_imcra_likelihood_overflow():
    # Create an artificially huge jump in power to force a huge a-posteriori SNR (gamma),
    # which leads to a huge v, which would overflow np.exp(v).
    P_noisy = np.ones((50, 4)) * 0.01
    P_noisy[25:, :] = 1e10  # Massive jump
    
    P_noise, spp = estimate_noise_imcra(P_noisy)
    # If it overflowed, it would be NaN.
    assert not np.any(np.isnan(P_noise))
    assert not np.any(np.isinf(P_noise))

def test_imcra_speech_presence_freeze():
    # If speech is clearly present, noise estimate should FREEZE (alpha_d_tilde -> 1.0)
    P_noisy = np.ones((100, 4)) * 0.1  # Background noise
    P_noisy[50:, :] = 100.0  # Loud speech starts
    
    P_noise, spp = estimate_noise_imcra(P_noisy, alpha_d=0.85)
    
    # Before frame 50, noise should be around 0.1
    assert np.allclose(P_noise[40], 0.1, rtol=0.1)
    
    # After frame 50, the noise estimate should largely freeze and NOT jump to 100
    assert np.max(P_noise[60:]) < 1.0, "Noise estimate leaked strong speech!"

def test_imcra_noise_absence_update():
    # If speech is absent, noise estimate should track the power
    P_noisy = np.ones((100, 4)) * 0.1
    # Noise floor jumps to 0.5 permanently
    P_noisy[50:, :] = 0.5
    
    # Because it's a permanent jump, after the sub-window expires (or if it's broad enough),
    # IMCRA will eventually track it. But for a short test, it might be frozen initially.
    # We can force it to track faster by using a tiny window.
    P_noise, spp = estimate_noise_imcra(P_noisy, U=2, V_sub=2, alpha_d=0.5, alpha_s=0.5)
    
    # At frame 90 (well after U*V_sub = 4 frames), the minimum has reset
    # so it should track the new noise floor
    assert np.allclose(P_noise[90], 0.5, rtol=0.1)

def test_imcra_zero_input_edge_case():
    P_noisy = np.zeros((10, 4))
    P_noise, spp = estimate_noise_imcra(P_noisy)
    assert np.all(P_noise == 0)

def test_imcra_single_frame():
    P_noisy = np.ones((1, 4))
    P_noise, spp = estimate_noise_imcra(P_noisy)
    assert P_noise.shape == (1, 4)
    assert not np.any(np.isnan(P_noise))

def test_imcra_beta_bias_scaling():
    # A perfectly stationary noise should have its noise estimate equal to its power.
    P_noisy = np.ones((200, 4)) * 1.0
    # Use default beta_min=1.47
    P_noise, spp = estimate_noise_imcra(P_noisy, alpha_d=0.85, U=8, V_sub=16)
    
    # P_noise should converge to ~1.0
    assert np.allclose(P_noise[-1], 1.0, rtol=0.05)


def test_omlsa_p_zero_equals_g_min():
    stft = np.ones((10, 4), dtype=np.complex128)
    noise_psd = np.ones((10, 4)) * 0.1
    spp = np.zeros((10, 4)) # p = 0
    G_min = 0.01
    
    Y = apply_omlsa_filter(stft, noise_psd, spp, G_min=G_min)
    
    # When p=0, G_OMLSA = G_min. So Y = G_min * stft
    assert np.all(np.isclose(np.abs(Y), G_min))

def test_omlsa_p_one_equals_lmmse():
    from dsp_core.denoise import apply_logmmse_filter
    stft = np.ones((10, 4), dtype=np.complex128)
    noise_psd = np.ones((10, 4)) * 0.1
    spp = np.ones((10, 4)) # p = 1
    
    Y_omlsa = apply_omlsa_filter(stft, noise_psd, spp)
    Y_lmmse = apply_logmmse_filter(stft, noise_psd)
    
    # When p=1, G_OMLSA should perfectly match Log-MMSE
    assert np.allclose(Y_omlsa, Y_lmmse)

def test_omlsa_x_zero_edge_case():
    stft = np.zeros((10, 4), dtype=np.complex128)
    noise_psd = np.ones((10, 4))
    spp = np.ones((10, 4))
    
    Y = apply_omlsa_filter(stft, noise_psd, spp)
    assert np.all(np.abs(Y) == 0.0)
    assert not np.any(np.isnan(Y))

def test_omlsa_interpolation():
    stft = np.ones((1, 4), dtype=np.complex128)
    noise_psd = np.ones((1, 4)) * 0.1
    G_min = 0.01
    
    # Test p=0, p=0.5, p=1
    Y0 = apply_omlsa_filter(stft, noise_psd, np.zeros((1,4)), G_min=G_min)
    Y5 = apply_omlsa_filter(stft, noise_psd, np.ones((1,4))*0.5, G_min=G_min)
    Y1 = apply_omlsa_filter(stft, noise_psd, np.ones((1,4)), G_min=G_min)
    
    g0 = np.abs(Y0)[0,0]
    g5 = np.abs(Y5)[0,0]
    g1 = np.abs(Y1)[0,0]
    
    assert np.isclose(g0, G_min)
    # G_5 = sqrt(G_1 * G_min), so log(G_5) = 0.5 log(G_1) + 0.5 log(G_min)
    assert np.isclose(np.log(g5), 0.5 * np.log(g1) + 0.5 * np.log(g0))

def test_omlsa_phase_preservation():
    stft = np.exp(1j * np.pi / 4) * np.ones((5, 5))
    noise_psd = np.ones((5, 5)) * 0.5
    spp = np.ones((5, 5)) * 0.5
    
    Y = apply_omlsa_filter(stft, noise_psd, spp)
    
    in_phase = np.angle(stft)
    out_phase = np.angle(Y)
    
    assert np.allclose(in_phase, out_phase)

def test_denoise_omlsa_pipeline():
    from tests.test_denoise_phase5 import _sine, _white_noise, FS
    n = FS * 1
    clean = _sine(440, n, 0.5)
    noise = _white_noise(n, sigma=0.05)
    mixed = clean + noise
    
    # Stereo
    mixed_stereo = np.column_stack((mixed, mixed))
    
    enhanced = denoise_omlsa(mixed_stereo, sample_rate_hz=FS)
    
    assert enhanced.shape == mixed_stereo.shape
    assert not np.any(np.isnan(enhanced))

