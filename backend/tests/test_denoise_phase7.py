import numpy as np
import pytest
from dsp_core.denoise import (
    stft_process,
    compute_robust_initial_noise,
    estimate_noise_minima,
    estimate_noise_imcra
)

@pytest.fixture
def dummy_noisy_power():
    # 100 frames, 1025 bins
    np.random.seed(42)
    return np.random.exponential(2.0, (100, 1025))

def test_robust_initial_noise_basic(dummy_noisy_power):
    # Base functionality check
    P_N = compute_robust_initial_noise(dummy_noisy_power, initial_frame=0, percentile=10.0)
    assert P_N.shape == (1025,)
    assert np.all(P_N > 0)
    # The lowest 10% frames of exp(mean=2) will have a mean strictly < 2.0
    assert np.mean(P_N) < 2.0

def test_robust_initial_noise_short_recording():
    # Test fallback when recording has no valid frames left
    P_short = np.ones((2, 1025))
    P_N = compute_robust_initial_noise(P_short, initial_frame=3) # initial_frame clamped to 1
    assert P_N.shape == (1025,)
    assert np.allclose(P_N, 1.0)

def test_robust_initial_noise_excludes_initial_padding(dummy_noisy_power):
    # Make the first 10 frames artificially low (mimicking padding, taking up the full 10%)
    dummy_noisy_power[:10] = 0.0001
    
    # Without padding exclusion, the candidate frames would select these 10 frames.
    P_N_wrong = compute_robust_initial_noise(dummy_noisy_power, initial_frame=0, percentile=10.0)
    assert np.mean(P_N_wrong) < 1.0
    
    # With padding exclusion, they are ignored.
    P_N_correct = compute_robust_initial_noise(dummy_noisy_power, initial_frame=3, percentile=10.0)
    assert np.mean(P_N_correct) > np.mean(P_N_wrong) * 100  # Correct is much larger than the artificial padding

def test_estimators_accept_robust_init(dummy_noisy_power):
    P_N_init = compute_robust_initial_noise(dummy_noisy_power, initial_frame=3)
    
    # Minimum statistics
    P_noise_ms = estimate_noise_minima(
        dummy_noisy_power, 
        initial_frame=3, 
        P_N_initial=P_N_init
    )
    # The first valid frames (m=3) should be initialized to exactly P_N_init, except for the bias term!
    # Wait, MS sets P_noise[m] = bias * P_min. P_min is initialized to P_N_init.
    assert np.allclose(P_noise_ms[3], 1.5 * P_N_init)
    
    # IMCRA
    P_noise_imcra, spp = estimate_noise_imcra(
        dummy_noisy_power, 
        initial_frame=3, 
        P_N_initial=P_N_init
    )
    # P_N should be initialized to P_N_init. At frame 3, P_noise = P_N_init exactly (if SPP is updated correctly).
    # Since P_m = dummy[3], P_N = P_N_init.
    assert P_noise_imcra.shape == (100, 1025)
