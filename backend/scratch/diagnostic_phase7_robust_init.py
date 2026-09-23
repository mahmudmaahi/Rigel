import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dsp_core.denoise import stft_process

def generate_cases(sr=44100):
    signals = {}
    n_samples = sr * 4
    t = np.linspace(0, 4.0, n_samples, endpoint=False)
    
    # Base elements
    speech = np.zeros_like(t)
    env1 = np.exp(-((t - 1.5)**2) / 0.05)
    env2 = np.exp(-((t - 2.5)**2) / 0.05)
    speech += env1 * 0.5 * np.sin(2 * np.pi * 400 * t)
    speech += env2 * 0.5 * np.sin(2 * np.pi * 800 * t)
    
    # Case A: Noise first, speech later
    noise_A = np.random.normal(0, 0.05, n_samples)
    signals['A'] = (noise_A + speech, speech, noise_A)
    
    # Case B: Speech + noise from sample 0
    env_b = np.exp(-((t - 0.2)**2) / 0.05) + np.exp(-((t - 1.2)**2) / 0.05)
    speech_B = env_b * 0.5 * np.sin(2 * np.pi * 400 * t)
    noise_B = np.random.normal(0, 0.05, n_samples)
    signals['B'] = (noise_B + speech_B, speech_B, noise_B)
    
    # Case C: Quiet noise -> loud noise
    noise_C = np.random.normal(0, 0.02, n_samples)
    noise_C[n_samples//2:] = np.random.normal(0, 0.08, n_samples//2)
    signals['C'] = (noise_C + speech, speech, noise_C)
    
    # Case D: Loud noise -> quiet noise
    noise_D = np.random.normal(0, 0.08, n_samples)
    noise_D[n_samples//2:] = np.random.normal(0, 0.02, n_samples//2)
    signals['D'] = (noise_D + speech, speech, noise_D)
    
    # Case E: Speech dominates most, occasional quiet regions
    env_e = np.ones_like(t)
    env_e[(t > 1.8) & (t < 2.2)] = 0.0
    speech_E = env_e * 0.5 * np.sin(2 * np.pi * 400 * t) * (np.sin(2*np.pi*2*t) > 0)
    noise_E = np.random.normal(0, 0.05, n_samples)
    signals['E'] = (noise_E + speech_E, speech_E, noise_E)
    
    # Case F: Continuous speech + stationary noise
    speech_F = 0.5 * np.sin(2 * np.pi * 400 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t)
    noise_F = np.random.normal(0, 0.05, n_samples)
    signals['F'] = (noise_F + speech_F, speech_F, noise_F)
    
    # Case G: Very short recording
    signals['G'] = (noise_A[:4096] + speech[:4096], speech[:4096], noise_A[:4096])
    
    return signals

def run_diagnostic():
    sr = 44100
    cases = generate_cases(sr)
    
    for case, (sig, clean_speech, true_noise) in cases.items():
        print(f"\n{'='*50}\nCASE {case}\n{'='*50}")
        
        S_noisy = stft_process(sig)
        S_speech = stft_process(clean_speech)
        S_noise = stft_process(true_noise)
        
        P_noisy = np.abs(S_noisy)**2
        P_speech = np.abs(S_speech)**2
        P_noise = np.abs(S_noise)**2
        
        # Valid frames only
        P_noisy = P_noisy[3:-3]
        P_speech = P_speech[3:-3]
        P_noise = P_noise[3:-3]
        
        if len(P_noisy) == 0:
            print("Recording too short to analyze.")
            continue
            
        frame_energies = np.sum(P_noisy, axis=1)
        
        for pct in [5, 10, 20]:
            thresh = np.percentile(frame_energies, pct)
            candidates_idx = np.where(frame_energies <= thresh)[0]
            if len(candidates_idx) == 0:
                candidates_idx = np.array([0])
            
            P_candidates = P_noisy[candidates_idx]
            P_speech_candidates = P_speech[candidates_idx]
            P_noise_candidates = P_noise[candidates_idx]
            
            # Contamination measurement
            total_cand_energy = np.sum(P_candidates)
            total_speech_energy = np.sum(P_speech_candidates)
            contamination_ratio = total_speech_energy / (total_cand_energy + 1e-12)
            
            # Estimates
            est_mean = np.mean(P_candidates, axis=0)
            est_median = np.median(P_candidates, axis=0)
            
            # True target depends on the case. 
            # We are initializing the noise floor. The "ideal" initialization would be the noise spectrum at startup.
            # Let's take the true noise power spectrum over the first 100 valid frames (or whatever is available)
            true_startup_noise = np.mean(P_noise[:min(100, len(P_noise))], axis=0)
            
            err_mean = np.mean(np.abs(est_mean - true_startup_noise))
            err_median = np.mean(np.abs(est_median - true_startup_noise))
            
            print(f"[{pct}% Threshold] Candidates: {len(candidates_idx):4d} | Speech Contamination: {contamination_ratio*100:5.1f}%")
            print(f"    Mean Est Err: {err_mean:6.2f} (Est power: {np.mean(est_mean):6.2f}, True: {np.mean(true_startup_noise):6.2f})")
            print(f"    Med  Est Err: {err_median:6.2f} (Est power: {np.mean(est_median):6.2f}, True: {np.mean(true_startup_noise):6.2f})")

if __name__ == "__main__":
    run_diagnostic()
