import sys
import math
import numpy as np

from dsp_core.denoise import denoise_wiener_dd
from tests.test_denoise_phase4 import _sine, _white_noise, _snr_db, FS, L, H, istft_process, stft_process

def get_metrics(clean, mixed, enhanced, active_mask):
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

def run_diagnostic():
    print("=== Phase 4 Decision-Directed Wiener Filter Diagnostic ===\n")
    
    alpha_vals = [0.5, 0.7, 0.9, 0.98, 0.99]
    n = FS * 5
    
    # Setup Test A: Pulsed Sine
    clean_A = np.zeros(n)
    sine = _sine(440.0, n, amplitude=0.4)
    clean_A[:FS] = sine[:FS]
    clean_A[int(2.5*FS):] = sine[int(2.5*FS):]
    mask_A = np.zeros(n, dtype=bool)
    mask_A[:FS] = True; mask_A[int(2.5*FS):] = True
    noise_A = _white_noise(n, sigma=0.05, seed=200)
    mixed_A = clean_A + noise_A
    
    # Setup Test B: Broadband Multi-tone
    clean_B = np.zeros(n)
    sine = (_sine(300.0, n, amplitude=0.2) + _sine(700.0, n, amplitude=0.13) + 
            _sine(1500.0, n, amplitude=0.1) + _sine(3000.0, n, amplitude=0.1))
    clean_B[:FS] = sine[:FS]
    clean_B[int(2.5*FS):] = sine[int(2.5*FS):]
    mask_B = np.zeros(n, dtype=bool)
    mask_B[:FS] = True; mask_B[int(2.5*FS):] = True
    noise_B = _white_noise(n, sigma=0.05, seed=201)
    mixed_B = clean_B + noise_B
    
    # Setup Test C: Overlapping Noise
    clean_C = np.zeros(n)
    sine = _sine(1000.0, n, amplitude=0.4)
    clean_C[:FS] = sine[:FS]
    clean_C[int(2.5*FS):] = sine[int(2.5*FS):]
    mask_C = np.zeros(n, dtype=bool)
    mask_C[:FS] = True; mask_C[int(2.5*FS):] = True
    white = _white_noise(n, sigma=0.25, seed=202)
    S_noise = stft_process(white, frame_length=L, hop_length=H)
    freqs = np.fft.rfftfreq(L, 1/FS)
    S_noise[:, (freqs < 900) | (freqs > 1100)] = 0.0
    noise_C = istft_process(S_noise, frame_length=L, hop_length=H, original_length=n)
    mixed_C = clean_C + noise_C

    tests = [
        ("Test A (440 Hz Pulsed)", clean_A, mixed_A, mask_A),
        ("Test B (Broadband Multi-tone)", clean_B, mixed_B, mask_B),
        ("Test C (1000 Hz Overlapping)", clean_C, mixed_C, mask_C)
    ]
    
    for name, clean, mixed, mask in tests:
        print(f"--- {name} ---")
        print(f"Input SNR: {_snr_db(clean, mixed - clean):.2f} dB\n")
        print(f"{'alpha_dd':>8} | {'out_SNR':>8} | {'dSNR':>7} | {'corr':>6} | {'sig_ratio':>9} | {'noise_attn':>10}")
        print("-" * 65)
        for a in alpha_vals:
            enhanced = denoise_wiener_dd(mixed, alpha_dd=a)
            _, out_snr, dSNR, corr, sig_ratio, noise_attn = get_metrics(clean, mixed, enhanced[:n], mask)
            
            print(f"{a:8.2f} | {out_snr:8.2f} | {dSNR:7.2f} | {corr:6.3f} | {sig_ratio:9.3f} | {noise_attn:10.2f}")
        print("\n")

if __name__ == "__main__":
    run_diagnostic()
