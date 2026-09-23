"""
Diagnostic Script: Investigation of Initial / Cold-Start Behavior

Goals:
1. Trace the complete processing path from the first STFT frame onward.
2. Diagnose P_noise (Minimum Statistics vs IMCRA).
3. Diagnose IMCRA Speech-Presence Probability (SPP).
4. Diagnose the Gain Mask G(m,k).
5. Compare the four methods (Spectral Subtraction, Wiener, Log-MMSE, OM-LSA).
6. Investigate the beginning separately vs later segments.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure the backend directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dsp_core.denoise import (
    stft_process,
    estimate_noise_minima,
    estimate_noise_imcra,
    apply_spectral_subtraction,
    apply_wiener_filter_dd,
    apply_logmmse_filter,
    apply_omlsa_filter,
)

def generate_test_signal(sample_rate: int = 44100, duration: float = 4.0) -> np.ndarray:
    """
    Generate a test signal:
    0.0 - 1.0s: Pure noise
    1.0 - 3.0s: Noise + speech-like tones (harmonics)
    3.0 - 4.0s: Pure noise
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Background noise (white noise, sigma=0.05)
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(t))
    
    # Speech-like tone (fundamental 200 Hz + harmonics)
    speech = np.zeros_like(t)
    speech_mask = (t >= 1.0) & (t <= 3.0)
    f0 = 200.0
    for h in range(1, 6):
        speech[speech_mask] += (0.2 / h) * np.sin(2 * np.pi * (f0 * h) * t[speech_mask])
        
    # Combine
    signal = noise + speech
    return signal.astype(np.float32)

def compute_snr(clean: np.ndarray, noisy_or_enhanced: np.ndarray) -> float:
    eps = 1e-10
    signal_power = np.mean(clean**2)
    noise_power = np.mean((clean - noisy_or_enhanced)**2)
    return 10 * np.log10(signal_power / (noise_power + eps))

def run_diagnostics():
    sr = 44100
    signal = generate_test_signal(sr, 4.0)
    
    # STFT
    frame_length = 2048
    hop_length = 512
    S = stft_process(signal, frame_length=frame_length, hop_length=hop_length)
    P_noisy = np.abs(S)**2
    n_frames, n_freq = S.shape
    
    print(f"Signal generated: {len(signal)} samples, {n_frames} STFT frames.")
    
    # Time axis for frames
    time_frames = np.arange(n_frames) * hop_length / sr
    
    # ---------------------------------------------------------
    # 1. NOISE ESTIMATION
    from dsp_core.denoise import get_first_valid_frame_index
    initial_frame = get_first_valid_frame_index(frame_length, hop_length)
    
    print("\n--- Estimating Noise ---")
    # Minimum Statistics (SS, Wiener, Log-MMSE)
    P_noise_ms = estimate_noise_minima(P_noisy, alpha_s=0.98, window_frames=129, bias=1.5, initial_frame=initial_frame)
    
    # IMCRA (OM-LSA)
    P_noise_imcra, spp_imcra = estimate_noise_imcra(
        P_noisy, alpha_s=0.86, alpha_d=0.85, alpha_dd=0.98, initial_frame=initial_frame
    )
    
    # ---------------------------------------------------------
    # 2. GAIN CALCULATION
    # ---------------------------------------------------------
    print("--- Calculating Gains ---")
    
    S_enh_ss = apply_spectral_subtraction(S, P_noise_ms, alpha=1.0, beta=0.01)
    G_ss = np.abs(S_enh_ss) / (np.abs(S) + 1e-12)
    
    S_enh_wiener = apply_wiener_filter_dd(S, P_noise_ms, alpha_dd=0.98)
    G_wiener = np.abs(S_enh_wiener) / (np.abs(S) + 1e-12)
    
    S_enh_logmmse = apply_logmmse_filter(S, P_noise_ms, alpha_dd=0.98)
    G_logmmse = np.abs(S_enh_logmmse) / (np.abs(S) + 1e-12)
    
    S_enh_omlsa = apply_omlsa_filter(S, P_noise_imcra, spp_imcra, alpha_dd=0.98, G_min=0.01)
    G_omlsa = np.abs(S_enh_omlsa) / (np.abs(S) + 1e-12)
    
    # ---------------------------------------------------------
    # 3. REGION ANALYSIS (INITIAL vs MIDDLE)
    # ---------------------------------------------------------
    print("\n--- Region Analysis ---")
    
    # 0 to 0.5s (Initial pure noise region)
    idx_init = (time_frames < 0.5)
    # 2.0 to 2.5s (Middle speech+noise region)
    idx_mid = (time_frames >= 2.0) & (time_frames < 2.5)
    
    methods = {
        "Spectral Sub": (P_noise_ms, G_ss),
        "DD Wiener": (P_noise_ms, G_wiener),
        "Log-MMSE": (P_noise_ms, G_logmmse),
        "IMCRA+OM-LSA": (P_noise_imcra, G_omlsa)
    }
    
    for name, (P_n, G) in methods.items():
        print(f"[{name}]")
        
        # P_noise
        mean_pn_init = np.mean(P_n[idx_init])
        mean_pn_mid = np.mean(P_n[idx_mid])
        
        # Gain
        mean_g_init = np.mean(G[idx_init])
        mean_g_mid = np.mean(G[idx_mid])
        
        print(f"  P_noise: Init={mean_pn_init:.5f}, Mid={mean_pn_mid:.5f}")
        print(f"  Mean G : Init={mean_g_init:.5f}, Mid={mean_g_mid:.5f}")
        
    print("\n[IMCRA SPP]")
    mean_spp_init = np.mean(spp_imcra[idx_init])
    mean_spp_mid = np.mean(spp_imcra[idx_mid])
    print(f"  Mean SPP: Init={mean_spp_init:.5f}, Mid={mean_spp_mid:.5f}")

    print("\n--- First 10 Frames Analysis ---")
    for m in range(10):
        print(f"Frame {m}:")
        print(f"  SPP(mean_k) = {np.mean(spp_imcra[m]):.4f}")
        print(f"  G_ss(mean_k) = {np.mean(G_ss[m]):.4f}")
        print(f"  G_wiener(mean_k) = {np.mean(G_wiener[m]):.4f}")
        print(f"  G_logmmse(mean_k) = {np.mean(G_logmmse[m]):.4f}")
        print(f"  G_omlsa(mean_k) = {np.mean(G_omlsa[m]):.4f}")
        print(f"  P_noisy(mean_k) = {np.mean(P_noisy[m]):.4f}")
        print(f"  P_noise_imcra(mean_k) = {np.mean(P_noise_imcra[m]):.4f}")

    # ---------------------------------------------------------
    # 4. PLOTTING
    # ---------------------------------------------------------
    out_dir = Path("scratch/diagnostic_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_frames, np.mean(P_noisy, axis=1), label="P_noisy (mean over bins)", color='gray', alpha=0.5)
    plt.plot(time_frames, np.mean(P_noise_ms, axis=1), label="P_noise (Min Stat)", color='blue')
    plt.plot(time_frames, np.mean(P_noise_imcra, axis=1), label="P_noise (IMCRA)", color='red')
    plt.title("Noise Power Estimate over Time")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_frames, np.mean(G_ss, axis=1), label="SS Gain")
    plt.plot(time_frames, np.mean(G_wiener, axis=1), label="Wiener Gain")
    plt.plot(time_frames, np.mean(G_logmmse, axis=1), label="Log-MMSE Gain")
    plt.plot(time_frames, np.mean(G_omlsa, axis=1), label="OM-LSA Gain")
    plt.title("Mean Gain over Time")
    plt.ylabel("Gain [0, 1]")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_frames, np.mean(spp_imcra, axis=1), label="IMCRA SPP", color='purple')
    plt.title("Mean Speech-Presence Probability (SPP) over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Probability [0, 1]")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(out_dir / "phase7_startup_analysis.png")
    plt.close()
    
    print("\nSaved plot to scratch/diagnostic_plots/phase7_startup_analysis.png")

if __name__ == "__main__":
    run_diagnostics()
