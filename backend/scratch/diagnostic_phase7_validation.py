import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dsp_core.denoise import (
    stft_process,
    estimate_noise_minima,
    estimate_noise_imcra,
    apply_spectral_subtraction,
    apply_wiener_filter_dd,
    apply_logmmse_filter,
    apply_omlsa_filter,
    get_first_valid_frame_index
)

def run_method(signal, sr, method):
    frame_length = 2048
    hop_length = 512
    S = stft_process(signal, frame_length=frame_length, hop_length=hop_length)
    P_noisy = np.abs(S)**2
    n_frames, n_freq = S.shape
    initial_frame = get_first_valid_frame_index(frame_length, hop_length)

    P_noise_ms = estimate_noise_minima(P_noisy, initial_frame=initial_frame)
    P_noise_imcra, spp = estimate_noise_imcra(P_noisy, initial_frame=initial_frame)
    
    if method == "ss":
        S_enh = apply_spectral_subtraction(S, P_noise_ms)
        G = np.abs(S_enh) / (np.abs(S) + 1e-12)
        return P_noisy, P_noise_ms, G, None
    elif method == "wiener":
        S_enh = apply_wiener_filter_dd(S, P_noise_ms)
        G = np.abs(S_enh) / (np.abs(S) + 1e-12)
        return P_noisy, P_noise_ms, G, None
    elif method == "logmmse":
        S_enh = apply_logmmse_filter(S, P_noise_ms)
        G = np.abs(S_enh) / (np.abs(S) + 1e-12)
        return P_noisy, P_noise_ms, G, None
    elif method == "omlsa":
        S_enh = apply_omlsa_filter(S, P_noise_imcra, spp)
        G = np.abs(S_enh) / (np.abs(S) + 1e-12)
        return P_noisy, P_noise_imcra, G, spp

def analyze_recording(signal, sr, name):
    print(f"\n====================================")
    print(f"ANALYZING: {name}")
    print(f"Duration: {len(signal)/sr:.3f} s")
    
    methods = ["ss", "wiener", "logmmse", "omlsa"]
    results = {}
    
    for m in methods:
        P_noisy, P_noise, G, spp = run_method(signal, sr, m)
        results[m] = {"P_noisy": P_noisy, "P_noise": P_noise, "G": G, "spp": spp}
        
    print("\n[First 4 Frames Behavior (OM-LSA)]")
    omlsa = results["omlsa"]
    for m in range(4):
        print(f" Frame {m}: P_noisy={np.mean(omlsa['P_noisy'][m]):.4f}, P_noise={np.mean(omlsa['P_noise'][m]):.4f}, SPP={np.mean(omlsa['spp'][m]):.4f}, Gain={np.mean(omlsa['G'][m]):.4f}")
        
    print("\n[Start vs Middle Regions]")
    time_frames = np.arange(len(results["ss"]["P_noisy"])) * 512 / sr
    
    idx_01 = (time_frames < 0.1)
    idx_05 = (time_frames < 0.5)
    idx_10 = (time_frames < 1.0)
    idx_mid = (time_frames >= 1.0) & (time_frames < 2.0)
    
    for m in methods:
        G = results[m]["G"]
        print(f" {m.upper()}:")
        print(f"   Gain 0-0.1s: {np.mean(G[idx_01]):.4f}")
        print(f"   Gain 0-0.5s: {np.mean(G[idx_05]):.4f}")
        print(f"   Gain 0-1.0s: {np.mean(G[idx_10]):.4f}")
        print(f"   Gain 1-2.0s: {np.mean(G[idx_mid]):.4f}")

    return results

def main():
    out_dir = Path("scratch/diagnostic_plots")
    out_dir.mkdir(exist_ok=True)
    
    # 1. Real problematic recording
    real_path = Path("test.wav")
    if real_path.exists():
        sr_real, sig_real = wavfile.read(real_path)
        if sig_real.ndim > 1:
            sig_real = sig_real[:, 0]
        if sig_real.dtype == np.int16:
            sig_real = sig_real.astype(np.float32) / 32768.0
        
        analyze_recording(sig_real, sr_real, "Original Noisy Recording (test.wav)")
        
        # Plot real recording start
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(sig_real))/sr_real, sig_real)
        plt.xlim(0, 0.5)
        plt.title("Waveform - First 0.5s")
        plt.savefig(out_dir / "real_recording_start.png")
        plt.close()
    else:
        print("Note: test.wav not found. Please provide the real recording to analyze it.")
    
    # 2. Synthetic Case A (Noise begins immediately, speech later)
    sr = 44100
    t_a = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    noise_a = np.random.normal(0, 0.05, len(t_a))
    speech_a = np.zeros_like(t_a)
    speech_a[t_a >= 1.0] = 0.5 * np.sin(2 * np.pi * 400 * t_a[t_a >= 1.0])
    sig_a = noise_a + speech_a
    analyze_recording(sig_a.astype(np.float32), sr, "Case A: Noise first, Speech at 1.0s")
    
    # 3. Synthetic Case B (Speech+Noise begins immediately)
    t_b = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    noise_b = np.random.normal(0, 0.05, len(t_b))
    speech_b = 0.5 * np.sin(2 * np.pi * 400 * t_b) # Speech immediately
    sig_b = noise_b + speech_b
    analyze_recording(sig_b.astype(np.float32), sr, "Case B: Speech+Noise immediately")

if __name__ == "__main__":
    main()
