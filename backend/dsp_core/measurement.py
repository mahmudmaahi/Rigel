"""
dsp_core/measurement.py

Pure DSP algorithms for audio measurement:
- Loudness (RMS and Peak dBFS)
- Pitch / Fundamental Frequency (YIN algorithm)
- Rhythm / BPM (Envelope + Autocorrelation)

Independent of FastAPI and React.
"""

from __future__ import annotations

import numpy as np


def measure_loudness(samples: np.ndarray) -> dict[str, float]:
    """
    Measure RMS and Peak loudness in dBFS.
    Input should be a 1D or 2D NumPy array of floats in [-1.0, 1.0].
    """
    if samples.size == 0:
        return {"rms_dbfs": -120.0, "peak_dbfs": -120.0}

    # Ensure float64 for precision
    x = samples.astype(np.float64, copy=False)
    
    # Peak
    peak_val = np.max(np.abs(x))
    peak_dbfs = 20.0 * np.log10(peak_val) if peak_val > 1e-10 else -120.0
    
    # RMS
    mean_sq = np.mean(x ** 2)
    rms_val = np.sqrt(mean_sq)
    rms_dbfs = 20.0 * np.log10(rms_val) if rms_val > 1e-10 else -120.0
    
    return {
        "rms_dbfs": max(rms_dbfs, -120.0),
        "peak_dbfs": max(peak_dbfs, -120.0),
    }


def _yin_difference(x: np.ndarray, max_tau: int) -> np.ndarray:
    """Step 1 & 2 of YIN: Difference function."""
    W = len(x) - max_tau
    diff = np.zeros(max_tau)
    for tau in range(max_tau):
        diff[tau] = np.sum((x[:W] - x[tau:tau + W]) ** 2)
    return diff


def _yin_cumulative_mean_normalized_difference(diff: np.ndarray) -> np.ndarray:
    """Step 3 of YIN."""
    cmndf = np.zeros_like(diff)
    cmndf[0] = 1.0
    running_sum = 0.0
    for tau in range(1, len(diff)):
        running_sum += diff[tau]
        cmndf[tau] = diff[tau] * tau / (running_sum + 1e-12)
    return cmndf


def _yin_absolute_threshold(cmndf: np.ndarray, threshold: float) -> int:
    """Step 4 of YIN: Absolute threshold."""
    tau_opt = -1
    for tau in range(1, len(cmndf)):
        if cmndf[tau] < threshold:
            while tau + 1 < len(cmndf) and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            tau_opt = tau
            break
    
    if tau_opt == -1:
        # Fallback to global minimum if threshold not met
        tau_opt = np.argmin(cmndf[1:]) + 1
    return tau_opt


def _yin_parabolic_interpolation(cmndf: np.ndarray, tau: int) -> float:
    """Step 5 of YIN: Parabolic interpolation for sub-sample accuracy."""
    if tau > 0 and tau < len(cmndf) - 1:
        s0, s1, s2 = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
        adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0) + 1e-12)
        return tau + adjustment
    return float(tau)


def estimate_pitch_yin_frame(
    frame: np.ndarray,
    sample_rate: int,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    threshold: float = 0.15,
) -> dict:
    """
    Apply YIN algorithm to a single frame.
    frame length must be at least 2 * max_tau.
    """
    # Silence check
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < 1e-4:  # roughly -80 dBFS
        return {"frequency_hz": 0.0, "confidence": 0.0, "voiced": False}

    # Voice-oriented config by default (60Hz to 500Hz)
    min_tau = int(sample_rate / f0_max)
    max_tau = int(sample_rate / f0_min)
    
    if len(frame) <= max_tau:
        # Frame too short
        return {"frequency_hz": 0.0, "confidence": 0.0, "voiced": False}
        
    diff = _yin_difference(frame, max_tau)
    cmndf = _yin_cumulative_mean_normalized_difference(diff)
    
    # Only search within valid range
    cmndf[:min_tau] = 1.0
    
    tau_est = _yin_absolute_threshold(cmndf, threshold)
    
    # Calculate confidence based on cmndf minimum
    min_val = cmndf[tau_est]
    confidence = max(0.0, 1.0 - min_val)
    
    voiced = bool(min_val < threshold and confidence > 0.5)
    
    if voiced and tau_est > 0:
        refined_tau = _yin_parabolic_interpolation(cmndf, tau_est)
        frequency_hz = sample_rate / refined_tau
    else:
        frequency_hz = 0.0
        
    return {
        "frequency_hz": float(frequency_hz),
        "confidence": float(confidence),
        "voiced": voiced,
    }


def aggregate_pitch(samples: np.ndarray, sample_rate: int, frame_ms: int = 40) -> dict:
    """
    Estimate pitch over an entire signal using overlapping frames.
    """
    if samples.ndim == 2:
        samples = samples.mean(axis=1)  # Mix down to mono
        
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = frame_length // 2
    
    # To detect 60 Hz, we need at least 1/60 sec = 16.6ms.
    # A max_tau for 60Hz is ~266 samples at 16kHz.
    # W needs to be at least max_tau. So frame_length must be >= 2 * max_tau.
    # 40ms at 16kHz = 640 samples. 2 * 266 = 532. This works.
    
    pitches = []
    confidences = []
    
    for i in range(0, len(samples) - frame_length, hop_length):
        frame = samples[i:i + frame_length]
        result = estimate_pitch_yin_frame(frame, sample_rate)
        if result["voiced"]:
            pitches.append(result["frequency_hz"])
            confidences.append(result["confidence"])
            
    if not pitches:
        return {
            "average_pitch": 0.0,
            "min_pitch": 0.0,
            "max_pitch": 0.0,
            "voiced_percentage": 0.0,
            "confidence": 0.0
        }
        
    total_frames = max(1, len(samples) // hop_length)
    voiced_percentage = len(pitches) / total_frames
    
    return {
        "average_pitch": float(np.mean(pitches)),
        "min_pitch": float(np.min(pitches)),
        "max_pitch": float(np.max(pitches)),
        "voiced_percentage": float(voiced_percentage),
        "confidence": float(np.mean(confidences)),
    }


def estimate_bpm(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Basic Rhythm / BPM estimator using envelope autocorrelation.
    """
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
        
    if len(samples) < sample_rate * 2:
        return {"bpm": 0.0, "confidence": 0.0, "reliable": False}

    # 1. Full-wave rectification
    rectified = np.abs(samples)
    
    # 2. Downsample heavily (e.g. to 200 Hz) to save computation and focus on rhythm
    target_sr = 200
    decimation_factor = sample_rate // target_sr
    if decimation_factor > 1:
        envelope = np.mean(
            np.pad(rectified, (0, decimation_factor - len(rectified) % decimation_factor))
            .reshape(-1, decimation_factor),
            axis=1
        )
    else:
        envelope = rectified
        
    # Remove DC
    envelope = envelope - np.mean(envelope)
    
    # Autocorrelation
    auto = np.correlate(envelope, envelope, mode='full')
    auto = auto[len(auto)//2:] # Take positive lags
    
    if auto[0] < 1e-10:
        return {"bpm": 0.0, "confidence": 0.0, "reliable": False}
        
    auto = auto / auto[0] # Normalize
    
    # Search for peaks between 50 and 200 BPM
    min_bpm = 50.0
    max_bpm = 200.0
    min_lag = int(target_sr * 60.0 / max_bpm)
    max_lag = int(target_sr * 60.0 / min_bpm)
    
    if max_lag >= len(auto):
        max_lag = len(auto) - 1
        
    if min_lag >= max_lag:
         return {"bpm": 0.0, "confidence": 0.0, "reliable": False}
         
    search_region = auto[min_lag:max_lag]
    best_lag_idx = np.argmax(search_region)
    best_lag = min_lag + best_lag_idx
    
    peak_val = search_region[best_lag_idx]
    
    # Confidence is peak autocorrelation value
    confidence = peak_val
    
    # Stricter reliability threshold for speech
    reliable = bool(confidence > 0.25)
    
    bpm = float(target_sr * 60.0 / best_lag) if reliable else 0.0
    
    return {
        "bpm": bpm,
        "confidence": float(confidence),
        "reliable": reliable,
    }
