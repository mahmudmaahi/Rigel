import numpy as np
import pytest

from dsp_core.measurement import (
    measure_loudness,
    estimate_pitch_yin_frame,
    aggregate_pitch,
    estimate_bpm,
)

def make_sine(freq_hz, sr=16000, duration_s=1.0, amplitude=1.0):
    t = np.arange(int(sr * duration_s)) / sr
    return amplitude * np.sin(2 * np.pi * freq_hz * t)

def test_loudness_silence():
    samples = np.zeros(16000)
    res = measure_loudness(samples)
    assert res["rms_dbfs"] == -120.0
    assert res["peak_dbfs"] == -120.0

def test_loudness_sine():
    samples = make_sine(440, amplitude=1.0) # peak 1.0, rms 0.707
    res = measure_loudness(samples)
    assert abs(res["peak_dbfs"] - 0.0) < 1e-5
    # 20*log10(1/sqrt(2)) = -3.0103
    assert abs(res["rms_dbfs"] - (-3.0103)) < 1e-4

def test_pitch_yin_frame_sine():
    sr = 16000
    samples = make_sine(200, sr=sr, duration_s=0.1) # 100ms frame
    res = estimate_pitch_yin_frame(samples, sr)
    assert res["voiced"] is True
    assert abs(res["frequency_hz"] - 200.0) < 2.0
    assert res["confidence"] > 0.8

def test_pitch_yin_frame_silence():
    sr = 16000
    samples = np.zeros(1600)
    res = estimate_pitch_yin_frame(samples, sr)
    assert res["voiced"] is False

def test_aggregate_pitch():
    sr = 16000
    # 0.5s of 100Hz, 0.5s of silence, 0.5s of 200Hz
    part1 = make_sine(100, sr=sr, duration_s=0.5)
    part2 = np.zeros(int(sr * 0.5))
    part3 = make_sine(200, sr=sr, duration_s=0.5)
    
    samples = np.concatenate([part1, part2, part3])
    res = aggregate_pitch(samples, sr)
    
    assert 0.60 < res["voiced_percentage"] < 0.75 # ~66% voiced
    assert abs(res["min_pitch"] - 100.0) < 3.0
    assert abs(res["max_pitch"] - 200.0) < 3.0
    assert 140.0 < res["average_pitch"] < 160.0

def test_bpm_periodic():
    sr = 16000
    # Create a 120 BPM click track (2 clicks per second, 0.5s spacing)
    samples = np.zeros(sr * 4) # 4 seconds
    for i in range(0, len(samples), sr // 2):
        samples[i:i+10] = 1.0 # click
        
    res = estimate_bpm(samples, sr)
    assert res["reliable"] is True
    assert abs(res["bpm"] - 120.0) < 2.0
    assert res["confidence"] > 0.3

def test_bpm_silence():
    sr = 16000
    samples = np.zeros(sr * 2)
    res = estimate_bpm(samples, sr)
    assert res["reliable"] is False
    assert res["bpm"] == 0.0

def test_bpm_noise_speech():
    sr = 16000
    # White noise (like unvoiced speech) should be unreliable
    np.random.seed(42)
    samples = np.random.randn(sr * 3)
    res = estimate_bpm(samples, sr)
    assert res["reliable"] is False
