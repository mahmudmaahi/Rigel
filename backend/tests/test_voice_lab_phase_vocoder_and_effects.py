import math
import pytest
import numpy as np
from dsp_core.time_scale import apply_time_stretch, apply_speed
from dsp_core.pitch_shift import apply_pitch_shift
from dsp_core.effects import apply_timbre_tilt, apply_reverb
SR = 44100

def _sine(freq=440.0, duration=1.0, sr=SR, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2.0 * np.pi * freq * t)

def _dominant_freq(samples, sr):
    win = np.hanning(len(samples))
    sp = np.fft.rfft(samples * win)
    freqs = np.fft.rfftfreq(len(samples), 1 / sr)
    idx = np.argmax(np.abs(sp))
    return freqs[idx]

class TestPhaseVocoderValidation:
    """Validate Phase Vocoder / Speed / Time Stretch / Pitch Shift"""

    def test_time_stretch_2x(self):
        x = _sine(440.0, 1.0, SR)
        y = apply_time_stretch(x, SR, 2.0)
        assert abs(len(y) - 2 * SR) < 1000
        f_dom = _dominant_freq(y, SR)
        assert abs(f_dom - 440.0) < 5.0

    def test_time_stretch_half(self):
        x = _sine(440.0, 1.0, SR)
        y = apply_time_stretch(x, SR, 0.5)
        assert abs(len(y) - 0.5 * SR) < 1000
        f_dom = _dominant_freq(y, SR)
        assert abs(f_dom - 440.0) < 5.0

    def test_speed_2x(self):
        x = _sine(1000.0, 1.0, SR)
        y = apply_speed(x, SR, 2.0)
        assert abs(len(y) - 0.5 * SR) < 1000
        f_dom = _dominant_freq(y, SR)
        assert abs(f_dom - 1000.0) < 5.0

    def test_speed_half(self):
        x = _sine(1000.0, 1.0, SR)
        y = apply_speed(x, SR, 0.5)
        assert abs(len(y) - 2.0 * SR) < 1000
        f_dom = _dominant_freq(y, SR)
        assert abs(f_dom - 1000.0) < 5.0

    def test_pitch_shift_plus_12(self):
        x = _sine(440.0, 1.0, SR)
        y = apply_pitch_shift(x, SR, 12.0)
        assert abs(len(y) - SR) < 1000
        f_dom = _dominant_freq(y, SR)
        assert abs(f_dom - 880.0) < 10.0

    def test_pitch_shift_minus_12(self):
        x = _sine(440.0, 1.0, SR)
        y = apply_pitch_shift(x, SR, -12.0)
        assert abs(len(y) - SR) < 1000
        f_dom = _dominant_freq(y, SR)
        assert abs(f_dom - 220.0) < 10.0

    def test_two_tone_stretch(self):
        x = _sine(440.0, 1.0, SR) + _sine(1000.0, 1.0, SR)
        y = apply_time_stretch(x, SR, 1.5)
        assert abs(len(y) - 1.5 * SR) < 1000

    def test_impulse(self):
        x = np.zeros(SR)
        x[100] = 1.0
        y = apply_speed(x, SR, 2.0)
        assert len(y) > 0 and np.isfinite(y).all()

    def test_silence(self):
        x = np.zeros(SR)
        y = apply_speed(x, SR, 2.0)
        assert np.allclose(y, 0)

    def test_stereo(self):
        x = np.vstack([_sine(440.0, 1.0, SR), _sine(880.0, 1.0, SR)]).T
        y = apply_speed(x, SR, 2.0)
        assert y.shape[1] == 2
        assert abs(y.shape[0] - 0.5 * SR) < 1000

    def test_short_signal(self):
        x = _sine(440.0, 0.05, SR)
        y = apply_time_stretch(x, SR, 2.0)
        assert len(y) > len(x)

    def test_multiple_sample_rates(self):
        for sr in [16000, 48000]:
            x = _sine(440.0, 0.5, sr)
            y = apply_speed(x, sr, 2.0)
            assert abs(len(y) - 0.25 * sr) < 500

class TestEffects:
    """Validate Timbre and Reverb"""

    def test_timbre_neutral(self):
        x = _sine(440.0, 1.0, SR)
        y = apply_timbre_tilt(x, SR, 0.0)
        np.testing.assert_allclose(x, y, rtol=1e-07, atol=1e-07)

    def test_timbre_bright(self):
        x = _sine(440.0, 0.1, SR) + _sine(4000.0, 0.1, SR)
        y = apply_timbre_tilt(x, SR, 1.0)
        assert np.isfinite(y).all()

    def test_reverb_neutral(self):
        x = _sine(440.0, 1.0, SR)
        y = apply_reverb(x, SR, wet=0.0)
        np.testing.assert_allclose(x, y, rtol=1e-07, atol=1e-07)

    def test_reverb_impulse_decay(self):
        x = np.zeros(SR)
        x[0] = 1.0
        y = apply_reverb(x, SR, room_size=0.8, decay=0.5, wet=1.0)
        assert np.sum(np.abs(y[1000:])) > 0.0
        assert np.max(np.abs(y[-1000:])) < 0.01

    def test_reverb_stereo(self):
        x = np.zeros((SR, 2))
        x[0, 0] = 1.0
        x[0, 1] = 0.5
        y = apply_reverb(x, SR, wet=0.5)
        assert y.shape == x.shape