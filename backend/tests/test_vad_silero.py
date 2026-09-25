import pytest
import numpy as np
from app.services.vad_service import SileroVADBackend, VadSession

def test_silero_initialization():
    backend = SileroVADBackend(16000)
    assert backend.source_sample_rate_hz == 16000
    assert backend.threshold == 0.5
    assert backend.current_state == "SILENCE"
    
    # Verify 48k is accepted
    backend48 = SileroVADBackend(48000)
    assert backend48.source_sample_rate_hz == 48000
    assert backend48.source_chunk_size == int(48000 * 0.032) # 1536

def test_engine_switching():
    session1 = VadSession(16000, backend_type="ten_vad")
    assert session1.backend.__class__.__name__ == "TENVADBackend"
    
    session2 = VadSession(16000, backend_type="silero")
    assert session2.backend.__class__.__name__ == "SileroVADBackend"
    
    with pytest.raises(ValueError, match="Unknown VAD backend"):
        VadSession(16000, backend_type="nonexistent")

def test_silero_silence():
    backend = SileroVADBackend(16000)
    silence = np.zeros(512 * 4, dtype=np.float32)
    results = backend.process_chunk(silence)
    
    assert len(results) == 4
    for r in results:
        assert r.is_speech is False
        assert r.state == "SILENCE"
        assert r.activity_score < backend.threshold

def test_silero_speech_simulated():
    backend = SileroVADBackend(16000)
    # Simulate a loud signal, use quiet_voice.wav for neural net
    import soundfile as sf
    import os
    path = os.path.join(os.path.dirname(__file__), 'data', 'quiet_voice.wav')
    waveform, sr = sf.read(path)
    if sr != 16000:
        import scipy.signal
        num_samples = int(len(waveform) * 16000 / sr)
        waveform = scipy.signal.resample(waveform, num_samples)
    
    samples = waveform.astype(np.float32)[:512 * 50] # first ~1.6s
    
    results = backend.process_chunk(samples)
    
    # We expect Silero to detect it as speech after a few chunks
    any_speech = any(r.is_speech for r in results)
    assert any_speech is True
    
    # State persistence check
    assert backend.total_processed_samples == 512 * 50

def test_silero_state_reset():
    # If we create a new session, state should be clean
    backend1 = SileroVADBackend(16000)
    backend2 = SileroVADBackend(16000)
    
    # They should have independent internal buffers/states
    assert backend1.model is not backend2.model

def test_streaming_chunk_processing():
    backend = SileroVADBackend(16000)
    # Give it exactly 1 chunk
    chunk = np.zeros(512, dtype=np.float32)
    res = backend.process_chunk(chunk)
    assert len(res) == 1
    
    # Give it half a chunk
    half = np.zeros(256, dtype=np.float32)
    res2 = backend.process_chunk(half)
    assert len(res2) == 0 # Should buffer
    
    # Give it the rest
    res3 = backend.process_chunk(half)
    assert len(res3) == 1 # Now it processed the buffer
