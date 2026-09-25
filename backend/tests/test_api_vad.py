import pytest
from fastapi.testclient import TestClient
import numpy as np
import json
from app.main import app
from app.services.vad_service import VadSession

client = TestClient(app)

def test_vad_stream_invalid_payload():
    with client.websocket_connect("/api/vad/stream") as websocket:
        websocket.send_text("not json")
        response = websocket.receive_json()
        assert response["error"] == "Invalid JSON payload"


def test_vad_stream_invalid_sample_rate():
    with client.websocket_connect("/api/vad/stream") as websocket:
        websocket.send_json({"event": "start", "sampleRate": 0, "channels": 1, "format": "float32"})
        response = websocket.receive_json()
        assert response["error"] == "Invalid sampleRate"


def test_vad_stream_invalid_channels():
    with client.websocket_connect("/api/vad/stream") as websocket:
        websocket.send_json({"event": "start", "sampleRate": 16000, "channels": 2, "format": "float32"})
        response = websocket.receive_json()
        assert response["error"] == "Only 1 channel (mono) is supported"


def test_vad_stream_invalid_format():
    with client.websocket_connect("/api/vad/stream") as websocket:
        websocket.send_json({"event": "start", "sampleRate": 16000, "channels": 1, "format": "int16"})
        response = websocket.receive_json()
        assert response["error"] == "Format must be float32"


def test_vad_stream_valid_start_and_stop():
    with client.websocket_connect("/api/vad/stream") as websocket:
        websocket.send_json({"event": "start", "sampleRate": 16000, "channels": 1, "format": "float32"})
        response = websocket.receive_json()
        assert response["event"] == "started"
        
        websocket.send_json({"event": "stop"})
        response = websocket.receive_json()
        assert response["event"] == "stopped"


def test_vad_stream_process_chunk():
    with client.websocket_connect("/api/vad/stream") as websocket:
        # Start session
        websocket.send_json({"event": "start", "sampleRate": 16000, "channels": 1, "format": "float32"})
        assert websocket.receive_json()["event"] == "started"
        
        # Send 1024 floats of silence
        chunk = np.zeros(1024, dtype=np.float32)
        websocket.send_bytes(chunk.tobytes())
        
        # It should process it and send back results
        # 1024 samples / 16000 Hz for silero -> yields some frames (Silero chunk size is 512).
        # We will get 2 frames out for 1024 samples.
        for _ in range(2):
            res = websocket.receive_json()
            assert "timestamp_seconds" in res
            assert "state" in res
            assert "is_speech" in res
            assert res["state"] == "SILENCE"
            assert res["is_speech"] is False


def test_vad_stream_multiple_sessions_isolated():
    # Session 1
    with client.websocket_connect("/api/vad/stream") as ws1:
        ws1.send_json({"event": "start", "sampleRate": 16000, "channels": 1, "format": "float32"})
        assert ws1.receive_json()["event"] == "started"
        
        # Session 2
        with client.websocket_connect("/api/vad/stream") as ws2:
            ws2.send_json({"event": "start", "sampleRate": 16000, "channels": 1, "format": "float32"})
            assert ws2.receive_json()["event"] == "started"
            
            # Send loud speech to ws1
            chunk = (np.ones(1024, dtype=np.float32) * 0.5).astype(np.float32)
            ws1.send_bytes(chunk.tobytes())
            
            # Receive results from ws1
            res1 = [ws1.receive_json() for _ in range(2)]
            assert "activity_score" in res1[-1]
            
            # Send silence to ws2
            chunk2 = np.zeros(1024, dtype=np.float32)
            ws2.send_bytes(chunk2.tobytes())
            
            res2 = [ws2.receive_json() for _ in range(2)]
            assert res2[-1]["activity_score"] < 0.1  # Isolated

def test_vad_stream_virtual_chunking():
    """
    Test that feeding arbitrary-sized chunks through the streaming pipeline
    produces the exact same VAD results as processing the full sample sequence at once.
    """
    # 1. Generate full sample sequence (e.g., 8000 samples = 0.5s at 16kHz)
    # We mix silence and a burst to cause state changes
    total_samples = 8000
    samples = np.zeros(total_samples, dtype=np.float32)
    samples[2000:6000] = 0.5  # loud speech burst

    # 2. Get baseline truth from direct processing
    session = VadSession(sample_rate_hz=16000)
    baseline_results = session.process_chunk(samples)
    baseline_dicts = [r.model_dump() for r in baseline_results]

    # 3. Process via WebSocket with random arbitrary chunks
    chunk_sizes = [500, 1024, 333, 2048, 4095]
    # Verify sum matches
    assert sum(chunk_sizes) == total_samples

    with client.websocket_connect("/api/vad/stream") as ws:
        ws.send_json({"event": "start", "sampleRate": 16000, "channels": 1, "format": "float32"})
        assert ws.receive_json()["event"] == "started"

        ws_results = []
        cursor = 0
        for size in chunk_sizes:
            chunk = samples[cursor : cursor + size]
            ws.send_bytes(chunk.tobytes())
            cursor += size
            
            # Receive whatever is available right now.
            # Starlette TestClient's receive_json blocks if empty, but we can know exactly 
            # how many frames to expect using the FrameBuffer logic, OR we can just wait 
            # until we've received the same number of frames as the baseline.
            # It's safer to just collect them after sending everything? No, websocket
            # messages might queue up, we can just receive exactly len(baseline_results)
            # times total.
            
        for _ in range(len(baseline_results)):
            ws_results.append(ws.receive_json())

        assert len(ws_results) == len(baseline_dicts)

        # Compare results
        for i in range(len(baseline_dicts)):
            br = baseline_dicts[i]
            wr = ws_results[i]
            # Floating point timestamps might differ slightly if serialized differently?
            # Model dump handles serialization.
            assert wr["timestamp_seconds"] == br["timestamp_seconds"]
            assert wr["state"] == br["state"]
            assert wr["is_speech"] == br["is_speech"]
            assert wr["activity_score"] == pytest.approx(br["activity_score"], rel=1e-5)
            assert wr["frames_in_state"] == br["frames_in_state"]
            assert wr["speech_duration_frames"] == br["speech_duration_frames"]

