"""
tests/test_measure_stream_api.py — /api/audio/measure/stream WebSocket

Regression coverage for the live-pitch handshake fix: the server must use the
sample rate the client actually negotiated (sent in the JSON "start" message)
rather than assuming a hardcoded rate for every connection.
"""
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _sine(freq_hz: float, duration_s: float, sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def test_measure_stream_invalid_json():
    with client.websocket_connect("/api/audio/measure/stream") as websocket:
        websocket.send_text("not json")
        response = websocket.receive_json()
        assert response["error"] == "Invalid JSON payload"


def test_measure_stream_invalid_sample_rate():
    with client.websocket_connect("/api/audio/measure/stream") as websocket:
        websocket.send_json({"event": "start", "sampleRate": 0})
        response = websocket.receive_json()
        assert response["error"] == "Invalid sampleRate"


def test_measure_stream_started_ack_sent():
    with client.websocket_connect("/api/audio/measure/stream") as websocket:
        websocket.send_json({"event": "start", "sampleRate": 16000, "channels": 1, "format": "float32"})
        response = websocket.receive_json()
        assert response["event"] == "started"


def test_measure_stream_bytes_before_start_rejected():
    with client.websocket_connect("/api/audio/measure/stream") as websocket:
        chunk = np.zeros(1024, dtype=np.float32)
        websocket.send_bytes(chunk.tobytes())
        response = websocket.receive_json()
        assert response["error"] == "Session not started"

        # Socket must still be usable after a rejected frame.
        websocket.send_json({"event": "start", "sampleRate": 16000, "channels": 1, "format": "float32"})
        assert websocket.receive_json()["event"] == "started"

        websocket.send_bytes(chunk.tobytes())
        response = websocket.receive_json()
        assert "voiced" in response


def test_measure_stream_uses_negotiated_sample_rate():
    """Core regression: a non-16000 negotiated rate must actually be used for YIN.

    A 150 Hz tone generated at 48000 Hz would be misinterpreted (effectively
    aliased to a much higher apparent frequency) if the server still assumed
    a hardcoded 16000 Hz, so this would fail without the handshake fix.
    """
    sr = 48000
    with client.websocket_connect("/api/audio/measure/stream") as websocket:
        websocket.send_json({"event": "start", "sampleRate": sr, "channels": 1, "format": "float32"})
        assert websocket.receive_json()["event"] == "started"

        frame = _sine(150.0, 4096 / sr, sr, amplitude=0.6)
        websocket.send_bytes(frame.tobytes())
        response = websocket.receive_json()

        assert response["voiced"] is True
        assert abs(response["pitch_hz"] - 150.0) < 5.0
