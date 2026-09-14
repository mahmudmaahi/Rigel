import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_export_audio_success():
    # Provide synthetic 2-channel samples
    payload = {
        "sample_rate_hz": 44100,
        "samples": [
            [0.1, -0.1, 0.5],
            [-0.1, 0.1, -0.5]
        ]
    }
    response = client.post("/api/export", json=payload)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert "attachment; filename=" in response.headers["content-disposition"]
    
    # Read response content
    audio_bytes = response.content
    assert len(audio_bytes) > 44  # WAV header is 44 bytes

def test_export_audio_missing_samples():
    payload = {
        "sample_rate_hz": 44100,
        "samples": []
    }
    response = client.post("/api/export", json=payload)
    
    assert response.status_code == 400
    assert "No audio samples" in response.json()["detail"] or "Empty channels" in response.json()["detail"]

def test_export_audio_invalid_sample_rate():
    payload = {
        "sample_rate_hz": 0,
        "samples": [[0.0]]
    }
    response = client.post("/api/export", json=payload)
    
    assert response.status_code == 400
    assert "Invalid sample rate" in response.json()["detail"]

def test_export_audio_mismatched_channel_lengths():
    payload = {
        "sample_rate_hz": 44100,
        "samples": [
            [0.1, -0.1, 0.5],
            [-0.1, 0.1]  # Shorter channel
        ]
    }
    response = client.post("/api/export", json=payload)
    
    assert response.status_code == 400
    assert "same number of samples" in response.json()["detail"]
