from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_measure_offline():
    # Make a dummy request payload
    # 0.5s of silence
    samples = [[0.0] * 8000]
    payload = {
        "samples": samples,
        "sample_rate_hz": 16000
    }
    
    response = client.post("/api/audio/measure", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "loudness" in data
    assert "pitch" in data
    assert "rhythm" in data
    
    assert data["loudness"]["average_rms_dbfs"] == -120.0
    assert data["pitch"]["voiced_percentage"] == 0.0
    assert data["rhythm"]["reliable"] is False
