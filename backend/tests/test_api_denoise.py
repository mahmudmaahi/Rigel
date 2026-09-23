import io

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

from app.main import create_app

client = TestClient(create_app())


def _wav_bytes(samples: np.ndarray, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, samples)
    return buffer.getvalue()


def test_denoise_spectral_subtraction() -> None:
    samples = np.random.normal(0, 0.1, 16000).astype(np.float32)
    
    response = client.post(
        "/api/audio/denoise",
        data={
            "method": "spectral_subtraction",
            "alpha": 1.0,
            "beta": 0.01,
        },
        files={"file": ("test.wav", _wav_bytes(samples, 16000), "audio/wav")}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_denoise_wiener() -> None:
    samples = np.random.normal(0, 0.1, 16000).astype(np.float32)
    
    response = client.post(
        "/api/audio/denoise",
        data={
            "method": "wiener",
            "alpha_dd": 0.98,
        },
        files={"file": ("test.wav", _wav_bytes(samples, 16000), "audio/wav")}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_denoise_logmmse() -> None:
    samples = np.random.normal(0, 0.1, 16000).astype(np.float32)
    
    response = client.post(
        "/api/audio/denoise",
        data={
            "method": "logmmse",
            "alpha_dd": 0.98,
        },
        files={"file": ("test.wav", _wav_bytes(samples, 16000), "audio/wav")}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_denoise_imcra() -> None:
    samples = np.random.normal(0, 0.1, 16000).astype(np.float32)
    
    response = client.post(
        "/api/audio/denoise",
        data={
            "method": "imcra",
            "alpha_dd": 0.98,
            "g_min": 0.01,
        },
        files={"file": ("test.wav", _wav_bytes(samples, 16000), "audio/wav")}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_invalid_denoise_method() -> None:
    samples = np.random.normal(0, 0.1, 16000).astype(np.float32)
    
    response = client.post(
        "/api/audio/denoise",
        data={
            "method": "neural_net", # invalid
        },
        files={"file": ("test.wav", _wav_bytes(samples, 16000), "audio/wav")}
    )
    
    assert response.status_code == 422
    assert "Unknown denoising method" in response.text
