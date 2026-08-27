from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

from app.main import create_app


client = TestClient(create_app())


def _wav_bytes(samples: np.ndarray, sample_rate: int = 8000) -> bytes:
    buffer = BytesIO()
    wavfile.write(buffer, sample_rate, samples)
    return buffer.getvalue()


def test_valid_wav_upload_returns_audio_metadata() -> None:
    samples = np.array([0, 1000, -1000, 500], dtype=np.int16)

    response = client.post(
        "/api/audio/upload",
        files={"file": ("tone.wav", _wav_bytes(samples), "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    metadata = data["metadata"]

    assert data["status"] == "loaded"
    assert metadata["filename"] == "tone.wav"
    assert metadata["format"] == "wav"
    assert metadata["sample_rate_hz"] == 8000
    assert metadata["channels"] == 1
    assert metadata["samples_per_channel"] == 4
    assert metadata["total_samples"] == 4
    assert metadata["duration_seconds"] == 0.0005
    assert metadata["bit_depth"] == 16
    assert metadata["sample_summary"]["array_shape"] == [4]
    assert metadata["sample_summary"]["dtype"] == "int16"
    assert metadata["sample_summary"]["min_value"] == -1000
    assert metadata["sample_summary"]["max_value"] == 1000


def test_stereo_wav_upload_reports_channels_and_shape() -> None:
    samples = np.array([[0, 100], [200, -200], [300, -300]], dtype=np.int16)

    response = client.post(
        "/api/audio/upload",
        files={"file": ("stereo.wav", _wav_bytes(samples), "audio/wav")},
    )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["channels"] == 2
    assert metadata["samples_per_channel"] == 3
    assert metadata["total_samples"] == 6
    assert metadata["sample_summary"]["array_shape"] == [3, 2]


def test_unsupported_upload_is_rejected() -> None:
    response = client.post(
        "/api/audio/upload",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only WAV audio files are supported right now."


def test_missing_file_is_rejected_with_clear_message() -> None:
    response = client.post("/api/audio/upload")

    assert response.status_code == 400
    assert response.json()["detail"] == "Select a WAV audio file before uploading."


def test_malformed_wav_is_rejected() -> None:
    response = client.post(
        "/api/audio/upload",
        files={"file": ("broken.wav", b"not really a wav", "audio/wav")},
    )

    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]
