from fastapi.testclient import TestClient

from app.main import create_app


client = TestClient(create_app())


def test_health_endpoint_returns_service_status() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "rigel-api",
        "version": "0.1.0",
    }


def test_status_endpoint_returns_current_module() -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Rigel"
    assert data["team"] == "Orion"
    assert data["module"] == "Module 03 - Waveform Visualization"
    assert data["api"] == "online"
    assert data["dsp_core"] == "available"
