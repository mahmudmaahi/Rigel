from app.core.config import get_settings
from app.models.status import HealthResponse, StatusResponse
from dsp_core import CORE_STATUS


def get_health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.version,
    )


def get_project_status() -> StatusResponse:
    return StatusResponse(
        project="Rigel",
        team="Orion",
        phase="Phase 1 - Foundation",
        module="Module 03 - Waveform Visualization",
        api="online",
        dsp_core=CORE_STATUS,
        features=[
            "project skeleton",
            "frontend-backend communication",
            "health/status API",
            "black-and-white UI foundation",
            "WAV upload and metadata extraction",
            "waveform decimation and visualization",
            "audio playback",
        ],
    )
