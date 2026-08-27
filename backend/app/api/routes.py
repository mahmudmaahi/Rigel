from fastapi import APIRouter, File, UploadFile

from app.models.audio import AudioAnalyzeResponse, AudioUploadResponse
from app.models.status import HealthResponse, StatusResponse
from app.services.audio_service import analyze_uploaded_audio, load_uploaded_audio
from app.services.status_service import get_health, get_project_status

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["status"])
def health() -> HealthResponse:
    return get_health()


@router.get("/status", response_model=StatusResponse, tags=["status"])
def status() -> StatusResponse:
    return get_project_status()


@router.post("/audio/upload", response_model=AudioUploadResponse, tags=["audio"])
async def upload_audio(file: UploadFile | None = File(default=None)) -> AudioUploadResponse:
    """Module 02 endpoint — returns metadata only, no waveform data."""
    return await load_uploaded_audio(file)


@router.post("/audio/analyze", response_model=AudioAnalyzeResponse, tags=["audio"])
async def analyze_audio(file: UploadFile | None = File(default=None)) -> AudioAnalyzeResponse:
    """Module 03 endpoint — returns metadata plus decimated waveform data."""
    return await analyze_uploaded_audio(file)
