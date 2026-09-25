from fastapi import APIRouter, File, UploadFile, Form

from app.models.audio import (
    AudioAnalyzeResponse,
    AudioExportRequest,
    AudioFilterRequest,
    AudioFilterResponse,
    AudioFilterResponseRequest,
    AudioSpectrogramResponse,
    AudioSpectrumResponse,
    AudioUploadResponse,
)
from app.models.status import HealthResponse, StatusResponse
from app.services.audio_service import (
    analyze_uploaded_audio,
    compute_spectrogram_for_upload,
    compute_spectrum_for_upload,
    load_uploaded_audio,
)
from app.services.filter_service import (
    apply_filter_to_upload,
    get_filter_frequency_response,
)
from app.services.status_service import get_health, get_project_status
from app.services.export_service import export_samples_to_wav
from app.api.vad import router as vad_router
from app.models.voice_lab import VoiceProcessRequest, VoiceProcessResponse
from app.services.voice_lab_service import process_voice_lab

router = APIRouter()
router.include_router(vad_router, prefix="/vad", tags=["vad"])


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


@router.post("/audio/spectrum", response_model=AudioSpectrumResponse, tags=["audio"])
async def spectrum_audio(file: UploadFile | None = File(default=None)) -> AudioSpectrumResponse:
    """Module 04 endpoint — returns metadata plus one-sided magnitude spectrum (dBFS).

    The spectrum is computed over the full signal (N = all samples per channel),
    giving frequency resolution Δf = Fs / N.  The raw FFT result is reduced to
    at most 1024 display bins before being returned to the frontend.
    """
    return await compute_spectrum_for_upload(file)


@router.post("/audio/spectrogram", response_model=AudioSpectrogramResponse, tags=["audio"])
async def spectrogram_audio(
    file: UploadFile | None = File(default=None),
    window: str = Form("hann")
) -> AudioSpectrogramResponse:
    """Module 05 endpoint — returns metadata plus a display-ready STFT spectrogram.

    The STFT is computed with:
        frame_length = 2048 samples  (Δf = Fs / 2048)
        hop_length   = 512  samples  (75% overlap, Δt = 512 / Fs)
        window       = Hann

    The resulting magnitude matrix is downsampled to at most
    512 × 256 (time × frequency) before being returned to the frontend.
    """
    return await compute_spectrogram_for_upload(file, window=window)


@router.post("/export", tags=["export"])
async def export_audio(request: AudioExportRequest):
    """Generic endpoint to export processed floating-point samples to a downloadable WAV file."""
    return export_samples_to_wav(request.samples, request.sample_rate_hz)


@router.post("/audio/filter", tags=["audio", "filtering"])
async def filter_audio(
    file: UploadFile | None = File(default=None),
    filter_type: str = Form(...),
    family: str = Form("butterworth"),
    order: int = Form(...),
    cutoff_hz: float | None = Form(default=None),
    low_hz: float | None = Form(default=None),
    high_hz: float | None = Form(default=None),
    # IIR family-specific
    ripple_db: float = Form(default=1.0),
    attenuation_db: float = Form(default=40.0),
    # FIR-specific
    fir_window: str = Form(default="hamming"),
    transition_bandwidth_hz: float = Form(default=200.0),
    # Peaking EQ
    center_hz: float | None = Form(default=None),
    gain_db: float = Form(default=0.0),
    q_factor: float = Form(default=1.0),
):
    """Module 06 — Apply a digital filter to an uploaded WAV file.

    Supports all filter families (Butterworth, Chebyshev I/II, Elliptic,
    Bessel, FIR Window, Parks-McClellan) and filter types (lowpass, highpass,
    bandpass, bandstop, peaking EQ).

    Zero-phase forward-backward filtering is applied (sosfiltfilt / filtfilt).
    Appropriate for offline audio; NOT suitable for real-time filtering.
    """
    req = AudioFilterRequest(
        filter_type=filter_type,  # type: ignore[arg-type]
        family=family,  # type: ignore[arg-type]
        order=order,
        cutoff_hz=cutoff_hz,
        low_hz=low_hz,
        high_hz=high_hz,
        ripple_db=ripple_db,
        attenuation_db=attenuation_db,
        fir_window=fir_window,
        transition_bandwidth_hz=transition_bandwidth_hz,
        center_hz=center_hz,
        gain_db=gain_db,
        q_factor=q_factor,
    )
    return await apply_filter_to_upload(file, req)


@router.post("/audio/filter/response", response_model=AudioFilterResponse, tags=["filtering"])
async def filter_frequency_response(request: AudioFilterResponseRequest) -> AudioFilterResponse:
    """Module 06 endpoint — compute the theoretical frequency response of a filter.

    Does not require an audio file. Returns the filter magnitude response
    in dB so the frontend can visualise the passband, stopband, cutoff, and
    transition behaviour before applying the filter.
    """
    return await get_filter_frequency_response(request)


@router.post("/audio/denoise", tags=["audio", "denoise"])
async def denoise_audio(
    file: UploadFile | None = File(default=None),
    method: str = Form(...),
    alpha: float = Form(default=1.0),
    beta: float = Form(default=0.01),
    alpha_dd: float = Form(default=0.98),
    noise_alpha_s: float = Form(default=0.98),
    noise_bias: float = Form(default=1.5),
    g_min: float = Form(default=0.01),
    imcra_alpha_s: float = Form(default=0.86),
    imcra_alpha_d: float = Form(default=0.85),
):
    """Module 07 — Apply statistical noise removal to an uploaded WAV file.

    Supports Spectral Subtraction, Decision-Directed Wiener Filtering,
    Log-MMSE, and IMCRA + OM-LSA.
    """
    from app.models.denoise import AudioDenoiseRequest
    from app.services.denoise_service import apply_denoise_to_upload

    req = AudioDenoiseRequest(
        method=method,
        alpha=alpha,
        beta=beta,
        alpha_dd=alpha_dd,
        noise_alpha_s=noise_alpha_s,
        noise_bias=noise_bias,
        g_min=g_min,
        imcra_alpha_s=imcra_alpha_s,
        imcra_alpha_d=imcra_alpha_d,
    )
    return await apply_denoise_to_upload(file, req)


@router.post("/audio/voice/process", response_model=VoiceProcessResponse, tags=["voice-lab"])
def voice_process(request: VoiceProcessRequest) -> VoiceProcessResponse:
    """Module 09 — Apply the Voice Lab processing chain to audio samples.

    Canonical chain (non-reorderable):
        Input → Gain → Speed → TimeStretch → PitchShift → Effect → Output

    All processing begins from the original input samples.
    Never feeds previous processed output back into the chain.

    Operations with neutral parameters are bypassed:
        gain: mode=db, gain_value=0.0 (0 dB)
        speed: speed=1.0
        time_stretch: stretch=1.0
        pitch_shift: semitones=0.0
        effect: null

    If prevent_clipping=True and output peak > 1.0, peak normalization is
    applied as an explicit final step and reported in the response.
    The output is never silently normalized otherwise.
    """
    from fastapi import HTTPException
    try:
        return process_voice_lab(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice Lab DSP error: {exc}") from exc
