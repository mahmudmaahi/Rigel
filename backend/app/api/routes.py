from fastapi import APIRouter, File, UploadFile, Form, Response, HTTPException

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

@router.post("/audio/image/encode", tags=["audio", "image"])
async def encode_audio_image(file: UploadFile = File(...)):
    """Encode an uploaded WAV audio file into a Rigel Data Image (PNG)."""
    from app.services.audio_service import _read_and_validate_upload
    from app.services.audio_image_service import encode_audio_to_image
    from dsp_core.audio_loader import load_audio_bytes, AudioDecodeError
    import numpy as np

    try:
        file_bytes, filename, _ = await _read_and_validate_upload(file)
        loaded = load_audio_bytes(file_bytes, filename)
    except AudioDecodeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal trace error 1: " + str(e))
        
    if loaded.samples.dtype != np.int16:
        # If float, we could scale, but the spec says: "Do NOT silently convert arbitrary float WAVs... if that would violate exact sample recovery. Reject unsupported WAV formats cleanly."
        # However, scipy wavfile.read will return int16 if the original file is 16-bit PCM.
        # But `_infer_bit_depth` returns 16 if int16.
        if loaded.bit_depth != 16 or loaded.samples.dtype != np.int16:
            raise HTTPException(
                status_code=400, 
                detail="Only 16-bit PCM WAV files are supported for exact sample encoding."
            )

    try:
        res = encode_audio_to_image(loaded.samples, loaded.sample_rate_hz)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=res.png_bytes,
        media_type="image/png",
        headers={
            "X-Rigel-Image-Dim": str(res.image_dim),
            "X-Rigel-Payload-Bytes": str(res.payload_bytes_len),
            "X-Rigel-PCM-Bytes": str(res.original_pcm_bytes),
            "X-Rigel-FLAC-Bytes": str(res.flac_bytes_len),
            "X-Rigel-Compression-Ratio": f"{res.compression_ratio:.4f}"
        }
    )

@router.post("/audio/image/decode", tags=["audio", "image"])
async def decode_audio_image(file: UploadFile = File(...)):
    """Decode a Rigel Data Image (PNG) back into a WAV audio file."""
    from app.services.audio_image_service import decode_image_to_audio
    from dsp_core.audio_codec import AudioCodecError
    from dsp_core.payload_protocol import ProtocolError
    from dsp_core.image_codec import ImageCodecError
    from app.services.export_service import export_samples_to_wav
    
    file_bytes = await file.read()
    
    try:
        res = decode_image_to_audio(file_bytes)
    except (AudioCodecError, ProtocolError, ImageCodecError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    import io
    from scipy.io import wavfile
    
    buffer = io.BytesIO()
    wavfile.write(buffer, res.sample_rate_hz, res.samples)
    wav_bytes = buffer.getvalue()
    
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Rigel-Protocol": "RGL1",
            "X-Rigel-Image-Dim": str(res.image_dim),
            "X-Rigel-Payload-Bytes": str(res.payload_bytes_len),
            "X-Rigel-Sample-Rate": str(res.sample_rate_hz),
            "X-Rigel-Channels": str(1 if res.samples.ndim == 1 else res.samples.shape[1]),
            "X-Rigel-Sample-Count": str(len(res.samples))
        }
    )



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

    Chain (deterministic, sequential):
        Effect 1 → Effect 2 → Effect 3 → ...

    All processing begins from the original input samples.
    Never feeds previous processed output back into the chain.

    Operations with enabled=False are bypassed.

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


from app.models.measurement import MeasurementRequest, MeasurementResponse, LiveMeasurementFrame
from app.services.measurement_service import process_measurements
import numpy as np

@router.post("/audio/measure", response_model=MeasurementResponse, tags=["measurement"])
def measure_audio(request: MeasurementRequest) -> MeasurementResponse:
    """Analyze uploaded audio to measure Loudness, Pitch, and Rhythm."""
    from fastapi import HTTPException
    try:
        return process_measurements(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Measurement DSP error: {exc}") from exc


from fastapi import WebSocket, WebSocketDisconnect
import json

@router.websocket("/audio/measure/stream")
async def measure_audio_stream(websocket: WebSocket):
    """
    Real-time measurement endpoint.

    Protocol:
        1. Client sends a JSON handshake first: {"event": "start", "sampleRate": <int>}.
           Server replies {"event": "started"} once negotiated.
        2. All subsequent client messages are raw binary Float32 PCM frames.
           Server replies with a JSON-stringified LiveMeasurementFrame per frame,
           computed using the sample rate negotiated in step 1.
    """
    from dsp_core.measurement import measure_loudness, estimate_pitch_yin_frame

    await websocket.accept()
    sample_rate: int | None = None

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON payload"})
                    continue

                if data.get("event") == "start":
                    sr = data.get("sampleRate")
                    if not sr or type(sr) not in (int, float) or sr <= 0:
                        await websocket.send_json({"error": "Invalid sampleRate"})
                        continue
                    sample_rate = int(sr)
                    await websocket.send_json({"event": "started"})
                continue

            elif "bytes" in message:
                if sample_rate is None:
                    await websocket.send_json({"error": "Session not started"})
                    continue

                frame = np.frombuffer(message["bytes"], dtype=np.float32)

                loudness = measure_loudness(frame)
                pitch = estimate_pitch_yin_frame(frame, sample_rate)

                response = LiveMeasurementFrame(
                    rms_dbfs=loudness["rms_dbfs"],
                    peak_dbfs=loudness["peak_dbfs"],
                    pitch_hz=pitch["frequency_hz"],
                    pitch_confidence=pitch["confidence"],
                    voiced=pitch["voiced"]
                )

                await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
