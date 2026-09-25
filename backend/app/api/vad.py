from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import numpy as np
import logging

from app.services.vad_service import VadSession

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/stream")
async def vad_stream(websocket: WebSocket):
    await websocket.accept()
    session = None
    chunk_count = 0
    
    try:
        while True:
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                break
                
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("event") == "start":
                        sample_rate = data.get("sampleRate")
                        channels = data.get("channels")
                        fmt = data.get("format")
                        
                        if not sample_rate or type(sample_rate) not in (int, float) or sample_rate <= 0:
                            await websocket.send_json({"error": "Invalid sampleRate"})
                            continue
                        if channels != 1:
                            await websocket.send_json({"error": "Only 1 channel (mono) is supported"})
                            continue
                        if fmt != "float32":
                            await websocket.send_json({"error": "Format must be float32"})
                            continue
                            
                        engine = data.get("engine", "silero")
                        
                        print(f"VAD START REQUEST\nrequested_engine = {engine}\nsource_sample_rate = {sample_rate}")
                        
                        # Sample rate invariant strictly enforced here
                        try:
                            session = VadSession(sample_rate_hz=int(sample_rate), backend_type=engine)
                            print(f"VAD SESSION CREATED\nactual_engine = {engine}\nbackend_class = {session.backend.__class__.__name__}")
                            chunk_count = 0
                            await websocket.send_json({"event": "started", "engine": engine})
                        except Exception as e:
                            print(f"ERROR CREATING VAD SESSION: {repr(e)}")
                            import traceback
                            traceback.print_exc()
                            await websocket.send_json({"error": str(e)})
                        
                    elif data.get("event") == "stop":
                        session = None
                        await websocket.send_json({"event": "stopped"})
                        
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON payload"})
                    
            elif "bytes" in message:
                if session is None:
                    await websocket.send_json({"error": "Session not started"})
                    continue
                    
                chunk_bytes = message["bytes"]
                try:
                    # Explicit Float32 PCM conversion
                    chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
                    chunk_count += 1
                    
                    if chunk_count == 1:
                        print(f"FIRST CHUNK\nengine = {session.backend.__class__.__name__}\nsamples_received = {len(chunk)}\nsource_sample_rate = {session.backend.source_sample_rate_hz if hasattr(session.backend, 'source_sample_rate_hz') else session.backend.sample_rate_hz}")
                        
                    results = session.process_chunk(chunk)
                    for i, r in enumerate(results):
                        json_resp = r.model_dump()
                        if chunk_count == 1 and i == 0:
                            print(f"FIRST BACKEND RESULT:\n{json.dumps(json_resp, indent=2)}")
                        await websocket.send_json(json_resp)
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    await websocket.send_json({"error": "Failed to process audio chunk"})
                    
    except WebSocketDisconnect:
        session = None
