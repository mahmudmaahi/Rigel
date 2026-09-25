import time
import numpy as np
from pydantic import BaseModel

from pydantic import BaseModel
from typing import Protocol, List

class VadSessionResult(BaseModel):
    timestamp_seconds: float
    state: str
    is_speech: bool
    activity_score: float
    frames_in_state: int
    speech_duration_frames: int
    compute_ms: float
    engine: str = "unknown"


from typing import Protocol, List

class VadBackend(Protocol):
    def process_chunk(self, chunk: np.ndarray) -> List[VadSessionResult]:
        ...



_SILERO_MODEL = None

def get_silero_model():
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        import torch
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
        _SILERO_MODEL = model
    return _SILERO_MODEL

class SileroVADBackend:
    def __init__(self, sample_rate_hz: int):
        self.source_sample_rate_hz = sample_rate_hz
        self.model_sample_rate = 16000
        
        import copy
        self.model = copy.deepcopy(get_silero_model())
        self.model.reset_states()
        self.threshold = 0.5
        
        # Silero expects 512 chunks at 16000Hz (32ms).
        # To get 32ms from the source, we need `source_sample_rate * 0.032` samples.
        self.source_chunk_size = int(self.source_sample_rate_hz * 0.032)
        self.pcm_buffer = np.array([], dtype=np.float32)
        
        self.frames_in_state = 0
        self.speech_duration_frames = 0
        self.current_state = "SILENCE"
        self.total_processed_samples = 0
        
    def process_chunk(self, chunk: np.ndarray) -> List[VadSessionResult]:
        import torch
        import scipy.signal
        
        self.pcm_buffer = np.concatenate([self.pcm_buffer, chunk])
        results = []
        
        while len(self.pcm_buffer) >= self.source_chunk_size:
            frame = self.pcm_buffer[:self.source_chunk_size]
            self.pcm_buffer = self.pcm_buffer[self.source_chunk_size:]
            
            t0 = time.perf_counter()
            
            if self.source_sample_rate_hz != 16000:
                # Resample 32ms frame to exactly 512 samples
                silero_frame = scipy.signal.resample(frame, 512).astype(np.float32)
            else:
                silero_frame = frame
                
            tensor_chunk = torch.from_numpy(silero_frame).unsqueeze(0)
            
            prob = self.model(tensor_chunk, 16000).item()
            t1 = time.perf_counter()
            
            is_speech = prob >= self.threshold
            
            new_state = "SPEECH" if is_speech else "SILENCE"
            if new_state == self.current_state:
                self.frames_in_state += 1
            else:
                self.frames_in_state = 1
                self.current_state = new_state
                
            if is_speech:
                self.speech_duration_frames += 1
            else:
                self.speech_duration_frames = 0
                
            self.total_processed_samples += self.source_chunk_size
            ts = self.total_processed_samples / self.source_sample_rate_hz
            
            compute_ms = (t1 - t0) * 1000.0
            
            results.append(VadSessionResult(
                timestamp_seconds=ts,
                state=self.current_state,
                is_speech=is_speech,
                activity_score=prob,
                frames_in_state=self.frames_in_state,
                speech_duration_frames=self.speech_duration_frames,
                compute_ms=compute_ms,
                engine="silero",
            ))
            
        return results

class TENVADBackend:
    def __init__(self, sample_rate_hz: int):
        import ten_vad
        self.source_sample_rate_hz = sample_rate_hz
        self.model_sample_rate = 16000
        
        # 10ms at 16000Hz is 160 samples
        self.hop_size = 160
        self.vad = ten_vad.TenVad(hop_size=self.hop_size, threshold=0.5)
        
        self.source_chunk_size = int(self.source_sample_rate_hz * (self.hop_size / 16000.0))
        self.pcm_buffer = np.array([], dtype=np.float32)
        
        self.frames_in_state = 0
        self.speech_duration_frames = 0
        self.current_state = "SILENCE"
        self.total_processed_samples = 0

    def process_chunk(self, chunk: np.ndarray) -> List[VadSessionResult]:
        import scipy.signal
        self.pcm_buffer = np.concatenate([self.pcm_buffer, chunk])
        results = []
        
        while len(self.pcm_buffer) >= self.source_chunk_size:
            frame = self.pcm_buffer[:self.source_chunk_size]
            self.pcm_buffer = self.pcm_buffer[self.source_chunk_size:]
            
            t0 = time.perf_counter()
            if self.source_sample_rate_hz != 16000:
                frame_16k = scipy.signal.resample(frame, self.hop_size).astype(np.float32)
            else:
                frame_16k = frame
            
            # Convert to int16
            # Multiply by 32767.0 and clip
            frame_int16 = np.clip(frame_16k * 32767.0, -32768, 32767).astype(np.int16)
            
            # Process via TEN VAD
            prob, state = self.vad.process(frame_int16)
            # state seems to be an int: -1 for non-speech, 0 for speech? Wait, earlier it returned 0 and -1.
            # In my test: Result (zeros): (-1.0, -1), Result (noise): (0.4147210717201233, 0)
            is_speech = prob >= 0.5
            
            t1 = time.perf_counter()
            compute_ms = (t1 - t0) * 1000.0
            
            new_state = "SPEECH" if is_speech else "SILENCE"
            if new_state == self.current_state:
                self.frames_in_state += 1
            else:
                self.current_state = new_state
                self.frames_in_state = 1
                
            if is_speech:
                self.speech_duration_frames += 1
            else:
                self.speech_duration_frames = 0
                
            self.total_processed_samples += self.hop_size
            timestamp_seconds = self.total_processed_samples / 16000.0
            
            # Since prob can be -1.0 for absolute zero/invalid frames, clamp it for UI presentation
            display_prob = max(0.0, float(prob))
            
            results.append(VadSessionResult(
                timestamp_seconds=timestamp_seconds,
                state=self.current_state,
                is_speech=is_speech,
                activity_score=display_prob,
                frames_in_state=self.frames_in_state,
                speech_duration_frames=self.speech_duration_frames,
                compute_ms=compute_ms,
                engine="ten_vad",
            ))
            
        return results

class VadSession:
    """
    A transport-level abstraction that owns a complete isolated instance of the VAD DSP pipeline.
    Responsible for receiving raw PCM chunks, chunking them into frames, and returning results.
    """
    def __init__(self, sample_rate_hz: int, backend_type: str = "silero"):
        if backend_type == "silero":
            self.backend: VadBackend = SileroVADBackend(sample_rate_hz)
        elif backend_type == "ten_vad":
            self.backend: VadBackend = TENVADBackend(sample_rate_hz)
        else:
            raise ValueError(f"Unknown VAD backend: {backend_type}")
            
    def process_chunk(self, chunk: np.ndarray) -> List[VadSessionResult]:
        return self.backend.process_chunk(chunk)

