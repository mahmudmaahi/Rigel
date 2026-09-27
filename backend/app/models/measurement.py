from pydantic import BaseModel

class MeasurementRequest(BaseModel):
    samples: list[list[float]]
    sample_rate_hz: int

class LoudnessMeasurement(BaseModel):
    average_rms_dbfs: float
    peak_dbfs: float

class PitchMeasurement(BaseModel):
    average_pitch_hz: float
    min_pitch_hz: float
    max_pitch_hz: float
    voiced_percentage: float
    confidence: float

class RhythmMeasurement(BaseModel):
    bpm: float
    confidence: float
    reliable: bool

class MeasurementResponse(BaseModel):
    loudness: LoudnessMeasurement
    pitch: PitchMeasurement
    rhythm: RhythmMeasurement

class LiveMeasurementFrame(BaseModel):
    rms_dbfs: float
    peak_dbfs: float
    pitch_hz: float
    pitch_confidence: float
    voiced: bool
