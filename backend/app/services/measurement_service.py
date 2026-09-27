import numpy as np

from app.models.measurement import (
    MeasurementRequest,
    MeasurementResponse,
    LoudnessMeasurement,
    PitchMeasurement,
    RhythmMeasurement,
)
from dsp_core.measurement import (
    measure_loudness,
    aggregate_pitch,
    estimate_bpm,
)

def _samples_to_numpy(samples: list[list[float]]) -> np.ndarray:
    arr = np.array(samples, dtype=np.float64)
    if arr.shape[0] == 1:
        return arr[0]
    return arr.T

def process_measurements(req: MeasurementRequest) -> MeasurementResponse:
    x = _samples_to_numpy(req.samples)
    
    loudness = measure_loudness(x)
    pitch = aggregate_pitch(x, req.sample_rate_hz)
    rhythm = estimate_bpm(x, req.sample_rate_hz)
    
    return MeasurementResponse(
        loudness=LoudnessMeasurement(
            average_rms_dbfs=loudness["rms_dbfs"],
            peak_dbfs=loudness["peak_dbfs"],
        ),
        pitch=PitchMeasurement(
            average_pitch_hz=pitch["average_pitch"],
            min_pitch_hz=pitch["min_pitch"],
            max_pitch_hz=pitch["max_pitch"],
            voiced_percentage=pitch["voiced_percentage"],
            confidence=pitch["confidence"],
        ),
        rhythm=RhythmMeasurement(
            bpm=rhythm["bpm"],
            confidence=rhythm["confidence"],
            reliable=rhythm["reliable"],
        )
    )
