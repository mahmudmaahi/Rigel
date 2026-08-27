from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final

import numpy as np
from scipy.io import wavfile

SUPPORTED_WAV_EXTENSIONS: Final[set[str]] = {".wav"}


@dataclass(frozen=True)
class LoadedAudio:
    sample_rate_hz: int
    samples: np.ndarray
    format: str
    bit_depth: int | None


class AudioDecodeError(ValueError):
    """Raised when audio bytes cannot be decoded as a supported audio format."""


def is_supported_audio_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_WAV_EXTENSIONS


def load_audio_bytes(file_bytes: bytes, filename: str) -> LoadedAudio:
    if not file_bytes:
        raise AudioDecodeError("The uploaded audio file is empty.")

    if not is_supported_audio_filename(filename):
        raise AudioDecodeError("Only WAV audio files are supported in this module.")

    try:
        sample_rate_hz, samples = wavfile.read(BytesIO(file_bytes))
    except Exception as exc:
        raise AudioDecodeError("The uploaded file could not be decoded as WAV audio.") from exc

    if sample_rate_hz <= 0:
        raise AudioDecodeError("The decoded audio file has an invalid sample rate.")

    if samples.size == 0:
        raise AudioDecodeError("The decoded audio file contains no samples.")

    bit_depth = _infer_bit_depth(samples.dtype)

    return LoadedAudio(
        sample_rate_hz=int(sample_rate_hz),
        samples=np.asarray(samples),
        format="wav",
        bit_depth=bit_depth,
    )


def _infer_bit_depth(dtype: np.dtype) -> int | None:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return dtype.itemsize * 8
    if np.issubdtype(dtype, np.floating):
        return dtype.itemsize * 8
    return None
