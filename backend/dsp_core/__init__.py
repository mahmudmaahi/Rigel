"""Framework-independent DSP package for Rigel.

Modules
-------
audio_loader
    Decodes WAV audio bytes into NumPy arrays.  Implemented in Module 02.
waveform
    Decimates audio samples into peak-envelope waveform data suitable for
    browser visualisation.  Implemented in Module 03.

This package has no dependency on FastAPI, Starlette, or any frontend code.
It can be reused by both the web API layer and a future desktop application.
"""

CORE_STATUS = "available"
