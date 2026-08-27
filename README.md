# Rigel

Team Orion's Rigel project is an incremental intelligent audio platform. The current implementation is **Module 03 - Waveform Visualization**: a canvas-based interactive waveform viewer, a synchronized audio player, peak-preserving waveform decimation in the DSP core, and a new `/api/audio/analyze` endpoint that returns both metadata and waveform data.

Rigel does **not** yet implement FFT, spectrograms, filters, denoising, VAD, voice effects, ANC, desktop functionality, or AI. Those belong to later modules.

## Project Structure

```text
rigel/
|-- frontend/              # Next.js + TypeScript web application
|   |-- src/app/           # App Router pages and global styles
|   |-- src/components/    # Product UI and reusable components
|   |   |-- audio/         # AudioUploader, WaveformViewer, AudioPlayer
|   |   `-- ui/            # Shared UI primitives (Card, Badge)
|   `-- src/lib/           # Frontend API client and shared types
|-- backend/               # FastAPI API layer and reusable Python packages
|   |-- app/               # FastAPI application code
|   |   |-- api/           # Route handlers
|   |   |-- models/        # Pydantic request/response models
|   |   `-- services/      # Business logic (audio loading, waveform, status)
|   |-- dsp_core/          # Framework-independent audio/DSP package
|   |   |-- audio_loader.py  # WAV decoding → NumPy (Module 02)
|   |   `-- waveform.py      # Peak-envelope decimation (Module 03)
|   `-- tests/             # Backend tests
|-- AGENT_INSTRUCTIONS.md  # Governing project instructions
|-- README.md              # Current implementation state
`-- .gitignore
```

## Architecture

Rigel is being built so the Python audio/DSP layer can later be reused by both the web API and a future desktop application.

```text
Next.js Frontend
      |
      | HTTP multipart upload
      v
FastAPI Backend
      |
      v
Audio Service
      |
      v
dsp_core Audio Loader   +   dsp_core Waveform
      |                           |
      v                           v
NumPy samples          Decimated bins (min, max, time)
      |                           |
      +---------------------------+
                    |
                    v
           AudioAnalyzeResponse
        (metadata + waveform data)
```

The DSP core is framework-independent. It accepts NumPy arrays and returns plain Python dataclasses. FastAPI does not contain decoding or DSP logic directly.

The frontend uses the browser's local `File` object URL (`URL.createObjectURL`) for audio playback rather than a backend streaming endpoint. This keeps Module 03 simple and avoids introducing persistent storage.

## Implemented

### Module 01 - Project Skeleton

- Next.js frontend initialized with TypeScript, App Router, Tailwind CSS, ESLint, and Lucide icons.
- FastAPI backend initialized with application settings, CORS configuration, and versioned API routing.
- Minimal health/status API: `GET /api/health`, `GET /api/status`.
- Backend tests for health/status responses.
- Development configuration and environment examples.

### Module 02 - Audio Upload and Loading

- Product-facing Rigel landing/upload page with a prominent title and no public module/API/DSP diagnostic cards.
- Theme-ready black-and-white visual foundation using semantic tokens.
- Drag-and-drop or file-picker WAV upload in the frontend.
- Upload/loading/success/error UI states.
- FastAPI endpoint: `POST /api/audio/upload` (metadata only, no waveform).
- WAV validation and error handling for missing file, unsupported type, empty file, malformed WAV, decoding failure, and oversized upload.
- Audio loading through `backend/dsp_core/audio_loader.py`.
- NumPy sample representation on the backend.
- Structured metadata returned to the frontend.

### Module 03 - Waveform Visualization

- `backend/dsp_core/waveform.py` — framework-independent peak-preserving min/max decimation and time-axis generation.
- FastAPI endpoint: `POST /api/audio/analyze` — returns metadata plus decimated waveform data (replaces the primary upload flow; the old `/api/audio/upload` endpoint is preserved for backward compatibility).
- `frontend/src/components/audio/waveform-viewer.tsx` — canvas-based waveform component with:
  - Peak-envelope bars (min/max per bin) rendered on HTML5 Canvas.
  - Responsive sizing via ResizeObserver with device pixel ratio correction.
  - Time-axis tick marks with sensible auto-interval selection.
  - Animated playhead line synchronized with audio playback.
  - Click-to-seek: maps canvas X position to audio timestamp.
  - Stereo: two separate waveform rows (L/R), not collapsed to mono.
- `frontend/src/components/audio/audio-player.tsx` — HTML5 audio player component with:
  - Play/Pause, seek slider, current time / duration display, mute toggle.
  - `onTimeUpdate` callback to drive waveform playhead.
  - `seekTarget` prop for programmatic seeking from waveform clicks.
- Updated `frontend/src/components/audio/audio-uploader.tsx` — orchestrates the full flow: analyzes audio, creates local object URL for playback, manages shared `currentTime` and `seekTarget` state between viewer and player.
- Metadata badges in the analysis panel (format, sample rate, channels, duration, bit depth, dtype).
- Signal details section (expandable) showing full sample summary.
- 33 new backend unit tests across 6 test classes covering: mono/stereo, bin count clamping, time mapping, min/max peak preservation, edge cases, long audio, and API response structure.

## Waveform Decimation Design

### Bin timestamp convention

Each bin represents a chunk of consecutive samples. The `time_seconds` value in each `WaveformBin` is the **start time** of that chunk:

```
t_bin[k] = (k × bin_size) / sample_rate_hz
```

This convention is applied consistently when rendering time-axis ticks and when translating a waveform click back into an audio seek position.

### Downsampling method

Peak-envelope (min/max) decimation: for each bin the minimum and maximum amplitude among all samples in that chunk are recorded. This ensures transient peaks remain visible at any zoom level. Mean decimation or linear resampling would alias sharp transients away.

### Number of bins

Default: **1 500 bins**. Rationale:
- Sufficient resolution at typical canvas widths of 600–2 000 px (≈ 1 bin/pixel or small multiples).
- Keeps JSON payload manageable (≈ 36 KB for 1 500 float64 bins).
- Clamped to the actual number of samples for very short audio.

### Stereo strategy

Stereo audio is preserved channel-by-channel and rendered as two vertically stacked waveform rows (L, R). Collapsing to mono was deliberately avoided because:
- It hides stereo panning information.
- Phase cancellation artefacts can cause misleadingly small amplitudes in the mixed waveform.
- The L/R view gives a more truthful representation of the signal.

### Audio playback — object URL

The frontend creates a local `URL.createObjectURL(file)` from the browser's `File` object. This means audio plays back directly in the browser without requiring a backend streaming endpoint. Object URLs are revoked on unmount and when a new file is loaded to prevent memory leaks.

## Supported Audio Formats

Module 03 inherits Module 02's WAV-only support. The initial focus is uncompressed PCM-style WAV because it is directly useful for DSP education and can be decoded without adding large media-processing dependencies.

## API Contract

### `GET /api/health`

Returns basic service liveness.

```json
{
  "status": "ok",
  "service": "rigel-api",
  "version": "0.1.0"
}
```

### `GET /api/status`

Returns project/module state for developer tooling and tests. Not shown in the public product UI.

```json
{
  "project": "Rigel",
  "team": "Orion",
  "phase": "Phase 1 - Foundation",
  "module": "Module 03 - Waveform Visualization",
  "api": "online",
  "dsp_core": "available",
  "features": [
    "project skeleton",
    "frontend-backend communication",
    "health/status API",
    "black-and-white UI foundation",
    "WAV upload and metadata extraction",
    "waveform decimation and visualization",
    "audio playback"
  ]
}
```

### `POST /api/audio/upload` (Module 02 — preserved)

Returns metadata only. No waveform data.

### `POST /api/audio/analyze` (Module 03)

Receives a multipart form upload with field name `file`.

Example success response (abbreviated):

```json
{
  "status": "analyzed",
  "metadata": {
    "filename": "speech.wav",
    "format": "wav",
    "sample_rate_hz": 16000,
    "channels": 1,
    "samples_per_channel": 16000,
    "total_samples": 16000,
    "duration_seconds": 1.0,
    "bit_depth": 16,
    "sample_summary": {
      "array_shape": [16000],
      "dtype": "int16",
      "min_value": -4000,
      "max_value": 4000,
      "mean_value": 0.5,
      "representation": "NumPy ndarray; mono shape is [samples], multi-channel shape is [samples, channels]."
    }
  },
  "waveform": {
    "n_channels": 1,
    "n_bins": 1500,
    "duration_seconds": 1.0,
    "sample_rate_hz": 16000,
    "channels": [
      {
        "channel_index": 0,
        "bins": [
          { "time_seconds": 0.0, "min_amplitude": -12.0, "max_amplitude": 15.0 },
          { "time_seconds": 0.000667, "min_amplitude": -20.0, "max_amplitude": 18.0 }
        ]
      }
    ]
  }
}
```

The full sample array is not returned to the frontend. Samples remain on the backend for future DSP modules.

## Run The Backend

From the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend API docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Run The Frontend

In a second terminal, from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

The frontend calls the backend at `http://127.0.0.1:8000` by default. To override this, create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## IDE Import Resolution

Python dependencies are installed in the backend virtual environment:

```text
backend/.venv
```

If VS Code/Pylance shows red underlines such as `Import "fastapi" could not be resolved`, select this interpreter:

```text
D:\rigel\backend\.venv\Scripts\python.exe
```

The repository also includes `backend/pyrightconfig.json` to point Pyright-compatible tools at that virtual environment.

## Tests And Checks

Backend:

```powershell
cd backend
python -m pytest
```

Frontend:

```powershell
cd frontend
npm run lint
npm run build
npm audit --audit-level=high
```

Verification performed during Module 03:

- `backend`: `python -m pytest` passed with **40 tests** (7 Module 01/02 + 33 Module 03).
- `frontend`: `npm run lint` passed (exit code 0).
- `frontend`: `npm run build` passed (exit code 0, compiled in 37.0s, TypeScript passed in 2.5s).
- Manual smoke check: `POST /api/audio/analyze` with a generated WAV returned correct metadata and 1500 waveform bins.
- Manual smoke check: `POST /api/audio/upload` (Module 02 endpoint) still returns metadata only without waveform data.
- Backend tests continue to emit a Python 3.14 deprecation warning from FastAPI/Starlette internals; tests still pass.

## Module Status

- [x] Module 01 - Project Skeleton
- [x] Module 02 - Audio Upload and Loading
- [x] Module 03 - Waveform Visualization
- [ ] Module 04 - Fourier Analysis
- [ ] Module 05 - Spectrogram / STFT
- [ ] Module 06 - Digital Filters
- [ ] Module 07 - Noise Removal
- [ ] Module 08 - Voice Activity Detection
- [ ] Module 09 - Voice Tweaks
- [ ] Module 10 - Adaptive Noise Cancellation
- [ ] Module 11 - Desktop Application
- [ ] Module 12 - Virtual Microphone / Audio Routing
- [ ] Module 13 - Real-Time Noise Suppression
- [ ] Module 14 - Real-Time Voice Effects

## Important Decisions

- FastAPI remains the API/application layer.
- WAV decoding is isolated in `backend/dsp_core/audio_loader.py`.
- Waveform decimation is isolated in `backend/dsp_core/waveform.py`. Both DSP modules are framework-independent.
- Peak-envelope (min/max) decimation was chosen over mean-based resampling to preserve transient peaks.
- 1 500 bins was selected as the default, balancing visual resolution and payload size.
- Bin timestamps represent the **start time** of each bin chunk. This convention is consistent across the backend, `WaveformBin.time_seconds`, the canvas renderer, and the click-to-seek handler.
- Stereo audio is rendered as two independent L/R rows rather than mixed to mono.
- Audio playback uses `URL.createObjectURL(file)` on the already-uploaded browser `File` object. No backend streaming endpoint was introduced because the file is already in browser memory.
- The public UI continues to use no internal module numbers, backend status, DSP-core status, or development milestones.
- Semantic frontend theme tokens are used so a future dark mode can be added without rewriting component color classes.
- Module 02's `/api/audio/upload` endpoint is preserved for backward compatibility and testing. The primary product flow now uses `/api/audio/analyze`.

## Known Limitations

- Only WAV upload/loading is supported.
- Uploaded files are processed immediately and not persisted.
- Audio playback depends on the browser's WAV decoding capability (all modern browsers support WAV PCM).
- The waveform canvas reads CSS custom properties at draw time. If the canvas is inside a shadow DOM or non-standard CSS context, colour tokens may not resolve correctly.
- No FFT, spectrogram, filtering, denoising, VAD, ANC, voice effects, desktop app, or AI functionality exists yet.
- The frontend expects the backend to be running separately during development.

## Next Module

**Module 04 - Fourier Analysis** is next. It should add DFT/FFT computation in the DSP core, a frequency-domain endpoint, and a frequency spectrum display in the frontend. Do not begin Module 04 until explicitly instructed.
