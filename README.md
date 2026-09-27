# Rigel

Team Orion's Rigel project is an incremental intelligent audio platform. The current implementation is **Module 07 — Noise Removal**: Classical statistical offline denoising. Features four progressive approaches: Improved Spectral Subtraction, Decision-Directed Wiener Filtering, Log-MMSE, and IMCRA + OM-LSA. Built upon a unified STFT analysis/synthesis framework in the new `dsp_core/denoise.py` module.

Rigel does **not** yet implement voice effects, ANC, desktop functionality, or AI. Those belong to later modules.

## Project Structure

```text
rigel/
|-- frontend/              # Next.js + TypeScript web application
|   |-- src/app/           # App Router pages and global styles
|   |-- src/components/    # Product UI and reusable components
|   |   |-- audio/         # AudioUploader, WaveformViewer, SpectrumViewer, SpectrogramViewer, AudioPlayer
|   |   `-- ui/            # Shared UI primitives (Card, Badge)
|   `-- src/lib/           # Frontend API client and shared types
|-- backend/               # FastAPI API layer and reusable Python packages
|   |-- app/               # FastAPI application code
|   |   |-- api/           # Route handlers
|   |   |-- models/        # Pydantic request/response models
|   |   `-- services/      # Business logic (audio loading, waveform, spectrum, spectrogram, status)
|   |-- dsp_core/          # Framework-independent audio/DSP package
|   |   |-- audio_loader.py  # WAV decoding → NumPy (Module 02)
|   |   |-- waveform.py      # Peak-envelope decimation (Module 03)
|   |   |-- spectrum.py      # FFT magnitude spectrum in dBFS (Module 04)
|   |   `-- stft.py          # STFT spectrogram with Hann windowing (Module 05)
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
dsp_core Loader  +  dsp_core Waveform  +  dsp_core Spectrum  +  dsp_core STFT  +  dsp_core Filtering
      |                    |                    |                      |                      |
      v                    v                    v                      v                      v
NumPy samples     Decimated bins       One-sided FFT mag.    STFT magnitude matrix      Filtered NumPy
                  (min, max, time)     (dBFS, 1024 bins)     (dBFS, 512×256 display)    (WAV download)
      |                    |                    |                      |                      |
      +--------------------+--------------------+----------------------+----------------------+
                                         |
                                         v
         AudioAnalyzeResponse / AudioSpectrumResponse / AudioSpectrogramResponse / WAV file
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
  - Generous 140px channel height with proper clipping-safe playhead handle.
  - Interactive hover scrubber line and floating timestamp tooltip for precise audio inspection.
  - Responsive sizing via ResizeObserver with device pixel ratio correction.
  - Time-axis tick marks with sensible auto-interval selection.
  - Animated playhead line synchronized with audio playback.
  - Click-to-seek: maps canvas X position to audio timestamp.
  - Stereo: two separate waveform rows (L/R), not collapsed to mono.
- `frontend/src/components/audio/audio-player.tsx` — HTML5 audio player component with:
  - Play/Pause, interactive seek slider with visible hover thumb, current time / duration display, mute toggle.
  - `onTimeUpdate` callback to drive waveform playhead.
  - `seekTarget` prop for programmatic seeking from waveform clicks.
- Updated `frontend/src/components/audio/audio-uploader.tsx` — orchestrates the full flow: analyzes audio, creates local object URL for playback, manages shared `currentTime` and `seekTarget` state between viewer and player. Product-focused copy throughout ("Audio Input", "Audio Workspace").
- UI Refinement:
  - **Comprehensive interactive visual system** with signal-inspired motion design throughout every element.
  - **Precision technical aesthetic**: Corner alignment marks, coordinate badges, calibration ticks, channel labels (`CH_L`, `CH_R`, `CH_01`), amplitude scale markers (`+1.0`, `0.0`, `-1.0`), and technical grid crosshairs.
  - **Living signal background**: Canvas-based ambient oscilloscope traces with mouse-proximity modulation and technical grid markers (respects `prefers-reduced-motion`).
  - **Hero section**: Dynamic waveform backdrop that reveals on hover, coordinated staggered reveals, live timestamp telemetry, and precision coordinate badges.
  - **Interactive audio input terminal**: Precision instrument styling with corner brackets, animated idle waveform inside dropzone, smooth drag-and-drop feedback, enhanced button with sliding highlight effect.
  - **Signal Matrix workspace**: High-end laboratory aesthetic with live telemetry bar, precision metadata badges with divider lines and hover states, enhanced waveform canvas with amplitude scales and channel labels, refined playhead with subtle glow effect.
  - **Micro-interactions**: Tactile button scaling, icon translations, hover shadow elevations, smooth state transitions across all interactive elements.
  - **Technical typography**: Monospace coordinate markers, uppercase tracking, tabular numerals, and precision formatting throughout.
  - **Refined footer**: Corner alignment marks, animated signal accent line, lift-on-hover social icons with shadow.
  - All animations respect `prefers-reduced-motion` for accessibility.
  - Clean metadata badges in the analysis panel (format, sample rate, channels, duration, bit depth, dtype).
  - Collapsible "Discrete-Time Signal Matrix" inspection chamber with hover states.
- Metadata badges in the analysis panel (format, sample rate, channels, duration, bit depth, dtype).
- Signal details section (expandable) showing full sample summary and discrete-time signal context.
- `frontend/src/components/layout/footer.tsx` — modern footer component with Rigel branding, Team Orion identity, precision corner marks, animated accent line, and interactive social links.
- `frontend/src/components/visual/signal-background.tsx` — full-page interactive signal background with oscilloscope traces, technical grid crosshairs, and subtle mouse-proximity modulation.
- `frontend/src/components/visual/hero-signal.tsx` — hero section signal visualization with moving waveform and time markers.
- `frontend/src/components/visual/dropzone-idle-waveform.tsx` — animated idle oscilloscope waveform inside the dropzone before file upload.
- 33 new backend unit tests across 6 test classes covering: mono/stereo, bin count clamping, time mapping, min/max peak preservation, edge cases, long audio, and API response structure.

### Module 04 — Fourier Analysis

- `backend/dsp_core/spectrum.py` — framework-independent one-sided FFT spectrum with:
  - `normalize_samples()` — PCM integer/float → float32 amplitude in [-1, 1] for consistent dBFS reference.
  - `compute_dft_naive()` — O(N²) reference implementation of the DFT equation `X[k] = Σ x[n]·exp(-j2πkn/N)`. Educational only; not called in production.
  - `_compute_one_sided_magnitude()` — O(N log N) FFT via `np.fft.rfft` with correct one-sided amplitude scaling (DC and Nyquist not doubled; interior bins doubled).
  - `_amplitudes_to_dbfs()` — converts linear amplitude to dBFS: `20·log10(A[k] + ε)`.
  - `_downsample_spectrum()` — reduces raw FFT bins to ≤1024 display bins by peak-preserving linear binning (visualization step, not DSP mathematics).
  - `compute_spectrum()` — main entry point: normalize → FFT → one-sided scaling → dBFS → display reduction.
- FastAPI endpoint: `POST /api/audio/spectrum` — returns metadata plus display-ready magnitude spectrum.
- `frontend/src/components/audio/spectrum-viewer.tsx` — canvas-based spectrum component with linear frequency X axis (0 Hz → Nyquist), dBFS Y axis (0 dBFS top, -90 bottom), grid lines at -20/-40/-60/-80 dBFS, filled indigo area chart, per-channel rows.
- `frontend/src/lib/api.ts` — added `SpectrumBin`, `ChannelSpectrum`, `SpectrumData`, `AudioSpectrumResponse` types and `fetchSpectrum()` function.
- 42 new backend tests in `tests/test_spectrum.py` covering normalization, naive DFT, FFT one-sided scaling, dBFS, display reduction, stereo, and API response structure.

### Module 05 — Spectrogram / STFT

- `backend/dsp_core/stft.py` — framework-independent STFT spectrogram:
  - `hann_window(L)` — Hann window `w[n] = 0.5·(1 − cos(2πn/(L−1)))` to reduce spectral leakage at frame boundaries.
  - `_stft_channel()` — extracts overlapping frames, applies Hann window, calls `np.fft.rfft`, normalizes magnitude by L.
  - `_magnitudes_to_dbfs()` — 2-D `20·log10(magnitude + ε)` conversion.
  - `_downsample_time()` / `_downsample_freq()` — peak-preserving downsampling to at most 512 time columns × 256 frequency rows.
  - `_build_time_axis()` / `_build_freq_axis()` — compute centre time/frequency for each display group.
  - `compute_spectrogram()` — main entry point: normalize → frame → Hann window → rfft → dBFS → downsample → `SpectrogramData`.
  - Default parameters: `frame_length=2048` (Δf ≈ 21.5 Hz at 44.1 kHz), `hop_length=512` (75% overlap, Δt ≈ 11.6 ms at 44.1 kHz).
- FastAPI endpoint: `POST /api/audio/spectrogram` — returns metadata plus display-ready 2-D magnitude matrix.
- `frontend/src/components/audio/spectrogram-viewer.tsx` — canvas-based spectrogram heat-map with:
  - Perceptually-ordered colour gradient: deep navy → indigo → purple → cyan → amber → white.
  - X axis: time (seconds). Y axis: frequency (Hz), low-freq at bottom.
  - Subtle time and frequency grid overlay.
  - Per-channel rows matching WaveformViewer and SpectrumViewer panel design.
- `frontend/src/lib/api.ts` — added `SpectrogramChannel`, `SpectrogramData`, `AudioSpectrogramResponse` types and `fetchSpectrogram()` function.
- `frontend/src/components/audio/audio-uploader.tsx` — all three requests (`analyze`, `spectrum`, `spectrogram`) now run in parallel via `Promise.allSettled`. `SpectrogramViewer` renders below the `SpectrumViewer` in the Analysis Panel.
- 63 new backend tests in `tests/test_stft.py` covering: Hann window properties, normalization, STFT shape/dtype, silence, 1 kHz sine peak-bin detection, short-signal padding, dBFS conversion, time/frequency downsampling (peak-preserving), axis builders, and full integration (mono/stereo, metadata, display caps, error handling, multiple sample rates).
- **Total backend tests: 145 passing (82 pre-existing + 63 new)**.

### Module 07 — Noise Removal (COMPLETE)

- `backend/dsp_core/denoise.py` — comprehensive frequency-domain statistical denoising:
  - **Shared Infrastructure**: Unified STFT/ISTFT architecture with robust energy-based offline noise initialization to eliminate cold-start speech suppression.
  - **Improved Spectral Subtraction**: Power subtraction with oversubtraction factors and spectral flooring.
  - **Decision-Directed Wiener Filtering**: Ephraim-Malah style a-priori SNR tracking for reduced musical noise.
  - **Log-MMSE**: Minimum Mean-Square Error Log-Spectral Amplitude estimator for optimal speech envelope preservation.
  - **IMCRA + OM-LSA**: Advanced dual-iteration noise tracking with speech-presence probability (SPP) controlled geometric gain modification.
- **Conceptual Progression**: The module advances structurally from raw frequency-domain attenuation -> statistical estimation -> decision-directed enhancement -> advanced speech-presence-controlled enhancement.
- **Rigel's Current Approach**: Rigel explicitly uses classical statistical DSP (separating noise tracking from gain estimation) rather than deep learning. It does not claim to be a perfect universal denoiser; it is an educational laboratory for observing the fundamental mathematical behaviors and limits of traditional algorithms.
- `frontend/src/app/playground/enhancement/page.tsx` — Full integration of the noise removal algorithms into the UI.
  - Dynamic parameter controls tailored to each method.
  - Real-time synchronized playback and visual comparison between the original signal and the processed output.
- Extensive backend testing (486 total tests) covering edge cases, initialization behaviors, mathematical validity, and structural limits of the estimators.

### Module 08 — Voice Activity Detection

- **Status:** Final integration validation.
- **Architecture:** Real-time VAD processing via a WebSocket session that connects an audio source (Microphone or Shared Audio) to a selected VAD engine.
- **Engines:** Integrates **Silero** (neural) and **TEN VAD** (classical/hybrid). These are external libraries; Rigel provides the orchestration, session management, and UI.
- **State Separation:** Audio/Source state (what is playing) is architecturally separated from VAD state (which engine is processing). Engine hot-swapping is supported without full page reloads or dropping the audio context.
- **Engineering Contributions:** Common engine interface, float32 chunk adaptation, WebSocket transport, sample-rate adjustment, backend-confirmed engine identity, and real-time visualization.

### Module 06 — Audio Filtering

- `backend/dsp_core/filtering.py` — comprehensive frequency-selective filtering implementation:
  - **IIR Families**: Butterworth, Chebyshev I, Chebyshev II, Elliptic, Bessel. Always returns Second-Order Sections (SOS) for numerical stability up to N=20.
  - **FIR Families**: Window-method FIR and Parks-McClellan equiripple FIR.
  - **Filter Types**: Low-pass, high-pass, band-pass, band-stop, and Peaking EQ (biquad).
  - **Zero-Phase Offline Processing**: Uses `sosfiltfilt` (IIR) and `filtfilt` (FIR) to prevent phase shift.
  - **Theoretical Frequency Response**: Calculates magnitude response using `freqz`/`sosfreqz` for display in the frontend.
- FastAPI endpoints:
  - `POST /api/audio/filter/design` — validates parameters and returns theoretical magnitude response.
  - `POST /api/audio/filter/apply` — applies the zero-phase filter to the audio and returns a WAV file for playback.
- `frontend/src/app/playground/filtering/page.tsx` — dedicated filter workspace with:
  - Custom UI controls: `NumInput` with chevron steppers, `FilterTypeGrid`, `PillGroup` for design families.
  - Live theoretical frequency response chart using Recharts.
  - Dynamic parameter exposure (e.g. hiding ripple/attenuation if Butterworth is selected).
  - Toast notifications and "Play Filtered" / "Download Filtered WAV" audio integration.
- Extensive backend tests covering zero-phase outputs, FIR tap counts, Nyquist validations, error raising, parameter combinations, and peaking EQ constraints.


### Module 09 — Voice Laboratory (COMPLETE)

- `backend/dsp_core/time_scale.py` — Phase vocoder pitch-preserving time-stretch (Speed/Stretch) and standard raw resampling.
- `backend/dsp_core/voice_gain.py` — Multi-mode gain and normalization (dB, linear, peak, RMS).
- `backend/dsp_core/pitch_shift.py` — High-quality pitch shifting combining phase vocoder time-stretch and raw resampling (`stretch * r`, then `speed * r`).
- `backend/dsp_core/effects.py` — Classical effects (Delay, Chorus, Soft Distortion, Schroeder Reverb) with strict NaN/Inf rejection.
- FastAPI endpoint: `POST /api/audio/voice/process` — single orchestration endpoint supporting a deterministic DSP chain (Gain → Speed → Time Stretch → Pitch Shift → Effect → Output) with strict clipping risk tracking and optional normalization.
- `frontend/src/app/playground/voice-lab/page.tsx` — Transformation Observatory UI:
  - Comprehensive controls with mathematical equations shown for the active operations.
  - Dual A/B panel comparing the original audio waveform and spectrum with the processed output.
  - Independent audio players and detailed output metrics (duration, peak, RMS).

**Live Voice Measurement** (part of Voice Laboratory, not a separate module):

- `backend/dsp_core/measurement.py` — real-time and offline audio measurement:
  - **RMS / Peak loudness** — frame-level energy and peak detection in dBFS.
  - **YIN Pitch Algorithm** — classical autocorrelation-based fundamental frequency estimator producing pitch_hz, voiced/unvoiced flag, and confidence.
  - **BPM / Rhythm estimation** — envelope autocorrelation over a decimated onset envelope.
- **WebSocket endpoint:** `WS /api/audio/measure/stream` — the client sends a JSON handshake (`{"event": "start", "sampleRate": ...}`) negotiating the real sample rate, then raw Float32 PCM binary frames; the server returns JSON `LiveMeasurementFrame` (rms_dbfs, peak_dbfs, pitch_hz, pitch_confidence, voiced) per chunk, computed at the negotiated rate.
- **Offline endpoint:** `POST /api/audio/measure` — full-file measurement returning nested `MeasurementResponse` (loudness, pitch, rhythm).
- **Frontend — Live Voice Measurements panel** (embedded in the Voice Lab page):
  - **Source modes:** Microphone (AudioWorklet Float32 PCM streaming, promoted to the Playground's current/shared audio once recording stops) and Shared Audio (chunked AudioBuffer playback sync).
  - **SVG timeline graph:** continuous RMS line chart synced to audio playback timeline (up to 60 s for shared audio; rolling 10 s window for microphone). Red dots mark transient clipping peaks (> -1 dBFS). Cyan ticks at the bottom indicate voiced frames.
  - **Session summary overlay:** once measurement completes (audio finishes, recording stops, or an effect chain is processed), shows Peak dBFS, RMS dBFS, BPM, Avg Pitch, Pitch Range, and Voiced % in a clean card grid — BPM is only shown when the backend judges it statistically reliable; unreliable rhythm (typical for ordinary speech, which has no steady beat) is shown as "---" rather than a fabricated number.

### Module 10 — Audio to Image Encoder & Decoder (COMPLETE)

- `backend/dsp_core/payload_protocol.py` — the **RGL1** binary protocol: packs a losslessly-compressed (FLAC) audio payload plus its original sample rate/channel metadata into a length-prefixed, CRC32-checksummed binary blob (`pack_payload`), and validates/unpacks it back (`unpack_payload`).
- `backend/dsp_core/image_codec.py` — embeds the RGL1 payload bytes into the pixel data of a lossless PNG "data image" (encode), and extracts them back out (decode).
- FastAPI endpoints: `POST /api/audio/image/encode` (audio → PNG data image, returns image bytes with an `X-Rigel-Protocol: RGL1` header) and `POST /api/audio/image/decode` (PNG data image → recovered audio, CRC32-validated).
- `frontend/src/app/playground/image/{encode,decode}/page.tsx` — the **Audio ↔ Image** workspace: an Encode tab (Audio → Data Image) and a Decode tab (Data Image → Audio), each showing payload/compression statistics and validating the round trip.

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

### `POST /api/audio/spectrum` (Module 04)

Receives a multipart form upload with field name `file`.

Example success response (abbreviated):

```json
{
  "status": "spectrum_computed",
  "metadata": { "...same as /analyze..." },
  "spectrum": {
    "n_channels": 1,
    "n_fft_full": 16000,
    "n_display_bins": 1024,
    "frequency_resolution_hz": 1.0,
    "nyquist_hz": 8000.0,
    "duration_seconds": 1.0,
    "sample_rate_hz": 16000,
    "channels": [
      {
        "channel_index": 0,
        "bins": [
          { "frequency_hz": 3.906, "magnitude_dbfs": -72.1 },
          { "frequency_hz": 11.719, "magnitude_dbfs": -68.4 }
        ]
      }
    ]
  }
}
```

**`n_fft_full`** is the full signal length N, giving frequency resolution `Δf = Fs / N`. The raw FFT has `N//2 + 1` one-sided bins; these are display-reduced to at most 1024 `n_display_bins` before being sent to the frontend. `magnitude_dbfs` is dBFS (0 dBFS = full-scale amplitude); values are negative for real-world signals.


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

Verification performed during Module 04:

- `backend`: `python -m pytest` passed with **82 tests** (40 Module 01-03 + 42 Module 04).
- `frontend`: `npm run lint` passed (exit code 0).
- `frontend`: `npm run build` passed (exit code 0, TypeScript passed).
- Manual smoke check: `POST /api/audio/spectrum` with a generated 440 Hz WAV returned correct metadata and 1024 display bins, with peak near 440 Hz.
- Manual smoke check: `POST /api/audio/analyze` (Module 03 endpoint) still returns metadata and waveform data correctly.
- Long audio structural test (1,469,952 samples) confirmed ≤ 1024 display bins returned.

## Module Status

- [x] Module 01 - Project Skeleton
- [x] Module 02 - Audio Upload and Loading
- [x] Module 03 - Waveform Visualization
- [x] Module 04 - Fourier Analysis
- [x] Module 05 - Spectrogram / STFT
- [x] Module 06 - Digital Filters
- [x] Module 07 - Noise Removal
- [x] Module 08 - Voice Activity Detection
- [x] Module 09 - Voice Laboratory
- [x] Module 10 - Audio to Image Encoder

## DSP Concepts Introduced in Module 04

**DFT (Discrete Fourier Transform)**

Transforms a discrete-time signal `x[n]` into the frequency domain:

```
X[k] = Σ_{n=0}^{N-1}  x[n] · exp(-j 2πkn/N)
```

Produces N complex coefficients. Each coefficient `X[k]` corresponds to frequency `f[k] = k·Fs/N`.

**FFT (Fast Fourier Transform)**

The FFT computes the DFT in O(N log N) instead of O(N²). The result is mathematically identical. Rigel uses `np.fft.rfft` which exploits real-signal symmetry to return only the positive-frequency (one-sided) coefficients.

**Frequency Bins and Resolution**

The DFT of a length-N signal at sample rate Fs produces N frequency bins. Each bin k corresponds to:
```
f[k] = k · Fs / N      Hz
```
Frequency resolution: `Δf = Fs / N`. Longer signals give finer resolution.

**One-Sided Spectrum**

For a real-valued signal the DFT is conjugate-symmetric. Only the positive-frequency bins (k = 0 … N/2) carry unique information. Interior bins are multiplied by 2 to preserve the correct amplitude after discarding the mirror bins.

**Magnitude and dBFS**

The magnitude of bin k is `|X[k]|`. After one-sided amplitude normalisation, the result is `A[k]`. Expressed in dBFS (decibels relative to digital full scale):
```
magnitude_dBFS = 20 · log10(A[k] + ε)
```
0 dBFS = full-scale amplitude. This is NOT acoustic sound-pressure dB.

**Display Reduction**

Long audio can produce hundreds of thousands of raw FFT bins. The display spectrum is reduced to at most 1024 equal-width bins, keeping the peak dBFS in each bin (peak-preserving, analogous to Module 03 waveform decimation).

## Important Decisions

- FastAPI remains the API/application layer.
- WAV decoding is isolated in `backend/dsp_core/audio_loader.py`.
- Waveform decimation is isolated in `backend/dsp_core/waveform.py`.
- Spectrum computation is isolated in `backend/dsp_core/spectrum.py`.
- STFT spectrogram computation is isolated in `backend/dsp_core/stft.py`.
- Filter design and processing is isolated in `backend/dsp_core/filtering.py`. All DSP modules are framework-independent.
- Peak-envelope (min/max) decimation for waveforms; peak-preserving linear binning for spectrum display reduction; peak-preserving group-max for spectrogram time/frequency downsampling.
- Default STFT parameters: `frame_length=2048` (Δf ≈ 21.5 Hz at 44.1 kHz), `hop_length=512` (75% overlap).
- Hann windowing was chosen (rather than rectangular) to suppress spectral leakage at frame boundaries.
- The spectrogram magnitude is normalized per-frame by `L` (not one-sided doubled), which is the correct convention for visualization — the goal is relative energy distribution, not amplitude matching.
- `compute_dft_naive()` implements the DFT equation with two explicit loops for education. It is tested but never called in production.
- Samples are normalized to float32 amplitude before FFT/STFT so spectra have a consistent dBFS reference (0 dBFS = full scale).
- Audio playback uses `URL.createObjectURL(file)` on the already-uploaded browser `File` object. No backend streaming endpoint was introduced.
- The public UI continues to use no internal module numbers, backend status, DSP-core status, or development milestones.

## Known Limitations

- Only WAV upload/loading is supported.
- Uploaded files are processed immediately and not persisted.
- Audio playback depends on the browser's WAV decoding capability (all modern browsers support WAV PCM).
- The frequency spectrum is computed over the full signal (global FFT, no windowing). For non-periodic signals spectral leakage occurs at the analysis boundary. The STFT spectrogram (Module 05) uses a Hann window per frame, which addresses per-frame leakage.
- The frequency axis is linear. A log-frequency display option may be added in a future refinement.
- STFT computation is sequential (one frame at a time in Python). For very long files, server response time may be noticeable. Vectorized framing could be introduced if needed.
- No filtering, denoising, VAD, ANC, voice effects, desktop app, or AI functionality exists yet.

## Module Status

```text
[x] Module 01 — Project Skeleton
[x] Module 02 — Audio Upload and Loading
[x] Module 03 — Waveform Visualization
[x] Module 04 — Fourier Analysis
[x] Module 05 — Spectrogram / STFT
[x] Module 06 — Digital Filters
[x] Module 07 — Noise Removal
[x] Module 08 — Voice Activity Detection
[x] Module 09 — Voice Laboratory
[x] Module 10 — Audio to Image Encoder
```