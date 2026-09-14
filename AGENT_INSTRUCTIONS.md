# Intelligent Audio Platform --- Agent Instructions

## 1. Project Overview

We are building an **Intelligent Audio Platform** as a long-term,
modular software-engineering and digital-signal-processing project.

The project has two major products:

### Part A --- Web Audio Laboratory

A web application where users can upload audio files and
progressively: - inspect audio signals - visualize waveforms - analyze
frequency content - view spectrograms - apply filters - remove noise -
perform Voice Activity Detection (VAD) - apply voice/pitch/timbre
effects - experiment with adaptive noise cancellation - eventually use
AI-driven audio modules

### Part B --- Desktop Real-Time Audio Engine

A future Windows desktop application that: - performs the same
processing available in the web application where practical - captures
microphone audio in real time - processes audio with low latency -
exposes processed audio through a virtual microphone/audio route -
allows applications such as calling/meeting software to receive the
processed signal - eventually supports real-time noise suppression,
voice effects, and potentially AI voice conversion

**Part A must be developed first. Do not begin Part B until Part A has
reached the appropriate milestone.**

------------------------------------------------------------------------

## 2. Core Engineering Principle

The project must be built **incrementally and module by module**.

Do NOT implement several future modules at once.

For every task: 1. Understand the current module. 2. Implement only the
requested module. 3. Test it. 4. Integrate it cleanly. 5. Update
documentation. 6. Update `README.md`. 7. Stop and wait for the next
module.

Do not prematurely implement: - AI voice conversion - neural denoising -
real-time microphone routing - virtual microphone support - desktop
application - advanced ANC - wake-word detection - speech recognition

unless the current project plan explicitly reaches those modules.

------------------------------------------------------------------------

# 3. Technology Stack

## Frontend

Use:

-   Next.js
-   TypeScript
-   React

The frontend is responsible for: - UI - audio upload - audio playback -
waveform visualization - spectrogram visualization - controls -
displaying processing results - communicating with the backend API

## Backend

Use:

-   Python
-   FastAPI

FastAPI is the API/application layer. It should NOT contain the actual
DSP algorithms.

## DSP Core

The DSP implementation must primarily be written in Python.

Use: - NumPy - SciPy where appropriate - standard Python libraries where
appropriate

Libraries may be used for: - audio file I/O - numerical arrays -
plotting/visualization support - validation - performance-critical
standard operations

However, the important DSP algorithms must be implemented and understood
by us rather than hidden behind a one-line third-party denoising
function.

For example, do not replace our noise-removal implementation with a
black-box library call.

------------------------------------------------------------------------

# 4. Critical Architecture

The DSP core must remain independent from the web framework.

Preferred architecture:

``` text
                 Next.js Frontend
                        |
                        | HTTP/API
                        v
                   FastAPI
                        |
                        v
                Processing Services
                        |
                        v
                   DSP Core
                        |
          +-------------+-------------+
          |             |             |
       Filters       Denoising       VAD
          |             |             |
          +-------------+-------------+
                        |
                  Audio Output
```

Future architecture:

``` text
                    DSP CORE
                       |
              +--------+--------+
              |                 |
           FastAPI          Desktop App
              |                 |
          Web App          Real-time I/O
```

This separation is mandatory because the same DSP modules should
eventually be reusable by the desktop application.

The DSP layer must not depend on FastAPI or Next.js.

------------------------------------------------------------------------

# 5. Software Engineering Principles

This is also a Software Engineering project.

Prefer: - separation of concerns - modularity - single responsibility -
clear interfaces - reusable components - testability - dependency
inversion where useful - configuration separated from implementation -
meaningful naming - small functions/classes - documentation

Avoid: - giant `main.py` - giant React components - putting DSP
algorithms inside API route handlers - duplicated processing code -
hard-coded paths - unnecessary global state - premature abstraction -
overengineering for hypothetical features

Use design patterns only when they solve a real problem.

A planned architecture is:

### Pipeline Pattern

``` text
Input
  |
Processor 1
  |
Processor 2
  |
Processor 3
  |
Output
```

### Strategy Pattern

For interchangeable algorithms:

``` text
AudioProcessor
      |
      +-- SpectralSubtraction
      +-- WienerFilter
      +-- FIRFilter
      +-- ...
```

Do not create abstractions that are not yet needed.

------------------------------------------------------------------------

# 6. Development Modules

The following is the project's canonical module order.

## Phase 1 --- Foundation

### Module 01 --- Project Skeleton

-   Create repository structure.
-   Set up Next.js frontend.
-   Set up FastAPI backend.
-   Establish frontend/backend communication.
-   Establish Python environment.
-   Establish basic configuration.
-   Establish development commands.

### Module 02 --- Audio Upload and Loading

-   Upload audio from frontend.
-   Send audio to FastAPI.
-   Read audio in Python.
-   Convert audio samples to NumPy representation.
-   Extract metadata.
-   Return useful information to frontend.

### Module 03 --- Waveform Visualization

-   Generate time axis.
-   Return signal data to frontend.
-   Display waveform similar to a voice recorder.
-   Add audio playback.
-   Display duration/sample rate/channel information.

### Module 04 --- Fourier Analysis

-   Explain/implement DFT conceptually.
-   Understand FFT.
-   Compute frequency spectrum.
-   Display spectrum in frontend.

### Module 05 --- Spectrogram / STFT

-   Frame audio.
-   Apply windowing.
-   Understand STFT.
-   Generate spectrogram.
-   Display spectrogram in frontend.

------------------------------------------------------------------------

## Phase 2 --- Classical DSP

### Module 06 --- Audio Filtering

Document the progression of frequency-selective filtering implementation:

1. **Basic Frequency-Selective Filtering**
2. **Butterworth:** Maximally flat magnitude response in the passband, no passband ripple, relatively smooth response. Good general-purpose starting point.
3. **Chebyshev Type I:** Introduces passband ripple but achieves a sharper transition than Butterworth for a given order. Useful when transition sharpness matters.
4. **Chebyshev Type II:** Flat passband, but has stopband ripple. Sharper transition than Butterworth for comparable constraints.
5. **Elliptic:** Ripple in both passband and stopband. Sharpest transition for a given order among these classical designs. More complicated response, treated as an advanced filter option.
6. **Bessel:** Approximately maximally flat group delay. Good phase/time-domain behavior, but poorer magnitude selectivity compared with sharper filters.
7. **Numerical Stability (SOS/Biquad):** Direct-form polynomial implementations of high-order IIR filters suffer numerical problems. The implementation must favor numerically stable Second-Order Sections (SOS)/biquad-style processing.
8. **FIR Window-Designed Filters:** Connects naturally with Module 05 windowing knowledge. FIR provides exact linear-phase designs and predictable phase characteristics.
9. **Parks-McClellan / Equiripple FIR:** Advanced FIR design method.
10. **Practical Audio Filters:** Band-pass, notch, parametric EQ, and cascaded biquads exposing meaningful parameters rather than raw polynomial coefficients.

There is no universally "best" filter. The correct choice depends on passband requirements, stopband requirements, transition width, phase/group-delay requirements, and numerical stability. Do not implement any of this now; this is a future roadmap.

### Module 07 --- Noise Removal

Document the progressive noise removal implementation focusing on spectral/statistical methods (single-channel/stereo audio enhancement):

1. **Basic Frequency Filtering:** Use ordinary filters when noise is frequency-localized (e.g. low-frequency rumble $\to$ high-pass, high-frequency hiss $\to$ low-pass, narrow interference $\to$ notch). Explicitly state that filtering is NOT a general noise-removal solution as it damages desired speech/music when noise overlaps the signal spectrum.
2. **Noise Estimation:** Estimate noise power spectrum $P_N(k)$ from noise-dominated frames (using STFT). Accurate noise estimation is central to successful spectral enhancement.
3. **Improved Spectral Subtraction:** Instead of naive $|S| = |Y| - |N|$, implement oversubtraction factor, spectral floor, temporal/frequency smoothing, and noise tracking. Must explicitly document the classic limitation: "musical noise" artifacts.
4. **Wiener Filtering:** A more principled statistical approach. The basic gain is $H(k) = P_S(k) / (P_S(k) + P_N(k))$, where $P_S(k)$ is estimated clean-signal power and $P_N(k)$ is estimated noise power.
5. **MMSE-STSA / log-MMSE:** Statistically motivated speech-enhancement methods that estimate the clean speech spectral amplitude (or its logarithm) rather than simply subtracting noise. These are advanced and significantly more mathematically involved than subtraction.
6. **IMCRA / OM-LSA (If feasible):** Improved Minima Controlled Recursive Averaging and Optimally Modified Log-Spectral Amplitude estimators as advanced/stretch goals.

**IMPORTANT NOISE-REMOVAL PRINCIPLES:**
- Explicitly exclude LMS, NLMS, RLS, adaptive noise cancellation, and reference-microphone ANC from this roadmap. We will NOT implement adaptive noise cancellation.
- There is no universally perfect noise-removal algorithm. Performance depends on noise type, stationarity, SNR, overlap, and quality of noise estimation.

### Module 08 --- Voice Activity Detection (VAD)

Document VAD as a progressively advanced module:

1. **Short-Time Energy:** $E_m = \sum x_m[n]^2$. Basic speech/non-speech indicator. Weaknesses: background noise has high energy, quiet speech missed, threshold depends on recording conditions.
2. **Zero-Crossing Rate (ZCR):** Complementary feature. Helps distinguish certain signal characteristics but is not sufficient by itself.
3. **Spectral Features:** Derive spectral energy, spectral centroid, spectral flux, spectral entropy, and band-energy ratios from the STFT infrastructure. Combine multiple features rather than relying on one arbitrary threshold.
4. **Likelihood / Statistical Classification:** Investigate likelihood-ratio testing.
5. **GMM / Statistical Modeling:** Gaussian/GMM-based speech-vs-noise modeling to calculate the probability a frame is speech.
6. **HMM / Temporal Modeling:** Hidden Markov Models.
7. **Temporal Smoothing / State Modeling:** Introduce hysteresis, hangover time, minimum speech duration, and minimum silence duration to prevent rapid flipping between speech and silence.
8. **Comparison with WebRTC VAD:** Benchmark against established implementations like WebRTC VAD.
9. **Optional Comparison against Silero VAD:** If dependency/privacy constraints allow, benchmark against modern neural VAD.

The educational goal is to compare hand-built classical DSP VAD vs statistical VAD vs modern neural VAD. Do not implement these now.

------------------------------------------------------------------------

## Phase 3 --- Voice Manipulation

### Module 09 --- Voice Tweaks

Possible features: - gain - pitch shifting - time stretching - speed -
effects - later formant/timbre manipulation

Start with classical DSP methods.

### Module 10 --- Adaptive Noise Cancellation

Study and implement: - reference signal - adaptive filter - LMS - error
signal - convergence - performance visualization

First demonstrate ANC offline in the web application.

Do not claim that simple signal inversion is equivalent to practical
ANC.

------------------------------------------------------------------------

## Phase 4 --- Desktop Real-Time System

### Module 11 --- Desktop Application

Only after the web application foundation is stable.

The desktop application should reuse the DSP core.

Potential responsibilities: - microphone capture - real-time framing -
real-time processing - audio playback/output - controls - monitoring

### Module 12 --- Virtual Microphone / Audio Routing

Future Windows-specific work: - route processed microphone audio to a
virtual audio device - allow calling applications to use processed
audio - carefully measure latency and stability

This is an audio I/O/system-integration problem, not just a DSP problem.

### Module 13 --- Real-Time Noise Suppression

Adapt suitable denoising algorithms for real-time processing.

### Module 14 --- Real-Time Voice Effects

Apply pitch/effect processing with low latency.

------------------------------------------------------------------------

## Phase 5 --- AI Extensions

Only after the classical DSP foundation works.

Potential future modules: - neural VAD - neural noise suppression -
speaker embeddings - speaker recognition - voice conversion - speech
enhancement - speech recognition - target-speaker voice conversion

AI modules should be added as separate strategies/modules rather than
replacing the classical DSP architecture.

------------------------------------------------------------------------

# 7. Current Project Scope

Our current scope covers Phase 1 through Phase 3 (Modules 01 to 09).

```text
Upload Audio → Waveform → Frequency Analysis → Spectrogram → Filters → Denoising → VAD → Voice Tweaks
```

Do NOT implement modules beyond Phase 3 (like ANC, real-time routing, or AI) unless explicitly requested.

The focus is building robust, explainable DSP implementations for each step in this pipeline while keeping the UI professional.
------------------------------------------------------------------------

# 8. Documentation Rule --- VERY IMPORTANT

**After every meaningful implementation/update, update `README.md`.**

The README should record: - what has been implemented - current project
state - available features - how to run the project - architecture
changes - important implementation decisions - current module -
completed modules - next module - known limitations/issues

Do not let README documentation fall behind the implementation.

When a module is completed, README should clearly mark it as completed.

Example:

``` text
## Module Status

- [x] Module 01 — Project Skeleton
- [ ] Module 02 — Audio Upload and Loading
- [ ] Module 03 — Waveform Visualization
...
```

Also document major architectural decisions when they occur.

------------------------------------------------------------------------

# 9. Agent Workflow

The coding agent must follow this workflow for every instruction.

### Before coding

1.  Read this instruction file.
2.  Read `README.md`.
3.  Identify the current module.
4.  Inspect the existing implementation.
5.  Do not assume future modules have already been implemented.

### During coding

1.  Work only on the requested module.
2.  Preserve existing functionality.
3.  Avoid unnecessary refactoring.
4.  Keep DSP code independent from FastAPI.
5.  Add tests where appropriate.
6.  Keep frontend and backend responsibilities separate.

### After coding

1.  Run relevant tests.
2.  Verify the application.
3.  Update `README.md`.
4.  Mention what changed.
5.  Mention how it was tested.
6.  Mention the current module status.
7.  Mention the next module, but do not implement it.

------------------------------------------------------------------------

# 10. Communication Protocol With the Human Developer

The human developer will work with a separate AI mentor.

The workflow is:

``` text
Human
  |
  v
AI Mentor
  |
  v
Instruction/plan
  |
  v
Coding Agent
  |
  v
Implementation
  |
  v
README update
  |
  v
Human sends updated README to AI Mentor
  |
  v
AI Mentor updates instructions/next plan
  |
  v
Coding Agent
```

Therefore, the coding agent must make `README.md` sufficiently
informative that another AI can understand the project's current state
from it.

Do not rely on hidden context or previous conversations.

------------------------------------------------------------------------

# 11. Current Task

We have completed Phase 1 (Foundation) and Module 06 (Audio Filtering).

We are currently at:

**Module 07 — Noise Removal**

The current implementation task should establish noise removal techniques (e.g. spectral subtraction, Wiener filtering) in the DSP core and their controls in the frontend.

Do not implement VAD or voice effects yet.

------------------------------------------------------------------------

# 12. Important Conceptual Rules for DSP

The project is educational as well as functional.

Whenever implementing a DSP algorithm: - understand the mathematical
model first - document important assumptions - explain parameter
choices - visualize results when useful - compare input/output -
consider computational complexity - consider numerical stability -
discuss limitations

Do not blindly call a library function and consider the DSP problem
solved.

For learning, it is acceptable and encouraged to implement simplified
mathematical versions ourselves before using optimized production
implementations.

Example:

``` text
Learning:
manual DFT
    ↓
understand mathematics

Production:
NumPy FFT
    ↓
optimized implementation
```

------------------------------------------------------------------------

# 13. Definition of Done

A module is not considered complete merely because the code runs.

A module should have:

-   working implementation
-   clean integration
-   basic tests where appropriate
-   error handling
-   documented API/interface
-   README update
-   no unnecessary coupling to future modules
-   clear limitations
-   reproducible run instructions

Only then should the project proceed to the next module.

------------------------------------------------------------------------

# 14. Current Principle

**Build the simplest correct version first.**

Then:

``` text
Correct
  ↓
Understand
  ↓
Test
  ↓
Visualize
  ↓
Improve
  ↓
Optimize
  ↓
Extend
```

Do not start with AI.

Do not start with real-time processing.

Do not start with voice conversion.

Start with the digital signal itself.

# 15. Project Identity

## Project Name

**Rigel**

## Team Name

**Orion**

The project name is intentionally inspired by the star Rigel, one of the prominent/brightest stars associated with the constellation Orion.

Use the following identity consistently:

```text
Team: Orion
Project: Rigel
```

Do not rename the project or introduce a different project identity without explicit instruction from the human developer.

---

# 16. Frontend Design & UI Requirements

The web application must have a **sophisticated, polished, modern UI** rather than looking like a basic university project.

## Visual Direction

The primary visual identity should be:

- Black & white
- Minimal
- Modern
- Premium
- Clean typography
- Strong spacing and hierarchy
- Subtle animations where appropriate
- Responsive
- Professional enough for a portfolio/research demonstration

Avoid:

- excessive colors
- generic default HTML controls
- cluttered dashboards
- unnecessary gradients
- overly decorative UI
- "student project" visual style

The interface should feel like a professional audio-processing product.

## Component Libraries

We may use modern UI/component libraries to reduce boilerplate and maintain consistency.

Preferred options include:

- shadcn/ui
- Radix UI
- Tailwind CSS
- Lucide icons

Using these libraries is encouraged when they improve implementation quality.

UI libraries are acceptable for:

- buttons
- dialogs
- tabs
- sliders
- cards
- dropdowns
- tooltips
- navigation
- layout
- forms
- switches
- progress indicators

However:

**UI libraries must never replace understanding or implementation of the DSP algorithms.**

The DSP algorithms must remain our own Python implementation wherever the project explicitly requires us to build them ourselves.

## UI Philosophy

The UI should expose the underlying signal-processing concepts instead of hiding them.

For example, when noise removal is eventually implemented, the interface should allow comparison of:

```text
Original
    ↓
Waveform
    ↓
Spectrum
    ↓
Spectrogram

        VS

Processed
    ↓
Waveform
    ↓
Spectrum
    ↓
Spectrogram
```

The application should make the DSP visually understandable.

## Progressive UI Development

Do not attempt to build the entire final UI in Module 01.

The UI should evolve incrementally along with the modules.

### Module 01

Establish:

- basic visual system
- black/white theme
- typography
- global layout
- basic navigation
- basic reusable components
- backend connection/status display

### Later modules

Progressively add:

- audio upload interface
- waveform viewer
- spectrum viewer
- spectrogram
- processing controls
- VAD timeline
- filter controls
- noise-removal controls
- voice manipulation controls
- desktop-app download section

## Reusable Components

Prefer reusable components over duplicated UI code.

A possible structure is:

```text
frontend/
└── components/
    ├── audio/
    │   ├── AudioUploader.tsx
    │   ├── AudioPlayer.tsx
    │   ├── Waveform.tsx
    │   ├── Spectrum.tsx
    │   └── Spectrogram.tsx
    │
    ├── processing/
    │   ├── ProcessingPanel.tsx
    │   ├── FilterControls.tsx
    │   ├── DenoiseControls.tsx
    │   └── VADTimeline.tsx
    │
    └── ui/
        └── ...
```

Use shadcn/ui components where appropriate rather than recreating common UI primitives.

## Design Consistency

Maintain a consistent design system across the entire application.

Before introducing a new visual pattern, check whether an existing component or design token can be reused.

Do not create different button, card, input, or panel styles for different modules without a strong reason.

The final application should feel like **one coherent product**, even though its functionality is developed module by module.

---

# 17. Important Development Rule

Do not spend an entire early module building features that belong to later modules.

The visual design should be established early, but functionality must remain incremental.

The guiding principle is:

```text
Beautiful foundation
        ↓
Working functionality
        ↓
Understandable DSP
        ↓
Visual demonstration
        ↓
Incremental improvement
```

The first module should not contain:

- noise removal
- VAD
- FFT
- spectrogram
- voice conversion
- ANC
- desktop audio routing

unless explicitly requested.

---

# 18. Final Project Vision

The long-term Rigel architecture is:

```text
                         RIGEL
                    Team Orion
                         │
          ┌──────────────┴──────────────┐
          │                             │
       WEB APP                     DESKTOP APP
          │                             │
      Next.js                      Real-time I/O
      TypeScript                         │
          │                              │
       FastAPI                           │
          │                              │
          └──────────┬───────────────────┘
                     │
                     ▼
                 DSP CORE
                     │
        ┌────────────┼────────────┐
        │            │            │
    Analysis     Processing      VAD
        │            │            │
    Waveform      Filters       Speech
    FFT           Denoising     Detection
    STFT          Voice FX
    Spectrogram   ANC
                     │
                     ▼
               FUTURE AI LAYER
                     │
        ┌────────────┼────────────┐
        │            │            │
    Neural VAD   AI Denoising   Voice
                              Conversion
```

The architecture must allow future modules to be added without rewriting existing modules.

---

# 19. Current Development State

Modules 01 through 04 have been completed.

```text
Project: Rigel
Team: Orion

Current Phase:
Phase 2 — Classical DSP

Completed:
Module 01 — Project Skeleton
Module 02 — Audio Upload and Loading
Module 03 — Waveform Visualization
Module 04 — Fourier Analysis
Module 05 — Spectrogram / STFT
Module 06 — Audio Filtering

Current Module:
Module 07 — Noise Removal

Next:
Module 08 — Voice Activity Detection (VAD)
```

The project must now proceed with **Module 07 only**.

Do not implement Module 08 or any later module unless explicitly instructed.

# 20. README Requirement

`README.md` is the project's **living source of implementation history and current state**.

After every meaningful implementation/update:

1. Update the README.
2. Mark completed modules.
3. Record the current module.
4. Record what changed.
5. Record how to run the project.
6. Record important architectural decisions.
7. Record known limitations/issues.
8. State the next planned module.

The README must remain understandable to a new developer or AI agent who has never seen the project before.

---

# 21. Agent Handoff Protocol

The human developer will periodically provide the latest README to the AI mentor.

The AI mentor will use the README and this instruction file to determine the next module and may update this instruction file.

The coding agent must therefore:

- keep README accurate
- never rely on hidden conversation context
- never assume future modules are complete
- preserve completed functionality
- implement only the explicitly requested module
- stop after the requested module is complete

The intended cycle is:

```text
README
   ↓
AI Mentor
   ↓
Updated Instructions
   ↓
Coding Agent
   ↓
Implementation
   ↓
Tests
   ↓
README Update
   ↓
Human
   ↓
AI Mentor
   ↓
Next Module
```

---

# 22. Definition of Done

A module is complete only when:

- implementation works
- existing functionality still works
- relevant tests pass
- errors are handled appropriately
- architecture remains modular
- DSP code remains independent from framework code
- README is updated
- run instructions are accurate
- important decisions are documented
- known limitations are documented

Only after these conditions are met should the project move to the next module.

---

# 23. Guiding Principle

**Build the simplest correct version first.**

Then:

```text
Correct
  ↓
Understand
  ↓
Test
  ↓
Visualize
  ↓
Improve
  ↓
Optimize
  ↓
Extend
```

Do not start with AI.

Do not start with real-time processing.

Do not start with voice conversion.

Do not start with virtual microphone routing.

Start with the digital signal itself.

**Rigel is a long-term modular project. Every module should be useful on its own and should become a clean foundation for the next module.**

---


# 24. Public UI and Design Requirements

The web application must maintain a **sophisticated, polished, modern UI**. It must look like a premium audio-processing product, not a development dashboard.

1. **No Internal State in Public UI:** Do not expose module numbers, backend connection status, or internal architecture boundaries to the user.
2. **Visual Identity:** Maintain the clean, black-and-white, minimal, editorial design. Use high-quality typography.
3. **Rigel Prominence:** The "Rigel" title must be visually important and immediately noticeable in the first viewport.
4. **Theme-Ready:** Use semantic tokens (background, foreground, muted, etc.) and avoid hardcoded literal colors to ensure dark/light modes work flawlessly.
5. **Living Interface:** Preserve the interactive visual system, including ambient signal movement, micro-interactions, smooth hover states, and precision coordinate markers.
6. **Professional Copywriting:** Use natural audio-engineering terminology (e.g., "Audio Workspace", "Signal Matrix") rather than academic or implementation-heavy phrases.

Do not break the existing design identity when adding new modules. New components must perfectly integrate into the existing cohesive product.
