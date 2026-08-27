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

### Module 06 --- Digital Filters

Implement and visualize: - low-pass - high-pass - band-pass - FIR - IIR
where appropriate

Show: - original signal - filtered signal - frequency response -
relevant parameters

### Module 07 --- Noise Removal

Progressively implement: 1. Simple filtering-based noise reduction 2.
Noise estimation 3. Spectral subtraction 4. Wiener filtering if feasible

The algorithm must be explainable mathematically.

### Module 08 --- Voice Activity Detection

Start with classical methods: 1. frame-based processing 2. short-time
energy 3. zero-crossing rate 4. spectral features where useful

The frontend should show: - speech/non-speech timeline - speech
segments - speech duration - option to extract speech-only audio

VAD works perfectly well on uploaded files; it does not require
real-time audio.

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

At the beginning, the only goal is:

``` text
Upload Audio
      |
      v
Read Digital Signal
      |
      v
Display Waveform
      |
      v
Play Audio
```

Do NOT implement FFT, spectrogram, denoising, VAD, voice effects, ANC,
or AI in the first module unless explicitly requested.

The purpose of the first module is to establish the foundation
correctly.

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

We are currently at:

**Module 01 --- Project Skeleton**

The first implementation task should establish:

``` text
intelligent-audio-platform/
│
├── frontend/
│
├── backend/
│
├── README.md
├── AGENT_INSTRUCTIONS.md
└── .gitignore
```

The exact internal structure can be created progressively.

The first implementation should: - initialize the Next.js frontend -
initialize the Python/FastAPI backend - establish a minimal
health/status API - establish frontend → backend communication - provide
clear development/run instructions - update README.md

Do not implement audio processing yet.

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

Module 01 — Project Skeleton has been completed.

```text
Project: Rigel
Team: Orion

Current Phase:
Phase 1 — Foundation

Completed:
Module 01 — Project Skeleton

Current Module:
Module 02 — Audio Upload and Loading

Next:
Module 03 — Waveform Visualization
```

The project must now proceed with **Module 02 only**.

Do not implement Module 03 or any later module unless explicitly instructed.

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

# 26. Cumulative UI Feedback Addendum

## 26.1 Rigel Title Prominence

The public application title **Rigel** must be visually important and immediately noticeable.

Do not treat the project name as a small, secondary header label. The Rigel identity should be a first-viewport signal with strong typographic presence, while still preserving the premium black-and-white product direction.

# 24. Cumulative UI Feedback — Apply From Next Development Session

This section contains cumulative feedback from the human developer after reviewing the current running application.

The coding agent must read and follow this section in every future development session. It is part of the project's current requirements.

## 24.1 Public Product UI vs Development Information

The current UI exposes information such as:
- `PROJECT SKELETON ONLINE`
- `MODULE 01`
- `FastAPI connection`
- `ONLINE`
- `AVAILABLE`
- `BOUNDARY`
- `DSP core is isolated`
- `NEXT — Audio Upload and Loading`

These are useful for development/testing, but they should NOT be part of the polished public-facing product interface.

The public website should communicate the product, not the internal implementation process.

Do not expose internal development terminology in the normal public UI, including module numbers, project skeleton status, backend/API status, DSP-core availability, internal architecture boundaries, "next module" indicators, or development milestones.

This information may remain in `README.md`, logs, API responses, tests, developer tooling, or development-only diagnostics.

The user should feel that they are visiting a real audio-processing product, not a software-development dashboard.

## 24.2 Public Website Design Direction

The current interface is a good starting foundation, but it is currently too minimal for the intended final product.

The visual direction should be elevated substantially.

The goal is:

> A sophisticated, premium audio-processing product with a strong editorial/technical aesthetic.

The typography style used in the large headline:

> "Intelligent audio, built from first principles."

is liked and should be preserved or used as inspiration.

Desired qualities:
- sophisticated
- premium
- minimal but not empty
- editorial
- technical
- intentional
- high-quality typography
- strong visual hierarchy
- generous whitespace
- carefully designed cards/panels
- subtle motion
- excellent interaction states
- responsive
- polished on desktop and mobile

The application should look credible as a serious portfolio project, research/demo platform, and future commercial audio tool.

It should NOT look like a generic admin dashboard, a default shadcn starter, a basic university assignment, or a collection of unrelated cards.

## 24.3 Black-and-White Design Must Be Theme-Ready

The current visual identity is black and white, but it must be implemented so future dark mode is effortless.

Do NOT hard-code black and white throughout individual components.

Use semantic design tokens / CSS variables and the existing shadcn/Tailwind theme system where appropriate.

Conceptually:

```text
Light Theme
background → near-white
foreground → near-black
card → white
card-foreground → near-black
border → light gray
muted → gray
muted-foreground → dark gray

Dark Theme
background → near-black
foreground → near-white
card → dark gray/near-black
card-foreground → near-white
border → dark gray
muted → gray
muted-foreground → light gray
```

Prefer semantic tokens such as:

```text
background
foreground
card
card-foreground
border
muted
muted-foreground
primary
primary-foreground
accent
accent-foreground
```

Avoid repeatedly using literal values such as `bg-black`, `text-white`, `bg-white`, `text-black`, or `border-black` where theme-aware semantic tokens are appropriate.

A future dark-mode implementation should primarily require changing theme token values rather than rewriting individual components.

Do not introduce a separate hard-coded color system.

## 24.4 Public Information Architecture

The landing page should eventually communicate something closer to:

```text
RIGEL

Intelligent audio,
built from first principles.

Analyze, enhance, and transform audio
through transparent signal processing.

[ Upload Audio ]
```

The exact copy is not fixed; the concept is.

The public interface should gradually evolve toward product-oriented navigation such as:

```text
Rigel

Analyze
Enhance
Transform
About
```

or another equally coherent structure.

Do not expose the internal module roadmap as primary navigation.

The product should communicate capabilities naturally as they become available.

## 24.5 Module Development Must Not Break Product Identity

Although development is strictly incremental, the public website should not look like a new unrelated application after every module.

Every new module must integrate into the same design system.

For example:

```text
Module 02 — Audio Upload
        ↓
integrates into existing Rigel UI

Module 03 — Waveform
        ↓
integrates into existing Rigel UI

Module 04 — Frequency Analysis
        ↓
integrates into existing Rigel UI
```

Typography, spacing, component language, animations, and semantic colors should remain consistent.

## 24.6 Development Status Still Matters — But Keep It Out of the Public UI

The project must continue documenting module progress in:
- `README.md`
- `AGENT_INSTRUCTIONS.md`
- tests
- source code
- development-only diagnostics

The public UI does not need to say:

```text
Module 01
Backend Online
DSP Core Available
```

Instead, it should show what the user can actually do.

As functionality grows, the interface should transition naturally from:

```text
Project foundation
    ↓
Audio analysis tool
    ↓
Audio enhancement platform
    ↓
Full intelligent audio platform
```

## 24.7 Current UI Revision Scope

This feedback does NOT change the module order.

Module 01 is complete. The next development work is **Module 02 — Audio Upload and Loading**.

The public UI refinements described above are requirements that should remain in effect while implementing Module 02. In particular:

1. Do not reintroduce internal development/status cards into the public interface.
2. Do not show module numbers, backend status, DSP-core status, or internal architecture as normal product UI.
3. Preserve the strong editorial typography and sophisticated black-and-white product identity.
4. Keep the theme-token architecture ready for future dark mode.
5. Integrate the audio-upload experience into the existing Rigel product UI rather than creating a separate dashboard.
6. Do not implement Module 03 or later functionality.

## 24.8 Important Rule for Future Agents

Whenever the human developer provides new UI or architecture feedback, append it to this section as a new cumulative subsection rather than silently replacing older requirements.

The newest subsection has priority if it explicitly changes an older requirement.

The coding agent must read the entire cumulative feedback section before beginning future work.\n\n# 25. CURRENT TASK — MODULE 02: AUDIO UPLOAD AND LOADING\n\n**Only implement Module 02.**\n\nDo not implement Module 03 or any later module.\n\n## Goal\n\nAllow a user to upload an audio file from the Rigel web application and have the backend:\n\n```text\nAudio File\n    ↓\nFastAPI\n    ↓\nAudio Loader\n    ↓\nDigital Samples\n    ↓\nNumPy representation\n    ↓\nMetadata\n    ↓\nFrontend\n```\n\nThe user should be able to see that Rigel has successfully received and understood their audio file.\n\n## Frontend Scope\n\nCreate a polished audio-upload experience integrated into the existing Rigel design system.\n\nThe UI should allow the user to:\n\n- select an audio file\n- preferably drag and drop an audio file\n- see the selected filename\n- see upload/loading state\n- receive success/error feedback\n- view returned audio metadata\n\nThe UI must not expose internal development information such as module numbers, API status, DSP-core status, or development milestones.\n\nDo not implement waveform visualization yet. Waveform visualization belongs to Module 03.\n\n## Backend Scope\n\nCreate an appropriate FastAPI endpoint for audio upload.\n\nThe endpoint should:\n\n1. Receive the uploaded file.\n2. Validate that it is an audio file.\n3. Safely process the uploaded file.\n4. Load the audio.\n5. Convert the audio samples into a suitable NumPy representation.\n6. Extract useful metadata.\n\nAt minimum, investigate/report:\n\n- filename\n- file format/type where available\n- sample rate\n- number of channels\n- number of samples\n- duration\n- dtype/sample representation where appropriate\n\nReturn structured JSON to the frontend.\n\nDo not return the complete audio sample array to the frontend unless there is a concrete architectural reason to do so. The samples belong to the backend/DSP processing layer for now.\n\n## Audio Representation\n\nThe important conceptual flow is:\n\n```text\nAudio File\n     ↓\nAudio Decoder / Loader\n     ↓\nDigital Samples\n     ↓\nNumPy Array\n```\n\nFor example, conceptually:\n\n```python\nsample_rate = 16000\n\nsignal = np.array([\n    123,\n    145,\n    132,\n    ...\n])\n```\n\nThe implementation should document the actual representation being used.\n\nDo not implement the waveform yet.\n\n## Audio Format Scope\n\nPrefer WAV as the initial supported format because uncompressed PCM WAV is directly useful for DSP education.\n\nIf the selected audio library reliably supports additional formats without introducing unnecessary dependencies, they may be supported.\n\nDocument exactly which formats are supported.\n\nDo not introduce a large media-processing dependency merely to support many formats.\n\n## Error Handling\n\nHandle cases such as:\n\n- no file selected\n- unsupported file type\n- malformed audio\n- empty file\n- decoding failure\n- excessively large file if a reasonable limit is needed\n\nReturn useful API errors.\n\nThe frontend should present user-friendly messages rather than raw FastAPI/Python stack traces.\n\nDo not expose internal filesystem paths or sensitive implementation details.\n\n## Storage\n\nDo not build permanent user-file storage yet.\n\nIf temporary storage is needed:\n\n- use a safe temporary location\n- avoid hard-coded machine-specific paths\n- clean up temporary files where appropriate\n\nDo not introduce databases, authentication, cloud storage, or user accounts in this module.\n\n## DSP Boundary\n\nKeep audio loading separate from future DSP algorithms:\n\n```text\nFastAPI\n   ↓\nAudio Service / Loader\n   ↓\nNumPy Signal\n   ↓\nDSP Core\n```\n\nAudio decoding logic should not be placed inside future DSP algorithm classes.\n\nThe DSP core should eventually receive something conceptually like:\n\n```text\nsignal + sample_rate + metadata\n```\n\nwithout knowing that the signal originally came from a browser upload.\n\n## Testing\n\nAdd appropriate tests for:\n\n- valid audio upload\n- metadata extraction\n- invalid/unsupported upload\n- API response structure\n- basic sample representation\n\nTests should not depend on the human developer's personal audio files. Create small deterministic fixtures if needed.\n\nRun relevant backend tests and frontend checks such as lint/build.\n\n## Strict Boundary\n\nDo NOT implement:\n\n- waveform\n- FFT\n- frequency spectrum\n- spectrogram\n- STFT\n- filters\n- noise removal\n- VAD\n- voice effects\n- ANC\n- desktop application\n- AI\n\nThe only goal of this module is:\n\n> **Upload audio → load it correctly → understand its metadata → show the result.**\n\n## Definition of Done\n\nModule 02 is complete only when:\n\n- a user can upload a supported audio file\n- FastAPI receives it\n- Python loads it successfully\n- samples are represented correctly in NumPy\n- metadata is extracted\n- frontend receives structured metadata\n- UI presents the result cleanly\n- invalid uploads are handled gracefully\n- relevant tests pass\n- frontend lint/build pass\n- README is updated\n- no Module 03 functionality has been implemented\n\nThen the project is ready for:\n\n**Module 03 — Waveform Visualization**\n\nStop after Module 02.\n
