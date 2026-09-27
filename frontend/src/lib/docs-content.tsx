import type { LucideIcon } from "lucide-react";
import { Waves, SlidersHorizontal, Music, Activity, Mic, Image as ImageIcon } from "lucide-react";

export type DocFile = { path: string; note: string };
export type DocAlgorithm = { name: string; detail: string; formula?: string };
export type DocEndpoint = { method: string; endpoint: string; detail: string };

export type WorkspaceDoc = {
  slug: string;
  name: string;
  path: string;
  icon: LucideIcon;
  tagline: string;
  overview: string[];
  pipeline: string[];
  files: DocFile[];
  algorithms: DocAlgorithm[];
  api: DocEndpoint[];
  limitations: string[];
};

export const WORKSPACE_DOCS: Record<string, WorkspaceDoc> = {
  analysis: {
    slug: "analysis",
    name: "Audio Analysis",
    path: "/playground/analysis",
    icon: Waves,
    tagline: "Understand a recording in the time and frequency domains before touching it.",
    overview: [
      "Audio Analysis is a passive viewer: it computes a waveform, a full-signal frequency spectrum, and an STFT spectrogram directly from the uploaded samples, and renders all three side by side. Nothing here modifies the audio.",
      "The spectrogram used for visualization is deliberately a different, simpler implementation than the 'processing STFT' used internally by Noise Removal and Voice Lab — it discards phase and is downsampled for the display, since it only needs to show relative energy, not support exact reconstruction.",
    ],
    pipeline: [
      "Upload -> normalize samples to float64",
      "Waveform: peak-envelope decimation into 1500 bins — each bin stores (min, max) of every sample inside it, not a mean, so sharp transients survive decimation",
      "Spectrum: one full-signal FFT (no windowing) -> one-sided amplitude scaling -> dBFS -> peak-preserving downsample to 1024 display bins",
      "Spectrogram: 2048-sample frames, 512-sample hop, selectable window (Hann by default; Hamming/Blackman/Kaiser/Bartlett/Welch/rectangular also available) -> magnitude-only -> downsampled to 512 (time) x 256 (frequency) for display",
    ],
    files: [
      { path: "backend/dsp_core/waveform.py", note: "peak-envelope bin decimation" },
      { path: "backend/dsp_core/spectrum.py", note: "full-signal one-sided FFT spectrum" },
      { path: "backend/dsp_core/stft.py", note: "visualization spectrogram (magnitude-only, downsampled)" },
      { path: "backend/app/services/audio_service.py", note: "analyze / spectrum / spectrogram orchestration" },
      { path: "backend/app/api/routes.py", note: "POST /api/audio/analyze, /spectrum, /spectrogram" },
      { path: "frontend/src/app/playground/analysis/page.tsx", note: "waveform, spectrum, spectrogram viewers" },
    ],
    algorithms: [
      {
        name: "Peak-envelope waveform decimation",
        detail: "Splits the signal into N_BINS=1500 chunks and records each chunk's min and max sample — not its mean — so that a decimated waveform still shows sharp peaks a naive average would erase.",
        formula: "bin[k] = (min(chunk_k), max(chunk_k))\nt_bin[k] = (k * bin_size) / sample_rate_hz   (bin start time)",
      },
      {
        name: "One-sided amplitude spectrum",
        detail: "A single FFT over the entire signal (no windowing), converted to a one-sided amplitude spectrum by doubling all interior bins (DC and Nyquist are not doubled), then to dBFS.",
        formula: "A[0] = |X[0]|/N ;  A[k] = 2|X[k]|/N  for 1<=k<N/2 ;  A[N/2] = |X[N/2]|/N   (even N)\nmagnitude_dBFS[k] = 20 * log10(A[k] + 1e-12)",
      },
      {
        name: "Visualization spectrogram (STFT)",
        detail: "Framed FFT with a selectable analysis window, magnitude-only (phase discarded), normalized by frame length only — no one-sided doubling, since only relative energy over time matters for the display. This is a different code path from dsp_core/denoise.py's processing STFT, which keeps the full complex spectrum (for phase-exact reconstruction) and is never downsampled.",
        formula: "STFT[m,k] = sum_n x[n + m*H] * w[n] * exp(-j*2*pi*k*n/L)\ndisplayed magnitude = |STFT[m,k]| / L",
      },
    ],
    api: [
      { method: "POST", endpoint: "/api/audio/analyze", detail: "file -> metadata + waveform bins" },
      { method: "POST", endpoint: "/api/audio/spectrum", detail: "file -> one-sided amplitude spectrum (dBFS)" },
      { method: "POST", endpoint: "/api/audio/spectrogram", detail: "file + window -> time/frequency magnitude grid (dBFS)" },
    ],
    limitations: [
      "The spectrum view applies no window function, so it is subject to spectral leakage on non-periodic or short signals — acceptable for a general-purpose viewer, but not a substitute for the windowed analysis Filtering/Noise Removal use internally.",
      "Both the spectrum and spectrogram are downsampled for display (1024 bins / 512x256 grid); they are not the full-resolution arrays used internally by other modules.",
      "The spectrogram's magnitude is not one-sided-corrected, since it is meant for relative visualization only, not absolute amplitude measurement.",
    ],
  },
  filtering: {
    slug: "filtering",
    name: "Filtering",
    path: "/playground/filtering",
    icon: SlidersHorizontal,
    tagline: "Design classical digital filters and hear the result immediately.",
    overview: [
      "Filtering exposes seven classical filter design families across five filter types, all built on scipy.signal, and always applies them zero-phase (forward-backward filtering), so frequency shaping never introduces phase distortion into the result.",
      "A live, purely theoretical frequency-response curve is computed independently of the audio (from the filter coefficients alone), so you can see the exact shape of a filter before applying it to anything.",
    ],
    pipeline: [
      "Choose a filter type (low-pass / high-pass / band-pass / band-stop / peaking EQ) and a design family",
      "Design filter coefficients: IIR families (Butterworth, Chebyshev I/II, Elliptic, Bessel) via scipy as second-order sections (SOS); FIR via the window method (firwin) or Parks-McClellan equiripple design (remez)",
      "Apply zero-phase: sosfiltfilt (IIR/peaking) or filtfilt (FIR) — always forward-backward, never a single causal pass",
      "Separately: compute the theoretical frequency response (freqz/sosfreqz) from the same coefficients for the live chart",
    ],
    files: [
      { path: "backend/dsp_core/filtering.py", note: "all filter design + application + frequency response" },
      { path: "backend/app/services/filter_service.py", note: "upload orchestration" },
      { path: "backend/app/models/audio.py", note: "AudioFilterRequest / AudioFilterResponse" },
      { path: "backend/app/api/routes.py", note: "POST /api/audio/filter, /api/audio/filter/response" },
      { path: "frontend/src/app/playground/filtering/page.tsx", note: "filter type/family picker, live response chart" },
    ],
    algorithms: [
      {
        name: "IIR design families",
        detail: "Butterworth (maximally flat passband), Chebyshev I (passband ripple), Chebyshev II (stopband ripple), Elliptic (ripple in both bands, steepest rolloff for a given order), and Bessel (maximally flat group delay) — all via scipy with output=\"sos\" for numerical stability.",
        formula: "butter(N, Wn, btype, output=\"sos\")\nellip(N, rp, rs, Wn, btype, output=\"sos\")   -- rp=passband ripple dB, rs=stopband attenuation dB",
      },
      {
        name: "FIR design",
        detail: "Window method (firwin) for straightforward linear-phase FIR filters, or Parks-McClellan / Remez exchange (remez) for optimal equiripple FIR design given explicit passband/stopband edges.",
      },
      {
        name: "Peaking EQ",
        detail: "A single biquad stage using the standard Audio EQ Cookbook (Bristow-Johnson) design equations.",
        formula: "A = 10^(gain_dB/40) ;  w0 = 2*pi*f0/fs ;  alpha = sin(w0)/(2*Q)\nb0=1+alpha*A, b1=-2cos(w0), b2=1-alpha*A\na0=1+alpha/A, a1=-2cos(w0), a2=1-alpha/A",
      },
      {
        name: "Zero-phase application",
        detail: "Every filter is applied with sosfiltfilt/filtfilt (forward, then backward), which squares the magnitude response but cancels all phase distortion. This is explicitly non-causal and not suited to real-time streaming, by design.",
        formula: "H_eff(e^{jw}) = |H(e^{jw})|^2   (zero net phase)",
      },
    ],
    api: [
      { method: "POST", endpoint: "/api/audio/filter", detail: "file + filter parameters -> filtered WAV" },
      { method: "POST", endpoint: "/api/audio/filter/response", detail: "filter parameters (no audio) -> theoretical frequency response curve" },
    ],
    limitations: [
      "Filtering is always zero-phase (filtfilt/sosfiltfilt) and therefore non-causal — it cannot be used as-is for real-time/streaming filtering, which would need a single causal pass (sosfilt/lfilter) instead.",
      "IIR order is capped at 20 and FIR taps at 1000 to keep designs numerically well-conditioned.",
      "Peaking EQ is a single band (one biquad), not a multi-band parametric equalizer.",
    ],
  },
  vad: {
    slug: "vad",
    name: "Voice Activity Detection",
    path: "/playground/speech",
    icon: Activity,
    tagline: "Real-time speech/silence detection over a live streaming connection.",
    overview: [
      "Voice Activity Detection streams audio — from the microphone or synced to shared-audio playback — through a neural VAD model in real time, over a WebSocket, and displays a live speech-probability curve with detected speech segments.",
      "Two interchangeable engines are available: Silero VAD (a small recurrent/convolutional neural network run via PyTorch) and TEN VAD, a lighter-weight alternative — selectable per session.",
    ],
    pipeline: [
      "Client sends a JSON handshake: {event:'start', sampleRate, channels:1, format:'float32', engine:'silero'|'ten_vad'}",
      "Server creates a VadSession bound to the requested engine and the client's real sample rate",
      "Client streams raw binary Float32 PCM frames (mono)",
      "Silero: each 32ms chunk is resampled to exactly 512 samples at 16kHz before inference; TEN VAD: 10ms/160-sample hops at 16kHz, int16 PCM",
      "Each frame gets an instantaneous speech probability -> is_speech = probability >= 0.5, with no backend hangover/smoothing",
      "Server replies with one JSON VadSessionResult per processed chunk (timestamp, state, is_speech, activity_score, ...)",
      "Frontend derives speech segments client-side from the is_speech stream and renders an activity-score curve with a segment overlay",
    ],
    files: [
      { path: "backend/app/api/vad.py", note: "WS /api/vad/stream handshake + frame loop" },
      { path: "backend/app/services/vad_service.py", note: "VadSession, SileroVADBackend, TENVADBackend" },
      { path: "frontend/src/app/playground/speech/page.tsx", note: "microphone/shared-audio streaming UI" },
      { path: "frontend/src/hooks/use-vad-history.ts", note: "client-side speech-segment derivation" },
      { path: "frontend/src/components/vad/vad-visualization.tsx", note: "activity curve + segment overlay" },
    ],
    algorithms: [
      {
        name: "Silero VAD",
        detail: "A pretrained neural VAD model (snakers4/silero-vad, loaded via torch.hub) that natively expects 16kHz, 512-sample frames. Audio at any other source rate is resampled per 32ms chunk before inference. Decision threshold: probability >= 0.5.",
      },
      {
        name: "TEN VAD",
        detail: "A lighter alternative engine (ten_vad.TenVad), operating on 160-sample (10ms @16kHz) hops of int16 PCM, same 0.5 probability threshold.",
      },
      {
        name: "State tracking",
        detail: "Each engine reports a strict SILENCE/SPEECH state with no built-in hangover or onset smoothing on the backend — every frame's raw probability decides the state for that frame instantly. Any richer state (onset-pending, hangover) is derived client-side from the raw stream, not sent by the server.",
      },
    ],
    api: [
      { method: "WS", endpoint: "/api/vad/stream", detail: "JSON handshake, then binary Float32 PCM frames -> per-frame VadSessionResult JSON" },
    ],
    limitations: [
      "Backend decisions are frame-instantaneous with no hysteresis, so a borderline signal can flip state frame-to-frame; smoothing exists only in the frontend's history/segment derivation, not as a guaranteed backend guarantee.",
      "Non-16kHz sources are resampled per-chunk on the fly rather than the model being run natively at the source rate.",
      "Both engines assume mono input; multi-channel audio is not supported by this workspace.",
    ],
  },
  "audio-image": {
    slug: "audio-image",
    name: "Audio ↔ Image",
    path: "/playground/image",
    icon: ImageIcon,
    tagline: "Encode audio losslessly into a data image, and recover it back exactly.",
    overview: [
      "Audio ↔ Image packs a losslessly-compressed copy of an audio file into a custom binary protocol (RGL1), then maps that protocol's bytes directly onto the pixels of a grayscale PNG — one byte per pixel, not steganography (no bits are hidden inside existing image data; the image *is* the data).",
      "Decoding reverses every step and validates two separate CRC32 checksums before trusting the recovered audio, so any corruption in the image is caught rather than silently producing wrong audio.",
    ],
    pipeline: [
      "Upload a 16-bit PCM WAV file (required exactly, to guarantee a bit-exact round trip)",
      "Encode PCM16 -> FLAC (via the soundfile/libsndfile library) for lossless compression",
      "Pack into an RGL1 packet: magic bytes + type + length + header CRC32 + payload CRC32 + FLAC bytes",
      "Map the packet's raw bytes 1:1 onto pixel values of an LxL grayscale PNG, where L = ceil(sqrt(packet length))",
      "Decode: flatten the PNG's pixels back to bytes -> validate both CRC32 checksums -> unpack -> FLAC-decode back to PCM16",
    ],
    files: [
      { path: "backend/dsp_core/payload_protocol.py", note: "RGL1 packet pack/unpack + CRC32 validation" },
      { path: "backend/dsp_core/image_codec.py", note: "byte <-> grayscale PNG pixel mapping" },
      { path: "backend/dsp_core/audio_codec.py", note: "PCM16 <-> FLAC via soundfile" },
      { path: "backend/app/services/audio_image_service.py", note: "encode/decode pipeline orchestration" },
      { path: "backend/app/api/routes.py", note: "POST /api/audio/image/encode, /decode" },
      { path: "frontend/src/app/playground/image/{encode,decode}/page.tsx", note: "Encode / Decode tabs" },
    ],
    algorithms: [
      {
        name: "RGL1 protocol packet",
        detail: "A minimal, versioned binary framing format with two independent CRC32 checksums, so header corruption and payload corruption are each detected separately.",
        formula: "bytes[0:4]   = \"RGL1\" (magic)\nbyte[4]      = payload type (0x01 = FLAC)\nbytes[5:9]   = payload length (uint32, little-endian)\nbytes[9:13]  = CRC32 of bytes[0:9]      (header checksum)\nbytes[13:17] = CRC32 of the FLAC payload (payload checksum)\nbytes[17:]   = FLAC payload",
      },
      {
        name: "Grayscale pixel packing",
        detail: "Every packet byte becomes one 8-bit grayscale pixel value, arranged into the smallest square image that fits (padded with zeros); this is a direct data-to-image mapping, not steganographic bit-hiding.",
        formula: "L = ceil(sqrt(packet_length_bytes))\npixel[i] = packet_byte[i]  for i < packet_length, else 0 (padding)",
      },
    ],
    api: [
      { method: "POST", endpoint: "/api/audio/image/encode", detail: "16-bit PCM WAV -> PNG data image (stats in X-Rigel-* response headers)" },
      { method: "POST", endpoint: "/api/audio/image/decode", detail: "PNG data image -> recovered WAV, CRC32-validated" },
    ],
    limitations: [
      "Only 16-bit PCM WAV input is accepted for encoding, specifically to guarantee an exact, lossless round trip — other bit depths/formats are rejected rather than silently down-converted.",
      "Maximum image side is 1024px, capping payload capacity at roughly 1MB per image.",
      "The reported 'compression ratio' is FLAC size divided by PCM size (a fraction below 1.0 for typical audio), not an 'N:1' style ratio.",
    ],
  },
  "noise-removal": {
    slug: "noise-removal",
    name: "Noise Removal",
    path: "/playground/enhancement",
    icon: Music,
    tagline: "Four classical statistical methods for suppressing background noise.",
    overview: [
      "Noise Removal estimates the power spectrum of the background noise in a recording and subtracts or attenuates it, frame by frame, while preserving the original phase. Every method operates entirely in the STFT domain and shares the same analysis/synthesis front end.",
      "The four methods represent increasing sophistication: a simple power-domain subtraction, a statistically optimal linear filter, a perceptually-motivated MMSE estimator, and a two-stage noise tracker with an explicit speech-presence model.",
    ],
    pipeline: [
      "Uploaded audio -> normalize to float64 in [-1, 1]",
      "STFT: 2048-sample symmetric Hann window, 512-sample hop (75% overlap), centered padding",
      "Robust initial noise estimate from the lowest-energy 10th percentile of frames (avoids cold-start bias from speech at t=0)",
      "Per-frame noise PSD tracking (Minimum Statistics, or IMCRA for OM-LSA)",
      "Per-frame, per-bin gain computation (method-specific formula below)",
      "Frequency-domain gain smoothing (3-bin average) + a gain floor, to suppress isolated musical-noise artifacts",
      "Gain x original complex STFT bin (phase always preserved exactly)",
      "ISTFT: overlap-add with element-wise window-power normalization",
    ],
    files: [
      { path: "backend/dsp_core/denoise.py", note: "all four algorithms, shared STFT/ISTFT, noise trackers" },
      { path: "backend/app/services/denoise_service.py", note: "dispatches by method name, WAV in/out" },
      { path: "backend/app/models/denoise.py", note: "AudioDenoiseRequest — per-method parameters" },
      { path: "backend/app/api/routes.py", note: "POST /api/audio/denoise" },
      { path: "frontend/src/app/playground/enhancement/page.tsx", note: "method picker, parameter sliders, A/B playback" },
    ],
    algorithms: [
      {
        name: "Improved Spectral Subtraction",
        detail: "Power-domain subtraction with an oversubtraction factor (alpha) and a spectral floor (beta), plus temporal (IIR) and frequency (3-bin) gain smoothing to reduce musical noise.",
        formula: "P_enh = max(P_noisy - alpha * P_noise, beta * P_noisy)\nG_raw = sqrt(P_enh / P_noisy), clipped to [sqrt(beta), 1.0]\nG_t[m] = 0.7 * G_t[m-1] + 0.3 * G_raw[m]  (temporal smoothing)",
      },
      {
        name: "Decision-Directed Wiener",
        detail: "Classical Wiener gain driven by a recursively estimated a priori SNR (xi), using the previous frame's ENHANCED power (not the noisy power) as required by the Ephraim-Malah decision-directed method.",
        formula: "gamma = P_noisy / P_noise  (a posteriori SNR)\nxi[m] = alpha_dd * (|Y[m-1]|^2 / P_noise[m]) + (1 - alpha_dd) * max(gamma[m] - 1, 0)\nG = xi / (1 + xi)",
      },
      {
        name: "Log-MMSE",
        detail: "Ephraim & Malah (1985) log-spectral-amplitude MMSE estimator, using the exponential integral E1(v). Same xi/gamma recursion as Wiener, different (perceptually motivated) gain function.",
        formula: "v = [xi / (1 + xi)] * gamma\nG = [xi / (1 + xi)] * exp(0.5 * E1(v))   (large v)\nG ~ sqrt(xi/(1+xi)) * exp(-0.5*gamma_E)/sqrt(gamma) * exp(0.5*v)   (small v, asymptotic)",
      },
      {
        name: "IMCRA + OM-LSA",
        detail: "Cohen (2003) two-stage Improved Minima-Controlled Recursive Averaging noise tracker feeds a Speech-Presence Probability (SPP) into the Optimally-Modified LSA gain, which geometrically blends the Log-MMSE gain with a fixed minimum gain (G_min) based on how confident the tracker is that speech is present.",
        formula: "log(G_omlsa) = p * log(G_lmmse) + (1 - p) * log(G_min)\np = Speech-Presence Probability, from IMCRA's smoothed-periodogram minimum tracking",
      },
    ],
    api: [
      { method: "POST", endpoint: "/api/audio/denoise", detail: "multipart form: file + method + per-method parameters -> processed WAV" },
    ],
    limitations: [
      "Blind, single-channel noise estimation cannot guarantee a clean estimate when speech occupies the entire recording from the start (continuous speech).",
      "Minimum Statistics / IMCRA can mistake a persistent tonal component for noise if it never drops out.",
      "All four estimators lag a sudden, non-stationary jump in the noise floor (their memory constants are ~0.5-1.5s).",
      "A small residual 'pre-echo' at sharp noise-to-speech transitions is structurally inherent to fixed-window STFT processing; the gain-smoothing/floor fix minimizes it but cannot make it exactly zero.",
    ],
  },
  "voice-lab": {
    slug: "voice-lab",
    name: "Voice Lab",
    path: "/playground/voice-lab",
    icon: Mic,
    tagline: "Measure pitch, loudness and rhythm; reshape a voice with an ordered effect chain.",
    overview: [
      "Voice Lab has two halves: a measurement side (pitch, loudness, rhythm — live from the microphone or offline from any uploaded/shared audio) and a transformation side (a strictly ordered chain of classical DSP effects applied to the shared audio).",
      "Every effect chain recomputation starts from the original uploaded samples — results are never accumulated across repeated calls, so re-ordering or changing one effect's parameters never compounds numerical error from a previous run.",
    ],
    pipeline: [
      "AUDIO SOURCES: Current/Shared Audio, or Microphone -> AudioWorklet -> mono Float32 PCM",
      "MEASUREMENT CORE: Level (RMS/peak dBFS), Pitch (YIN), Rhythm (BPM via envelope autocorrelation)",
      "Live path: WebSocket JSON handshake negotiates the real sample rate, then binary PCM frames -> per-frame LiveMeasurementFrame",
      "Offline path: POST the full sample array -> one MeasurementResponse (loudness, pitch, rhythm)",
      "EFFECT CHAIN: Gain -> Speed -> Time Stretch -> Pitch Shift -> Effect (Echo/Reverb/Chorus/Distortion) -> Output, applied in the exact order shown in the UI",
      "Recording completion promotes the microphone take to the Playground's current/shared audio (not to 'processed audio', which is reserved for actual effect-chain output)",
    ],
    files: [
      { path: "backend/dsp_core/measurement.py", note: "YIN pitch, RMS/peak loudness, BPM autocorrelation" },
      { path: "backend/dsp_core/time_scale.py", note: "polyphase resampling (Speed) and phase-vocoder time stretch" },
      { path: "backend/dsp_core/pitch_shift.py", note: "pitch shift = time-stretch + resample" },
      { path: "backend/dsp_core/effects.py", note: "Echo/Delay, Chorus, Soft Distortion, Schroeder Reverb" },
      { path: "backend/dsp_core/voice_gain.py", note: "linear/dB/peak/RMS gain and normalization" },
      { path: "backend/app/api/routes.py", note: "POST /api/audio/voice/process, POST /api/audio/measure, WS /api/audio/measure/stream" },
      { path: "frontend/src/app/playground/voice-lab/page.tsx", note: "effect chain UI, live measurement panel, mic capture" },
    ],
    algorithms: [
      {
        name: "YIN pitch estimation",
        detail: "Classical autocorrelation-based fundamental frequency estimator (de Cheveigne & Kawahara, 2002): cumulative mean normalized difference function, absolute threshold, parabolic interpolation for sub-sample accuracy. A frame is judged voiced only if the CMNDF dips below threshold (0.15) and confidence exceeds 0.5.",
        formula: "d'(tau) = d(tau) / [(1/tau) * sum_{j=1..tau} d(j)]   (cumulative mean normalized difference)\nconfidence = 1 - d'(tau_est)",
      },
      {
        name: "Speed (tape-style resampling)",
        detail: "Polyphase resampling (scipy.signal.resample_poly) — duration and pitch change together, proportionally. Distinct from Time Stretch, which preserves pitch.",
        formula: "speed = 2.0 -> half duration, pitch doubled",
      },
      {
        name: "Time Stretch (phase vocoder)",
        detail: "First-principles phase vocoder: STFT -> track instantaneous frequency via phase deviation from the expected per-hop advance -> accumulate a stretched synthesis phase -> overlap-add.",
        formula: "omega_true = omega_k + wrap(delta_phi - omega_k * Ha) / Ha\npsi[m] = psi[m-1] + omega_true * Hs",
      },
      {
        name: "BPM / Rhythm",
        detail: "Full-wave rectify -> decimate to a 200Hz envelope -> remove DC -> autocorrelate -> search the 50-200 BPM lag range for the peak. Confidence is the normalized autocorrelation peak; BPM is only reported when confidence exceeds 0.25 (deliberately strict, since ordinary speech has no steady beat and must not be assigned a fabricated tempo).",
      },
    ],
    api: [
      { method: "POST", endpoint: "/api/audio/voice/process", detail: "samples + sample_rate_hz + ordered effect_chain -> processed samples + metrics" },
      { method: "POST", endpoint: "/api/audio/measure", detail: "full-file offline measurement -> {loudness, pitch, rhythm}" },
      { method: "WS", endpoint: "/api/audio/measure/stream", detail: "JSON {event:'start', sampleRate} handshake, then binary Float32 PCM frames -> per-frame LiveMeasurementFrame" },
    ],
    limitations: [
      "BPM is intentionally not shown for audio with no steady beat (most ordinary speech) — this is a correct measurement result, not a missing feature.",
      "YIN's search range defaults to 60-500Hz, tuned for voice; it is not intended for detecting the pitch of arbitrary instruments outside that range.",
      "The phase vocoder can introduce mild 'phasiness' on highly transient or percussive material — a known, inherent limitation of STFT-based time stretching.",
    ],
  },
};

export const WORKSPACE_ORDER = ["analysis", "filtering", "noise-removal", "vad", "voice-lab", "audio-image"];
