"use client";

/**
 * Voice Lab — Module 09
 *
 * "Shape the temporal, spectral, and tonal character of a recording."
 *
 * Features:
 *   - Controls for Gain, Speed, Stretch, Pitch, and five classic effects
 *   - Transformation Observatory: dual waveform + spectrum side-by-side
 *   - Independent A/B playback for original vs processed
 *   - WAV download of processed audio
 *   - Clipping risk indicator
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { usePlayground } from "@/contexts/playground-context";
import {
  processVoiceLab,
  decodeBlobToSamples,
  encodeSamplesToWavBlob,
  type VoiceEffect,
  type VoiceGainOperation,
  type VoiceSpeedOperation,
  type VoiceTimeStretchOperation,
  type VoicePitchShiftOperation,
  type VoiceProcessResponse,
  analyzeAudio,
  fetchSpectrum,
  fetchSpectrogram,
  type WaveformData,
  type SpectrumData,
  type SpectrogramData,
} from "@/lib/api";
import {
  Mic, Loader2, AlertCircle, CheckCircle,
  Zap, Activity, Music2, Radio, Gauge,
  FlaskConical, Info, Play, Square, Settings2, CloudRain
} from "lucide-react";
import { SignalComparison } from "@/components/audio/signal-comparison";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type EffectType = "none" | "tremolo" | "ring_modulation" | "delay" | "chorus" | "soft_distortion" | "timbre" | "reverb";

interface GainParams   { mode: "db" | "linear" | "peak" | "rms"; value: number }
interface SpeedParams  { speed: number }
interface StretchParams { stretch: number }
interface PitchParams  { semitones: number }
interface TremoloParams     { rate_hz: number; depth: number }
interface RingModParams     { carrier_hz: number }
interface DelayParams       { delay_ms: number; feedback: number; mix: number }
interface ChorusParams      { rate_hz: number; depth_ms: number; base_delay_ms: number; mix: number }
interface DistortionParams  { drive: number }
interface TimbreParams      { tilt: number }
interface ReverbParams      { room_size: number; decay: number; wet: number }

// ---------------------------------------------------------------------------
// Default parameter values
// ---------------------------------------------------------------------------

const DEFAULT_GAIN: GainParams    = { mode: "db", value: 0 };
const DEFAULT_SPEED: SpeedParams  = { speed: 1.0 };
const DEFAULT_STRETCH: StretchParams = { stretch: 1.0 };
const DEFAULT_PITCH: PitchParams  = { semitones: 0 };
const DEFAULT_TREMOLO: TremoloParams     = { rate_hz: 5, depth: 0.5 };
const DEFAULT_RINGMOD: RingModParams     = { carrier_hz: 440 };
const DEFAULT_DELAY: DelayParams         = { delay_ms: 300, feedback: 0.4, mix: 0.5 };
const DEFAULT_CHORUS: ChorusParams       = { rate_hz: 1.5, depth_ms: 3, base_delay_ms: 15, mix: 0.5 };
const DEFAULT_DISTORTION: DistortionParams = { drive: 3 };
const DEFAULT_TIMBRE: TimbreParams         = { tilt: 0 };
const DEFAULT_REVERB: ReverbParams         = { room_size: 0.5, decay: 0.5, wet: 0.3 };

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function SectionHeader({ title, icon: Icon, color = "text-indigo-400" }: {
  title: string; icon: React.ComponentType<{ className?: string }>; color?: string;
}) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon className={`h-4 w-4 ${color}`} />
      <h3 className="text-xs font-bold tracking-widest text-slate-400 uppercase">{title}</h3>
    </div>
  );
}

function Knob({ label, value, min, max, step = 0.01, unit = "", onChange, precision = 2 }: {
  label: string; value: number; min: number; max: number; step?: number;
  unit?: string; onChange: (v: number) => void; precision?: number;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">{label}</label>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="flex-1 h-1.5 appearance-none rounded-full bg-white/10 accent-indigo-500 cursor-pointer"
        />
        <span className="text-xs font-mono text-indigo-300 w-16 text-right">
          {value.toFixed(precision)}{unit}
        </span>
      </div>
    </div>
  );
}

function NumField({ label, value, min, max, step = 1, onChange }: {
  label: string; value: number; min?: number; max?: number; step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">{label}</label>
      <input
        type="number" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full rounded-xl border border-white/10 bg-[#0d0d1a] px-3 py-2 text-sm font-mono text-white focus:border-indigo-500/50 focus:outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
      />
    </div>
  );
}

// Removed MiniWaveform and MiniSpectrum in favor of large analytical panels

// ---------------------------------------------------------------------------
// Main Voice Lab page
// ---------------------------------------------------------------------------

export default function VoiceLabPage() {
  const { state, setProcessedAudio } = usePlayground();

  // ---- parameter state ----
  const [gainParams, setGainParams]       = useState<GainParams>(DEFAULT_GAIN);
  const [speedParams, setSpeedParams]     = useState<SpeedParams>(DEFAULT_SPEED);
  const [stretchParams, setStretchParams] = useState<StretchParams>(DEFAULT_STRETCH);
  const [pitchParams, setPitchParams]     = useState<PitchParams>(DEFAULT_PITCH);
  const [effectType, setEffectType]       = useState<EffectType>("none");
  const [tremoloP, setTremoloP]           = useState<TremoloParams>(DEFAULT_TREMOLO);
  const [ringModP, setRingModP]           = useState<RingModParams>(DEFAULT_RINGMOD);
  const [delayP, setDelayP]               = useState<DelayParams>(DEFAULT_DELAY);
  const [chorusP, setChorusP]             = useState<ChorusParams>(DEFAULT_CHORUS);
  const [distortionP, setDistortionP]     = useState<DistortionParams>(DEFAULT_DISTORTION);
  const [timbreP, setTimbreP]             = useState<TimbreParams>(DEFAULT_TIMBRE);
  const [reverbP, setReverbP]             = useState<ReverbParams>(DEFAULT_REVERB);
  const [preventClipping, setPreventClipping] = useState(false);

  // ---- processing state ----
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VoiceProcessResponse | null>(null);
  const [processedSamples, setProcessedSamples] = useState<number[][]>([]);

  // ---- audio object URLs ----
  const [procAudioUrl, setProcAudioUrl] = useState<string | null>(null);

  const [procWaveform, setProcWaveform] = useState<WaveformData | null>(null);
  const [procSpectrum, setProcSpectrum] = useState<SpectrumData | null>(null);
  const [procSpectrogram, setProcSpectrogram] = useState<SpectrogramData | null>(null);
  // ---- Build request from current params ----
  const buildRequest = useCallback(async () => {
    if (state.status !== "success") throw new Error("No audio loaded.");
    const blob = await fetch(state.objectUrl).then(r => r.blob());
    const { samples, sampleRate } = await decodeBlobToSamples(blob);
    const gain: VoiceGainOperation | null = (gainParams.mode === "db" && Math.abs(gainParams.value) < 1e-9)
      ? null
      : { op: "gain", mode: gainParams.mode, gain_value: gainParams.value };

    const speed: VoiceSpeedOperation | null = Math.abs(speedParams.speed - 1.0) < 1e-9
      ? null : { op: "speed", speed: speedParams.speed };

    const stretch: VoiceTimeStretchOperation | null = Math.abs(stretchParams.stretch - 1.0) < 1e-9
      ? null : { op: "time_stretch", stretch: stretchParams.stretch };

    const pitch: VoicePitchShiftOperation | null = Math.abs(pitchParams.semitones) < 1e-9
      ? null : { op: "pitch_shift", semitones: pitchParams.semitones };

    let effect: VoiceEffect | null = null;
    if (effectType === "tremolo")        effect = { op: "tremolo", ...tremoloP };
    else if (effectType === "ring_modulation") effect = { op: "ring_modulation", ...ringModP };
    else if (effectType === "delay")     effect = { op: "delay", ...delayP };
    else if (effectType === "chorus")    effect = { op: "chorus", ...chorusP };
    else if (effectType === "soft_distortion") effect = { op: "soft_distortion", ...distortionP };
    else if (effectType === "timbre")    effect = { op: "timbre", ...timbreP };
    else if (effectType === "reverb")    effect = { op: "reverb", ...reverbP };

    return { samples, sampleRate, gain, speed, stretch, pitch, effect };
  }, [state, gainParams, speedParams, stretchParams, pitchParams, effectType,
      tremoloP, ringModP, delayP, chorusP, distortionP, timbreP, reverbP]);

  // ---- Process ----
  const handleProcess = useCallback(async () => {
    setProcessing(true);
    setError(null);
    try {
      const { samples, sampleRate, gain, speed, stretch, pitch, effect } = await buildRequest();
      const resp = await processVoiceLab({
        samples, sample_rate_hz: sampleRate,
        gain, speed, time_stretch: stretch, pitch_shift: pitch, effect,
        prevent_clipping: preventClipping,
      });
      setResult(resp);
      setProcessedSamples(resp.samples);
      const wavBlob = encodeSamplesToWavBlob(resp.samples, resp.sample_rate_hz);
      setProcessedAudio(wavBlob);
      // Create a stable object URL for the audio player
      if (procAudioUrl) URL.revokeObjectURL(procAudioUrl);
      setProcAudioUrl(URL.createObjectURL(wavBlob));

      // Fetch analytical data for the large viewers
      try {
        const file = new File([wavBlob], "processed.wav", { type: "audio/wav" });
        const [analyzeRes, specRes, spectroRes] = await Promise.all([
          analyzeAudio(file),
          fetchSpectrum(file),
          fetchSpectrogram(file)
        ]);
        setProcWaveform(analyzeRes.waveform);
        setProcSpectrum(specRes.spectrum);
        setProcSpectrogram(spectroRes.spectrogram);
      } catch (err) {
        console.error("Failed to generate analytical data for processed audio", err);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Processing failed.");
    } finally {
      setProcessing(false);
    }
  }, [buildRequest, preventClipping, setProcessedAudio]);

  if (state.status === "idle" || state.status === "loading" || state.status === "error") {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center gap-6 text-center">
        <div className="relative">
          <div className="absolute inset-0 rounded-full bg-indigo-500/20 blur-2xl scale-150" />
          <div className="relative rounded-full bg-indigo-900/30 border border-indigo-500/20 p-6">
            <Mic className="h-12 w-12 text-indigo-400" />
          </div>
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">VOICE LAB</h1>
          <p className="text-slate-400 mt-2 text-sm max-w-md">
            Shape the temporal, spectral, and tonal character of a recording.
          </p>
          <p className="text-slate-500 mt-4 text-xs">Upload a WAV file from the sidebar to begin.</p>
        </div>
      </div>
    );
  }

  const procSamplesFirst = processedSamples[0] ?? null;

  return (
    <div className="space-y-8 pb-16">
      {/* ── Page header ── */}
      <div className="flex flex-col items-center justify-center text-center mt-4 mb-10">
        <h1 className="flex items-center justify-center gap-3 text-3xl font-bold tracking-tight sm:text-4xl text-white">
          <Mic className="h-8 w-8 text-indigo-400" />
          Voice Laboratory
        </h1>
        <p className="mt-4 text-base text-slate-400 max-w-2xl mx-auto">
          Shape the temporal, spectral, and tonal character of a recording.
        </p>
      </div>

      <div className="flex flex-col gap-12">
        {/* ── Controls Panel (Grid) ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">



          {/* Gain */}
          <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c1a]/80 backdrop-blur-sm p-5">
            <SectionHeader title="Gain / Level" icon={Gauge} />
            <div className="space-y-4">
              <div className="flex gap-2 mb-3">
                {(["db", "linear", "peak", "rms"] as const).map(m => (
                  <button key={m}
                    onClick={() => setGainParams(p => ({ ...p, mode: m }))}
                    className={`flex-1 rounded-lg py-1.5 text-[10px] font-bold tracking-wider uppercase transition-colors ${
                      gainParams.mode === m
                        ? "bg-indigo-600 text-white"
                        : "bg-white/5 text-slate-400 hover:bg-white/10"
                    }`}>{m}</button>
                ))}
              </div>
              <Knob
                label={gainParams.mode === "db" ? "Gain (dB)" : gainParams.mode === "linear" ? "Gain Factor" : gainParams.mode === "peak" ? "Target Peak" : "Target RMS"}
                value={gainParams.value}
                min={gainParams.mode === "db" ? -40 : gainParams.mode === "linear" ? 0 : 0.01}
                max={gainParams.mode === "db" ? 40 : gainParams.mode === "linear" ? 4 : 1}
                step={gainParams.mode === "db" ? 0.5 : 0.01}
                unit={gainParams.mode === "db" ? " dB" : ""}
                onChange={v => setGainParams(p => ({ ...p, value: v }))}
                precision={gainParams.mode === "db" ? 1 : 3}
              />
            </div>
          </div>

          {/* Speed */}
          <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c1a]/80 backdrop-blur-sm p-5">
            <SectionHeader title="Speed" icon={Activity} color="text-amber-400" />
            <div className="space-y-3">
              <Knob label="Speed Factor" value={speedParams.speed} min={0.25} max={4.0} step={0.05}
                onChange={v => setSpeedParams({ speed: v })} precision={2} />
              <p className="text-[10px] text-slate-600 leading-relaxed">
                Changes duration and pitch proportionally. 2.0× = half duration, pitch ×2.
              </p>
            </div>
          </div>

          {/* Time Stretch */}
          <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c1a]/80 backdrop-blur-sm p-5">
            <SectionHeader title="Time Stretch (Phase Vocoder)" icon={Zap} color="text-cyan-400" />
            <div className="space-y-3">
              <Knob label="Stretch Factor" value={stretchParams.stretch} min={0.25} max={4.0} step={0.05}
                onChange={v => setStretchParams({ stretch: v })} precision={2} />
              <p className="text-[10px] text-slate-600 leading-relaxed">
                Changes duration while approximately preserving pitch. 2.0 = twice as long.
              </p>
            </div>
          </div>

          {/* Pitch */}
          <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c1a]/80 backdrop-blur-sm p-5">
            <SectionHeader title="Pitch Shift" icon={Music2} color="text-rose-400" />
            <div className="space-y-3">
              <Knob label="Semitones" value={pitchParams.semitones} min={-24} max={24} step={0.5}
                unit=" st" onChange={v => setPitchParams({ semitones: v })} precision={1} />
              <p className="text-[10px] text-slate-600">
                +12 semitones = one octave up. Duration approximately preserved.
              </p>
            </div>
          </div>

          {/* Effect selector */}
          <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c1a]/80 backdrop-blur-sm p-5">
            <SectionHeader title="Effect" icon={FlaskConical} color="text-emerald-400" />
            <div className="grid grid-cols-4 gap-1.5 mb-4">
              {[
                { id: "none", label: "None" },
                { id: "timbre", label: "Timbre" },
                { id: "tremolo", label: "Tremolo" },
                { id: "ring_modulation", label: "Ring Mod" },
                { id: "delay", label: "Delay" },
                { id: "chorus", label: "Chorus" },
                { id: "reverb", label: "Reverb" },
                { id: "soft_distortion", label: "Distort" },
              ].map(({ id, label }) => (
                <button key={id}
                  onClick={() => setEffectType(id as EffectType)}
                  className={`rounded-lg py-2 text-[10px] font-bold tracking-wider uppercase transition-colors ${
                    effectType === id
                      ? "bg-emerald-600 text-white shadow-lg shadow-emerald-500/20"
                      : "bg-white/5 text-slate-400 hover:bg-white/10"
                  }`}>{label}</button>
              ))}
            </div>

            {/* Effect params */}
            {effectType === "timbre" && (
              <div className="pt-3 border-t border-white/5">
                <Knob label="Spectral Tilt (Warmth ↔ Brightness)" value={timbreP.tilt} min={-1.0} max={1.0} step={0.01}
                  onChange={v => setTimbreP({ tilt: v })} precision={2} />
                <p className="text-[10px] text-slate-600 mt-2">
                  Adjusts the spectral balance without shifting pitch. <br/>-1.0 = warm (boost lows), +1.0 = bright (boost highs).
                </p>
              </div>
            )}
            {effectType === "tremolo" && (
              <div className="space-y-3 pt-3 border-t border-white/5">
                <Knob label="Rate (Hz)" value={tremoloP.rate_hz} min={0.1} max={20} step={0.1}
                  unit=" Hz" onChange={v => setTremoloP(p => ({ ...p, rate_hz: v }))} />
                <Knob label="Depth" value={tremoloP.depth} min={0} max={1} step={0.01}
                  onChange={v => setTremoloP(p => ({ ...p, depth: v }))} />
              </div>
            )}
            {effectType === "ring_modulation" && (
              <div className="pt-3 border-t border-white/5">
                <Knob label="Carrier (Hz)" value={ringModP.carrier_hz} min={0} max={8000} step={10}
                  unit=" Hz" onChange={v => setRingModP({ carrier_hz: v })} precision={0} />
              </div>
            )}
            {effectType === "delay" && (
              <div className="space-y-3 pt-3 border-t border-white/5">
                <Knob label="Delay (ms)" value={delayP.delay_ms} min={10} max={1000} step={10}
                  unit=" ms" onChange={v => setDelayP(p => ({ ...p, delay_ms: v }))} precision={0} />
                <Knob label="Feedback" value={delayP.feedback} min={0} max={0.98} step={0.01}
                  onChange={v => setDelayP(p => ({ ...p, feedback: v }))} />
                <Knob label="Mix" value={delayP.mix} min={0} max={1} step={0.01}
                  onChange={v => setDelayP(p => ({ ...p, mix: v }))} />
              </div>
            )}
            {effectType === "chorus" && (
              <div className="space-y-3 pt-3 border-t border-white/5">
                <Knob label="LFO Rate (Hz)" value={chorusP.rate_hz} min={0.1} max={10} step={0.1}
                  unit=" Hz" onChange={v => setChorusP(p => ({ ...p, rate_hz: v }))} />
                <Knob label="Depth (ms)" value={chorusP.depth_ms} min={0.5} max={14} step={0.5}
                  unit=" ms" onChange={v => setChorusP(p => ({ ...p, depth_ms: v }))} />
                <Knob label="Base Delay (ms)" value={chorusP.base_delay_ms} min={chorusP.depth_ms + 1} max={50} step={1}
                  unit=" ms" onChange={v => setChorusP(p => ({ ...p, base_delay_ms: v }))} precision={0} />
                <Knob label="Mix" value={chorusP.mix} min={0} max={1} step={0.01}
                  onChange={v => setChorusP(p => ({ ...p, mix: v }))} />
              </div>
            )}
            {effectType === "soft_distortion" && (
              <div className="pt-3 border-t border-white/5">
                <Knob label="Drive (β)" value={distortionP.drive} min={0.1} max={50} step={0.1}
                  onChange={v => setDistortionP({ drive: v })} />
                <p className="text-[10px] text-slate-600 mt-2">
                  y[n] = tanh(β·x[n]) / tanh(β). Higher β = heavier saturation.
                </p>
              </div>
            )}
            {effectType === "reverb" && (
              <div className="space-y-3 pt-3 border-t border-white/5">
                <Knob label="Room Size" value={reverbP.room_size} min={0.1} max={1.0} step={0.05}
                  onChange={v => setReverbP(p => ({ ...p, room_size: v }))} precision={2} />
                <Knob label="Decay" value={reverbP.decay} min={0.0} max={0.99} step={0.01}
                  onChange={v => setReverbP(p => ({ ...p, decay: v }))} />
                <Knob label="Wet/Dry Mix" value={reverbP.wet} min={0.0} max={1.0} step={0.01}
                  onChange={v => setReverbP(p => ({ ...p, wet: v }))} />
                <p className="text-[10px] text-slate-600">
                  Schroeder Reverberator (Parallel Comb Filters → Serial All-Pass).
                </p>
              </div>
            )}
          </div>

          {/* Output safety */}
          <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c1a]/80 backdrop-blur-sm p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-slate-300">Prevent Clipping</p>
                <p className="text-[10px] text-slate-600 mt-0.5">If peak &gt; 1.0, normalize output to 1.0 as a final step.</p>
              </div>
              <button
                onClick={() => setPreventClipping(p => !p)}
                className={`relative w-11 h-6 rounded-full transition-colors ${preventClipping ? "bg-indigo-600" : "bg-white/10"}`}
              >
                <span className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform ${preventClipping ? "translate-x-5" : ""}`} />
              </button>
            </div>
          </div>

          {/* Process button */}
          <div className="col-span-1 md:col-span-2 lg:col-span-3 flex justify-center mt-4">
            <button
              id="voice-lab-process-btn"
              onClick={handleProcess}
              disabled={processing}
              className="w-full max-w-md flex items-center justify-center gap-2 rounded-2xl bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-4 text-base font-bold text-white transition-all shadow-lg shadow-indigo-500/25"
            >
              {processing ? (
                <><Loader2 className="h-5 w-5 animate-spin" /> Processing…</>
              ) : (
                <><Zap className="h-5 w-5" /> Process Audio</>
              )}
            </button>
          </div>

          {/* Error banner */}
          {error && (
            <div className="col-span-1 md:col-span-2 lg:col-span-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 flex items-start justify-center gap-2 max-w-2xl mx-auto w-full">
              <AlertCircle className="h-5 w-5 text-red-400 mt-0.5 shrink-0" />
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}
        </div>

        {/* ── Transformation Observatory ── */}
        <div className="space-y-6 max-w-[100rem] mx-auto w-full mt-4">

          {/* Observatory header */}
          <div className="flex flex-col items-center justify-center gap-2 text-center mb-6">
            <Radio className="h-6 w-6 text-indigo-400" />
            <h2 className="text-xl font-bold tracking-widest text-slate-300 uppercase">Transformation Observatory</h2>
          </div>
          <OperationMathCard
            gainParams={gainParams}
            speedParams={speedParams}
            stretchParams={stretchParams}
            pitchParams={pitchParams}
            effectType={effectType}
            tremoloP={tremoloP}
            ringModP={ringModP}
            delayP={delayP}
            chorusP={chorusP}
            distortionP={distortionP}
            timbreP={timbreP}
            reverbP={reverbP}
          />

          {/* Result metadata strip */}
          {result && (
            <div className="flex flex-wrap justify-center gap-2 mb-6 mt-6">
              {result.operations_applied.map((op, i) => (
                <span key={i} className="rounded-lg bg-indigo-500/10 border border-indigo-500/20 px-2 py-1 text-[10px] font-mono text-indigo-300">{op}</span>
              ))}
              {result.clipping_risk && !result.clipping_prevented && (
                <span className="rounded-lg bg-amber-500/10 border border-amber-500/30 px-2 py-1 text-[10px] font-mono text-amber-300 flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" /> clipping risk (peak {result.output_peak.toFixed(3)})
                </span>
              )}
              {result.clipping_prevented && (
                <span className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-2 py-1 text-[10px] font-mono text-emerald-300 flex items-center gap-1">
                  <CheckCircle className="h-3 w-3" /> clipping prevented
                </span>
              )}
              <span className="rounded-lg bg-white/5 border border-white/10 px-2 py-1 text-[10px] font-mono text-slate-400">
                {result.input_duration_s.toFixed(2)}s → {result.output_duration_s.toFixed(2)}s
              </span>
              <span className="rounded-lg bg-white/5 border border-white/10 px-2 py-1 text-[10px] font-mono text-slate-400">
                peak {result.output_peak.toFixed(3)} | RMS {result.output_rms.toFixed(3)}
              </span>
            </div>
          )}

          {/* SIGNAL RESULT */}
          {state.status === "success" && state.waveform && (
            <SignalComparison
              originalWaveform={state.waveform}
              processedWaveform={procWaveform || null}
              originalSpectrum={state.spectrum!}
              processedSpectrum={procSpectrum || null}
              originalSpectrogram={state.spectrogram}
              processedSpectrogram={procSpectrogram || null}
              originalAudioUrl={state.objectUrl!}
              processedAudioUrl={procAudioUrl || null}
              originalDuration={state.metadata.duration_seconds}
              processedDuration={result?.output_duration_s || state.metadata.duration_seconds}
              processedLabel="Processed"
            />
          )}
        </div>
      </div>
    </div>
  );
}

const SR_DEFAULT = 44100;

// ---------------------------------------------------------------------------
// Math callout card
// ---------------------------------------------------------------------------

function OperationMathCard({
  gainParams, speedParams, stretchParams, pitchParams,
  effectType, tremoloP, ringModP, delayP, chorusP, distortionP, timbreP, reverbP
}: {
  gainParams: GainParams; speedParams: SpeedParams; stretchParams: StretchParams;
  pitchParams: PitchParams; effectType: EffectType;
  tremoloP: TremoloParams; ringModP: RingModParams; delayP: DelayParams;
  chorusP: ChorusParams; distortionP: DistortionParams; timbreP: TimbreParams; reverbP: ReverbParams;
}) {
  const equations: { label: string; eq: string; note?: string }[] = [];

  if (Math.abs(gainParams.value) > 1e-9 || gainParams.mode !== "db") {
    if (gainParams.mode === "db")
      equations.push({ label: "Gain", eq: `g = 10^(${gainParams.value.toFixed(1)} / 20) ≈ ${Math.pow(10, gainParams.value / 20).toFixed(3)}`, note: "y[n] = g · x[n]" });
    else if (gainParams.mode === "linear")
      equations.push({ label: "Gain", eq: `y[n] = ${gainParams.value.toFixed(3)} · x[n]` });
    else if (gainParams.mode === "peak")
      equations.push({ label: "Peak Norm", eq: `g = ${gainParams.value.toFixed(3)} / max(|x|)` });
    else
      equations.push({ label: "RMS Norm", eq: `g = ${gainParams.value.toFixed(3)} / RMS(x)` });
  }

  if (Math.abs(speedParams.speed - 1.0) > 0.01)
    equations.push({ label: "Speed", eq: `${speedParams.speed.toFixed(2)}×`, note: `time stretch = ${(1/speedParams.speed).toFixed(2)}×, pitch preserved` });

  if (Math.abs(stretchParams.stretch - 1.0) > 0.01)
    equations.push({ label: "Stretch", eq: `${stretchParams.stretch.toFixed(2)}×`, note: "duration expands/contracts, frequency remains approximately fixed" });

  if (Math.abs(pitchParams.semitones) > 0.1) {
    const r = Math.pow(2, pitchParams.semitones / 12);
    equations.push({ label: "Pitch Shift", eq: `r = 2^(${pitchParams.semitones.toFixed(1)}/12) ≈ ${r.toFixed(4)}`, note: "time stretch → raw resampling (duration preserved)" });
  }

  if (effectType === "tremolo")
    equations.push({ label: "Tremolo", eq: `a[n] = (1 − ${tremoloP.depth}) + ${tremoloP.depth} · (1 + sin(2π · ${tremoloP.rate_hz}Hz · n/Fs)) / 2`, note: "y[n] = x[n] · a[n]" });
  else if (effectType === "ring_modulation")
    equations.push({ label: "Ring Mod", eq: `y[n] = x[n] · cos(2π · ${ringModP.carrier_hz}Hz · n/Fs)`, note: "Sidebands at f ± carrier" });
  else if (effectType === "delay")
    equations.push({ label: "Delay", eq: `y[n] = x[n] + ${delayP.mix}·${delayP.feedback}·y[n − D]`, note: `D = ${delayP.delay_ms}ms` });
  else if (effectType === "chorus")
    equations.push({ label: "Chorus", eq: `D[n] = ${chorusP.base_delay_ms} + ${chorusP.depth_ms}·sin(2π·${chorusP.rate_hz}Hz·n/Fs) ms`, note: "Linear interpolation, mix = " + chorusP.mix });
  else if (effectType === "soft_distortion")
    equations.push({ label: "Distortion", eq: `y[n] = tanh(${distortionP.drive}·x[n]) / tanh(${distortionP.drive})`, note: "Odd harmonics generated" });
  else if (effectType === "timbre")
    equations.push({ label: "Spectral Tilt", eq: `voice color / spectral balance`, note: `tilt = ${timbreP.tilt.toFixed(2)}` });
  else if (effectType === "reverb")
    equations.push({ label: "Schroeder Reverb", eq: `comb filters → all-pass diffusion`, note: `room = ${reverbP.room_size.toFixed(2)}` });

  if (equations.length === 0) return null;

  return (
    <div className="rounded-2xl border border-indigo-500/10 bg-indigo-950/30 p-5">
      <div className="flex items-center gap-2 mb-3">
        <Info className="h-4 w-4 text-indigo-400/70" />
        <span className="text-[10px] font-bold tracking-widest text-indigo-400/70 uppercase">Active Mathematics</span>
      </div>
      <div className="space-y-2">
        {equations.map((e, i) => (
          <div key={i} className="flex flex-col gap-0.5">
            <span className="text-[10px] font-bold text-slate-500 uppercase">{e.label}</span>
            <code className="text-xs text-indigo-200 font-mono">{e.eq}</code>
            {e.note && <span className="text-[10px] text-slate-600">{e.note}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
