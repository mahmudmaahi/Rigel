"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { usePlayground } from "@/contexts/playground-context";
import {
  applyDenoise,
  fetchSpectrum,
  analyzeAudio,
  type DenoiseMethod,
  type DenoiseParams,
  type SpectrumData,
} from "@/lib/api";
import { SignalComparison } from "@/components/audio/signal-comparison";
import { Loader2, Music, CheckCircle, AlertCircle, Info, ChevronUp, ChevronDown, FileAudio } from "lucide-react";

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const DENOISE_METHODS: { value: DenoiseMethod; label: string; shortDesc: string }[] = [
  { value: "spectral_subtraction", label: "Spectral Subtraction", shortDesc: "Basic power subtraction" },
  { value: "wiener", label: "DD Wiener", shortDesc: "Decision-directed filter" },
  { value: "logmmse", label: "Log-MMSE", shortDesc: "Optimal envelope tracking" },
  { value: "imcra", label: "IMCRA + OM-LSA", shortDesc: "Advanced noise & speech tracking" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function NumInput({
  id, label, value, onChange, min, max, step, hint,
}: {
  id: string; label: string; value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number; hint?: string;
}) {
  const handleInc = () => {
    const val = value + (step ?? 0.1);
    if (max !== undefined && val > max) return;
    onChange(Number(val.toFixed(3)));
  };
  const handleDec = () => {
    const val = value - (step ?? 0.1);
    if (min !== undefined && val < min) return;
    onChange(Number(val.toFixed(3)));
  };

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-[11px] font-bold tracking-widest text-slate-500 uppercase ml-1">
        {label}
      </label>
      <div className="relative flex items-center">
        <input
          id={id}
          type="number"
          min={min}
          max={max}
          step={step ?? 0.1}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="
            w-full rounded-2xl border border-white/[0.08] bg-[#0d0d1a]
            pl-4 pr-[4.5rem] py-2.5 text-sm text-white font-mono
            focus:border-indigo-500/40 focus:outline-none focus:ring-1 focus:ring-indigo-500/20
            hover:border-white/15 transition-colors [appearance:textfield]
            [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none
          "
        />
        <div className="absolute right-2 flex items-center gap-1">
          <div className="flex flex-col -gap-0.5">
            <button
              type="button"
              tabIndex={-1}
              onClick={handleInc}
              className="text-slate-500 hover:text-indigo-300 hover:bg-white/[0.05] rounded p-1 transition-colors"
            >
              <ChevronUp className="h-3 w-3" strokeWidth={3} />
            </button>
            <button
              type="button"
              tabIndex={-1}
              onClick={handleDec}
              className="text-slate-500 hover:text-indigo-300 hover:bg-white/[0.05] rounded p-1 transition-colors"
            >
              <ChevronDown className="h-3 w-3" strokeWidth={3} />
            </button>
          </div>
        </div>
      </div>
      {hint && <p className="text-xs text-slate-600 ml-1 mt-0.5">{hint}</p>}
    </div>
  );
}

function MethodGrid({
  value, onChange,
}: {
  value: DenoiseMethod; onChange: (v: DenoiseMethod) => void;
}) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-3">
      {DENOISE_METHODS.map((ft) => {
        const active = ft.value === value;
        return (
          <button
            key={ft.value}
            onClick={() => onChange(ft.value)}
            className={[
              "flex flex-1 min-w-[140px] max-w-[200px] flex-col items-center justify-center gap-1.5 rounded-2xl border px-3 py-4 transition-all duration-150 text-center",
              active
                ? "border-indigo-500/50 bg-indigo-500/15 shadow-[0_0_12px_rgba(99,102,241,0.2)]"
                : "border-white/[0.06] bg-white/[0.02] hover:border-white/12 hover:bg-white/[0.04]",
            ].join(" ")}
          >
            <span className={["text-sm font-bold tracking-wide", active ? "text-indigo-200" : "text-slate-300"].join(" ")}>
              {ft.label}
            </span>
            <span className="text-xs text-slate-500 mt-1">{ft.shortDesc}</span>
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page Component
// ---------------------------------------------------------------------------

export default function EnhancementPage() {
  const { state, setProcessedAudio, setFilteredWaveform, setFilteredSpectrum, uploadFile } = usePlayground();


  const [processedUrl, setProcessedUrl] = useState<string | undefined>(undefined);

  const [method, setMethod] = useState<DenoiseMethod>("spectral_subtraction");
  const [alpha, setAlpha] = useState(1.0);
  const [beta, setBeta] = useState(0.01);
  const [alphaDd, setAlphaDd] = useState(0.98);
  const [gMin, setGMin] = useState(0.01);

  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const processedAudio = state.status === "success" ? state.processedAudio : null;

  useEffect(() => {
    if (processedAudio) {
      const url = URL.createObjectURL(processedAudio);
      setTimeout(() => setProcessedUrl(url), 0);
      return () => URL.revokeObjectURL(url);
    } else {
      setTimeout(() => setProcessedUrl(undefined), 0);
    }
  }, [processedAudio]);

  const handleMethodChange = (newMethod: DenoiseMethod) => {
    setMethod(newMethod);
    if (newMethod === "spectral_subtraction") {
      setAlpha(1.0);
      setBeta(0.01);
    } else if (newMethod === "wiener" || newMethod === "logmmse") {
      setAlphaDd(0.98);
    } else if (newMethod === "imcra") {
      setAlphaDd(0.98);
      setGMin(0.01);
    }
  };

  // Run Denoising
  const handleDenoise = async () => {
    if (state.status !== "success") return;

    setIsProcessing(true);
    setErrorMsg(null);
    setSuccessMsg(null);



    try {
      const params: DenoiseParams = { method };
      if (method === "spectral_subtraction") {
        params.alpha = alpha;
        params.beta = beta;
      } else if (method === "wiener" || method === "logmmse") {
        params.alphaDd = alphaDd;
      } else if (method === "imcra") {
        params.alphaDd = alphaDd;
        params.gMin = gMin;
      }

      const processedBlob = await applyDenoise(state.file, params);
      setProcessedAudio(processedBlob);

      const fileForAnalysis = new File([processedBlob], "processed.wav", { type: "audio/wav" });

      const [analyzeRes, spectrumRes] = await Promise.all([
        analyzeAudio(fileForAnalysis),
        fetchSpectrum(fileForAnalysis),
      ]);

      setFilteredWaveform(analyzeRes.waveform);
      setFilteredSpectrum(spectrumRes.spectrum);

      setSuccessMsg("Denoising complete.");
    } catch (err) {
      console.error("Denoising error:", err);
      setErrorMsg(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
      setIsProcessing(false);
    }
  };

  // No early return here so we can pre-visualize the enhancement controls!

  const hasProcessedData = state.status === "success" && state.processedAudio && state.filteredWaveform && state.filteredSpectrum;

  return (
    <div className="flex flex-col gap-6 lg:gap-10 max-w-[100rem] mx-auto pb-20">
      <div className="flex flex-col items-center justify-center text-center mt-4 mb-10">
        <h1 className="flex items-center justify-center gap-3 text-3xl font-bold tracking-tight sm:text-4xl text-white">
          <Music className="h-8 w-8 text-indigo-400" />
          Noise Removal
        </h1>
        <p className="mt-4 text-base text-slate-400 max-w-2xl mx-auto">
          Classical statistical offline denoising. Track and remove noise using frequency-domain estimation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8 min-w-0">
        
        {/* Left Column: Processing Method */}
        <div className="flex flex-col gap-6 min-w-0">

          {state.status !== "success" && (
            <div className="p-5 border border-dashed border-white/10 rounded-2xl bg-white/[0.02] text-center flex flex-col items-center justify-center gap-3">
              <div className="flex items-center justify-center h-10 w-10 rounded-full bg-white/5">
                <FileAudio className="h-4 w-4 text-slate-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-300">No audio loaded</p>
                <p className="text-xs text-slate-500 mt-1">Upload a WAV file from the sidebar to begin.</p>
              </div>
            </div>
          )}

          <div className="rounded-3xl border border-white/10 bg-[#0c0c16]/80 p-6 shadow-2xl backdrop-blur-sm h-full">
            <h2 className="text-sm font-bold tracking-widest text-slate-400 uppercase mb-5">
              Processing Method
            </h2>
            <MethodGrid value={method} onChange={handleMethodChange} />
          </div>
        </div>

        {/* Right Column: Parameters */}
        <div className="flex flex-col gap-6 min-w-0">
          <div className="rounded-3xl border border-white/10 bg-[#0c0c16]/80 p-6 shadow-2xl backdrop-blur-sm h-full">
            <h2 className="text-sm font-bold tracking-widest text-slate-400 uppercase mb-5">
              Parameters
            </h2>

            <div className="flex flex-col gap-6">
              {method === "spectral_subtraction" && (
                <>
                  <NumInput
                    id="alpha" label="Oversubtraction Factor (α)"
                    value={alpha} onChange={setAlpha}
                    min={1.0} max={10.0} step={0.1}
                    hint="Amount of noise to subtract. > 1 reduces residual noise but adds distortion."
                  />
                  <NumInput
                    id="beta" label="Spectral Floor (β)"
                    value={beta} onChange={setBeta}
                    min={0.001} max={0.1} step={0.001}
                    hint="Minimum permitted power ratio to prevent musical noise artifacts."
                  />
                </>
              )}

              {(method === "wiener" || method === "logmmse" || method === "imcra") && (
                <NumInput
                  id="alpha_dd" label="Decision-Directed Smoothing (α_DD)"
                  value={alphaDd} onChange={setAlphaDd}
                  min={0.9} max={0.999} step={0.01}
                  hint="A-priori SNR smoothing. Higher values reduce musical noise but blur transients."
                />
              )}

              {method === "imcra" && (
                <NumInput
                  id="g_min" label="OM-LSA Gain Floor (G_min)"
                  value={gMin} onChange={setGMin}
                  min={0.001} max={0.1} step={0.01}
                  hint="Minimum gain applied when speech is absent."
                />
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 pt-6 border-t border-white/[0.06] max-w-2xl mx-auto w-full">
        {errorMsg && (
          <div className="mb-4 flex items-start gap-3 rounded-xl bg-red-500/10 p-4 border border-red-500/20">
            <AlertCircle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
            <p className="text-sm text-red-200">{errorMsg}</p>
          </div>
        )}
        {successMsg && !isProcessing && (
          <div className="mb-4 flex items-start gap-3 rounded-xl bg-cyan-500/10 p-4 border border-cyan-500/20">
            <CheckCircle className="h-5 w-5 text-cyan-400 shrink-0 mt-0.5" />
            <p className="text-sm text-cyan-200">{successMsg}</p>
          </div>
        )}

        <button
          onClick={handleDenoise}
          disabled={isProcessing || state.status !== "success"}
          className="w-full flex items-center justify-center gap-2 rounded-2xl bg-indigo-600 px-6 py-4 text-base font-bold text-white transition-all hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_30px_rgba(79,70,229,0.5)]"
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              Processing Audio...
            </>
          ) : (
            <>
              <Music className="h-5 w-5" />
              Run Denoising
            </>
          )}
        </button>
      </div>

      {/* SIGNAL RESULT */}
      {state.status === "success" && state.waveform && (
        <SignalComparison
          originalWaveform={state.waveform}
          processedWaveform={state.filteredWaveform || null}
          originalSpectrum={state.spectrum!}
          processedSpectrum={state.filteredSpectrum || null}
          originalAudioUrl={state.objectUrl!}
          processedAudioUrl={processedUrl || null}
          originalDuration={state.metadata.duration_seconds}
          processedDuration={state.metadata.duration_seconds}
          processedLabel="Processed"
        />
      )}
    </div>
  );
}
