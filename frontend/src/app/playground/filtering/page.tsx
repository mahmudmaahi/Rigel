"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { usePlayground } from "@/contexts/playground-context";
import {
  applyFilter,
  fetchFilterResponse,
  fetchSpectrum,
  analyzeAudio,
  type FilterType,
  type FilterFamily,
  type FilterParams,
  type FilterResponseData,
  type SpectrumData,
} from "@/lib/api";
import { SlidersHorizontal, Loader2, CheckCircle, AlertCircle, Info, ChevronUp, ChevronDown, FileAudio } from "lucide-react";
import { SignalComparison } from "@/components/audio/signal-comparison";

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const FILTER_TYPES: { value: FilterType; label: string; shortDesc: string }[] = [
  { value: "lowpass",  label: "Low-pass",  shortDesc: "Below cutoff" },
  { value: "highpass", label: "High-pass", shortDesc: "Above cutoff" },
  { value: "bandpass", label: "Band-pass", shortDesc: "Within band" },
  { value: "bandstop", label: "Band-stop", shortDesc: "Notch / reject" },
  { value: "peaking",  label: "Peaking EQ", shortDesc: "Boost or cut" },
];

const FAMILIES: { value: FilterFamily; label: string; note: string; isIIR: boolean; isFIR: boolean; isPeaking?: boolean }[] = [
  { value: "butterworth", label: "Butterworth",    note: "Maximally flat passband",                isIIR: true,  isFIR: false },
  { value: "chebyshev1",  label: "Chebyshev I",    note: "Equiripple passband, sharper roll-off",  isIIR: true,  isFIR: false },
  { value: "chebyshev2",  label: "Chebyshev II",   note: "Flat passband, equiripple stopband",     isIIR: true,  isFIR: false },
  { value: "elliptic",    label: "Elliptic",        note: "Sharpest roll-off, ripple both bands",  isIIR: true,  isFIR: false },
  { value: "bessel",      label: "Bessel",          note: "Linear phase, smooth time response",    isIIR: true,  isFIR: false },
  { value: "fir_window",  label: "FIR Window",      note: "Always stable, exact linear phase",     isIIR: false, isFIR: true  },
  { value: "fir_remez",   label: "Parks-McClellan", note: "Equiripple FIR — optimal minimax",      isIIR: false, isFIR: true  },
];

const FIR_WINDOWS = ["hamming", "hann", "blackman", "bartlett", "kaiser"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildApiParams(
  filterType: FilterType,
  family: FilterFamily,
  order: number,
  cutoffHz: number,
  lowHz: number,
  highHz: number,
  rippleDb: number,
  attenuationDb: number,
  firWindow: string,
  transitionBw: number,
  centerHz: number,
  gainDb: number,
  qFactor: number,
): FilterParams & Record<string, unknown> {
  const isBand = filterType === "bandpass" || filterType === "bandstop";
  const isPeaking = filterType === "peaking";

  return {
    filterType,
    family,
    order,
    ...(isPeaking  ? { centerHz, gainDb, qFactor } :
        isBand      ? { lowHz, highHz } :
                      { cutoffHz }),
    rippleDb,
    attenuationDb,
    firWindow,
    transitionBandwidthHz: transitionBw,
  };
}

// ---------------------------------------------------------------------------
// Styled pill-button group
// ---------------------------------------------------------------------------

function PillGroup<T extends string>({
  options,
  value,
  onChange,
  size = "sm",
}: {
  options: { value: T; label: string; note?: string }[];
  value: T;
  onChange: (v: T) => void;
  size?: "sm" | "xs";
}) {
  return (
    <div className="flex flex-wrap justify-center gap-2.5">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            title={o.note}
            className={[
              "rounded-xl border transition-all duration-150 font-medium",
              size === "xs" ? "py-1.5 px-4 text-xs" : "py-2 px-5 text-sm",
              active
                ? "border-indigo-500/50 bg-indigo-500/20 text-indigo-200 shadow-[0_0_8px_rgba(99,102,241,0.3)]"
                : "border-white/[0.07] bg-white/[0.03] text-slate-400 hover:border-white/15 hover:text-slate-200",
            ].join(" ")}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Styled number input
// ---------------------------------------------------------------------------

function NumInput({
  id, label, value, onChange, min, max, step, unit, hint,
}: {
  id: string; label: string; value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number; unit?: string; hint?: string;
}) {
  const handleInc = () => {
    const val = value + (step ?? 1);
    if (max !== undefined && val > max) return;
    onChange(Number(val.toFixed(3)));
  };
  const handleDec = () => {
    const val = value - (step ?? 1);
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
          step={step ?? 1}
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
          {unit && <span className="mr-1.5 text-sm text-slate-500 font-mono select-none">{unit}</span>}
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

// ---------------------------------------------------------------------------
// Filter type card grid
// ---------------------------------------------------------------------------

function FilterTypeGrid({
  value, onChange,
}: {
  value: FilterType; onChange: (v: FilterType) => void;
}) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-3">
      {FILTER_TYPES.map((ft) => {
        const active = ft.value === value;
        return (
          <button
            key={ft.value}
            onClick={() => onChange(ft.value)}
            className={[
              "flex flex-1 min-w-[100px] max-w-[150px] flex-col items-center justify-center gap-1.5 rounded-2xl border px-3 py-4 transition-all duration-150 text-center",
              active
                ? "border-indigo-500/50 bg-indigo-500/15 shadow-[0_0_12px_rgba(99,102,241,0.2)]"
                : "border-white/[0.06] bg-white/[0.02] hover:border-white/12 hover:bg-white/[0.04]",
            ].join(" ")}
          >
            <span className={["text-base font-bold tracking-wide", active ? "text-indigo-200" : "text-slate-300"].join(" ")}>
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
// Frequency response SVG chart
// ---------------------------------------------------------------------------

function FrequencyResponseChart({
  data, sampleRateHz,
}: {
  data: FilterResponseData; sampleRateHz: number;
}) {
  const W = 540, H = 320;
  const PAD = { top: 10, right: 12, bottom: 34, left: 46 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const nyquist = sampleRateHz / 2;

  const toX = (f: number) =>
    PAD.left + (Math.log10(Math.max(f, 1)) - Math.log10(20)) /
    (Math.log10(nyquist) - Math.log10(20)) * plotW;

  const DB_MIN = -80, DB_MAX = 6;
  const toY = (db: number) =>
    PAD.top + (1 - (Math.min(Math.max(db, DB_MIN), DB_MAX) - DB_MIN) / (DB_MAX - DB_MIN)) * plotH;

  const freqs = data.frequencies_hz;
  const mags  = data.magnitude_db;

  const pts = freqs.map((f, i) => `${toX(f).toFixed(1)},${toY(mags[i]).toFixed(1)}`).join(" L ");
  const pathD = `M ${pts}`;
  const fillD = `M ${toX(freqs[0]||20).toFixed(1)},${(PAD.top+plotH).toFixed(1)} L ${pts} L ${toX(freqs[freqs.length-1]||nyquist).toFixed(1)},${(PAD.top+plotH).toFixed(1)} Z`;

  const markers: number[] = [];
  if (data.cutoff_hz)  markers.push(data.cutoff_hz);
  if (data.low_hz)     markers.push(data.low_hz);
  if (data.high_hz)    markers.push(data.high_hz);
  if (data.center_hz)  markers.push(data.center_hz);

  const xTicks = [30,50,100,200,500,1000,2000,5000,10000,20000].filter(f => f<nyquist && f>0);
  const yTicks = [-60,-40,-20,-6, 0];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
      <defs>
        <linearGradient id="fg1" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#818cf8" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#818cf8" stopOpacity="0" />
        </linearGradient>
        <clipPath id="plotClip">
          <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} />
        </clipPath>
      </defs>

      {/* Grid */}
      {yTicks.map(db => (
        <line key={db} x1={PAD.left} y1={toY(db)} x2={PAD.left+plotW} y2={toY(db)}
          stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
      ))}

      {/* Clip group */}
      <g clipPath="url(#plotClip)">
        <path d={fillD} fill="url(#fg1)" />
        <path d={pathD} fill="none" stroke="#818cf8" strokeWidth="1.8" />
        {markers.map(f => (
          <line key={f} x1={toX(f)} y1={PAD.top} x2={toX(f)} y2={PAD.top+plotH}
            stroke="rgba(167,139,250,0.5)" strokeWidth="1" strokeDasharray="3 3" />
        ))}
      </g>

      {/* −3 dB line */}
      <line x1={PAD.left} y1={toY(-3)} x2={PAD.left+plotW} y2={toY(-3)}
        stroke="rgba(251,191,36,0.2)" strokeWidth="1" strokeDasharray="6 4" />

      {/* Y labels */}
      {yTicks.map(db => (
        <text key={db} x={PAD.left-6} y={toY(db)+3.5} textAnchor="end"
          fontSize="10" fill="rgba(148,163,184,0.6)" fontFamily="monospace">
          {db}
        </text>
      ))}

      {/* X labels */}
      {xTicks.map(f => (
        <text key={f} x={toX(f)} y={PAD.top+plotH+14} textAnchor="middle"
          fontSize="9.5" fill="rgba(148,163,184,0.5)" fontFamily="monospace">
          {f>=1000?`${f/1000}k`:String(f)}
        </text>
      ))}

      <text x={PAD.left+plotW/2} y={H-2} textAnchor="middle" fontSize="10" fill="rgba(148,163,184,0.4)">
        Frequency (Hz)
      </text>
      <text x={10} y={PAD.top+plotH/2} textAnchor="middle" fontSize="10" fill="rgba(148,163,184,0.4)"
        transform={`rotate(-90,10,${PAD.top+plotH/2})`}>
        dB
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function FilteringPage() {
  const { state, setProcessedAudio, setFilteredSpectrum, setFilteredWaveform, uploadFile } = usePlayground();

  // Filter type & family
  const [filterType, setFilterType] = useState<FilterType>("lowpass");
  const [family, setFamily]         = useState<FilterFamily>("butterworth");

  // Frequency-selective params
  const [order, setOrder]           = useState(4);
  const [cutoffHz, setCutoffHz]     = useState(1000);
  const [lowHz, setLowHz]           = useState(500);
  const [highHz, setHighHz]         = useState(3000);

  // IIR extra params
  const [rippleDb, setRippleDb]         = useState(1.0);
  const [attenuationDb, setAttenuationDb] = useState(40.0);

  // FIR params
  const [firWindow, setFirWindow]       = useState("hamming");
  const [transitionBw, setTransitionBw] = useState(200);

  // Peaking EQ
  const [centerHz, setCenterHz]   = useState(1000);
  const [gainDb, setGainDb]       = useState(0);
  const [qFactor, setQFactor]     = useState(1.0);

  // UI state
  const [isApplying, setIsApplying]   = useState(false);
  const [isFetching, setIsFetching]   = useState(false);
  const [applyError, setApplyError]   = useState<string | null>(null);
  const [applySuccess, setApplySuccess] = useState(false);
  const [freqResp, setFreqResp]       = useState<FilterResponseData | null>(null);
  const [respError, setRespError]     = useState<string | null>(null);
  
  const processedAudio = state.status === "success" ? state.processedAudio : null;
  const filteredSpectrum = state.status === "success" ? state.filteredSpectrum : null;
  const filteredWaveform = state.status === "success" ? state.filteredWaveform : null;
  const [processedUrl, setProcessedUrl] = useState<string | null>(null);

  const sampleRateHz = state.status === "success" ? state.metadata.sample_rate_hz : 44100;
  const nyquist = sampleRateHz / 2;

  const isBand    = filterType === "bandpass" || filterType === "bandstop";
  const isPeaking = filterType === "peaking";
  const isFIR     = family === "fir_window" || family === "fir_remez";
  const needsRipple = family === "chebyshev1" || family === "elliptic";
  const needsAtten  = family === "chebyshev2" || family === "elliptic";

  // When peaking is selected, force family to butterworth (peaking uses biquad)
  useEffect(() => {
    if (isPeaking && isFIR) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setFamily("butterworth");
    }
  }, [isPeaking, isFIR]);

  const params = buildApiParams(
    filterType, family, isFIR ? Math.max(order, 20) : order,
    cutoffHz, lowHz, highHz,
    rippleDb, attenuationDb,
    firWindow, transitionBw,
    centerHz, gainDb, qFactor,
  );

  // Debounced frequency response fetch
  const fetchResp = useCallback(async () => {
    setIsFetching(true);
    setRespError(null);
    try {
      const resp = await fetchFilterResponse(sampleRateHz, params as FilterParams);
      setFreqResp(resp);
    } catch (e) {
      setRespError(e instanceof Error ? e.message : "Preview failed.");
    } finally {
      setIsFetching(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sampleRateHz, filterType, family, order, cutoffHz, lowHz, highHz,
      rippleDb, attenuationDb, firWindow, transitionBw, centerHz, gainDb, qFactor]);

  useEffect(() => {
    const t = setTimeout(fetchResp, 400);
    return () => clearTimeout(t);
  }, [fetchResp]);

  // Clean up object URLs
  useEffect(() => {
    if (processedAudio) {
      const url = URL.createObjectURL(processedAudio);
      // eslint-disable-next-line
      setProcessedUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      // eslint-disable-next-line
      setProcessedUrl(null); 
    }
  }, [processedAudio]);

  // Apply
  const handleApply = async () => {
    if (state.status !== "success") return;
    setIsApplying(true);
    setApplyError(null);
    setApplySuccess(false);
    try {
      const blob = await applyFilter(state.file, params as FilterParams);
      setProcessedAudio(blob);
      setApplySuccess(true);
      
      // Fetch spectrum and waveform of the filtered audio
      const processedFile = new File([blob], "filtered.wav", { type: "audio/wav" });
      
      const [specResp, waveResp] = await Promise.all([
        fetchSpectrum(processedFile),
        analyzeAudio(processedFile)
      ]);
      
      setFilteredSpectrum(specResp.spectrum);
      setFilteredWaveform(waveResp.waveform);
    } catch (e) {
      setApplyError(e instanceof Error ? e.message : "Filtering failed.");
    } finally {
      setIsApplying(false);
    }
  };



  const activeFamilyInfo = FAMILIES.find(f => f.value === family);

  return (
    <div className="flex h-full flex-col gap-5 overflow-y-auto px-5 py-7 lg:px-8">

      {/* Header */}
      <div className="mb-10 text-center">
        <h1 className="flex items-center justify-center gap-3 text-3xl font-bold tracking-tight sm:text-4xl text-white">
          <SlidersHorizontal className="h-8 w-8 text-indigo-400" />
          Audio Filtering
        </h1>
        <p className="mt-4 text-base text-slate-400 max-w-2xl mx-auto">
          Classical IIR / FIR frequency-selective filtering <span className="text-white">+</span> parametric EQ.
          Zero-phase offline processing via <code className="text-indigo-300/70">sosfiltfilt</code> / <code className="text-indigo-300/70">filtfilt</code>.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1fr_1.4fr]">

        {/* ── Left: controls ── */}
        <div className="flex flex-col gap-6">

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

          {/* Group 1: Filter Type & Family */}
          <div className="flex flex-col gap-6 rounded-3xl border border-white/[0.08] bg-[#161625] p-6 lg:p-8 shadow-sm">
            <div>
              <h2 className="mb-5 text-[11px] font-bold tracking-widest text-slate-500 uppercase ml-1">Filter Type</h2>
              <FilterTypeGrid value={filterType} onChange={(v) => { setFilterType(v); setApplySuccess(false); }} />
            </div>
            
            {!isPeaking && (
              <div className="pt-6 border-t border-white/[0.06]">
                <h2 className="mb-5 text-[11px] font-bold tracking-widest text-slate-500 uppercase ml-1">Design Family</h2>
                <PillGroup
                  value={family}
                  onChange={(v) => { setFamily(v); setApplySuccess(false); }}
                  options={FAMILIES.map(f => ({ value: f.value, label: f.label }))}
                />
              </div>
            )}
          </div>

          {/* Group 2: Frequency controls */}
          <section className="rounded-3xl border border-white/[0.08] bg-[#161625] p-6 lg:p-8 shadow-sm">
            {isPeaking ? (
              <div className="grid grid-cols-3 gap-3">
                <NumInput id="center-hz" label="Centre Freq" value={centerHz}
                  onChange={v => { setCenterHz(v); setApplySuccess(false); }}
                  min={20} max={nyquist - 1} step={10} unit="Hz"
                  hint={`< ${nyquist} Hz`} />
                <NumInput id="gain-db" label="Gain" value={gainDb}
                  onChange={v => { setGainDb(v); setApplySuccess(false); }}
                  min={-24} max={24} step={0.5} unit="dB"
                  hint="− = cut, + = boost" />
                <NumInput id="q-factor" label="Q Factor" value={qFactor}
                  onChange={v => { setQFactor(v); setApplySuccess(false); }}
                  min={0.1} max={20} step={0.1} unit="Q"
                  hint="Higher Q = narrower" />
              </div>
            ) : isBand ? (
              <div className="grid grid-cols-2 gap-3">
                <NumInput id="low-hz" label="Low Edge" value={lowHz}
                  onChange={v => { setLowHz(v); setApplySuccess(false); }}
                  min={1} max={highHz - 1} step={10} unit="Hz" />
                <NumInput id="high-hz" label="High Edge" value={highHz}
                  onChange={v => { setHighHz(v); setApplySuccess(false); }}
                  min={lowHz + 1} max={nyquist - 1} step={10} unit="Hz"
                  hint={`< ${nyquist} Hz`} />
              </div>
            ) : (
              <NumInput id="cutoff-hz" label="Cutoff Frequency" value={cutoffHz}
                onChange={v => { setCutoffHz(v); setApplySuccess(false); }}
                min={1} max={nyquist - 1} step={10} unit="Hz"
                hint={`Valid: 1 – ${nyquist - 1} Hz (Nyquist = ${nyquist} Hz)`} />
            )}
          </section>

          {/* Group 3: Order & Extras */}
          <section className="flex flex-col gap-6 rounded-3xl border border-white/[0.08] bg-[#161625] p-6 lg:p-8 shadow-sm">
            <div>
              <NumInput
                id="filter-order"
                label={isFIR ? "FIR Taps − 1 (order)" : "Filter Order"}
                value={order}
                onChange={v => { setOrder(v); setApplySuccess(false); }}
                min={1}
                max={isFIR ? 512 : 20}
                step={isFIR ? 4 : 1}
                hint={
                  isFIR
                    ? `${order + 1} taps. More taps → sharper roll-off.`
                    : `Eff. zero-phase order = ${order * 2}. IIR max = 20.`
                }
              />
            </div>

            {(needsRipple || needsAtten) && (
              <div className="pt-6 border-t border-white/[0.06]">
                <h2 className="mb-5 text-[11px] font-bold tracking-widest text-slate-500 uppercase ml-1">Design Parameters</h2>
                <div className="grid grid-cols-2 gap-3">
                  {needsRipple && (
                    <NumInput id="ripple-db" label="Passband Ripple" value={rippleDb}
                      onChange={v => { setRippleDb(v); setApplySuccess(false); }}
                      min={0.01} max={10} step={0.1} unit="dB" />
                  )}
                  {needsAtten && (
                    <NumInput id="atten-db" label="Stopband Atten." value={attenuationDb}
                      onChange={v => { setAttenuationDb(v); setApplySuccess(false); }}
                      min={1} max={120} step={1} unit="dB" />
                  )}
                </div>
              </div>
            )}

            {isFIR && (
              <div className="pt-6 border-t border-white/[0.06]">
                <h2 className="mb-5 text-[11px] font-bold tracking-widest text-slate-500 uppercase ml-1">
                  {family === "fir_window" ? "Window Function" : "Transition Bandwidth"}
                </h2>
                {family === "fir_window" ? (
                  <PillGroup
                    value={firWindow as never}
                    onChange={v => { setFirWindow(v); setApplySuccess(false); }}
                    options={FIR_WINDOWS.map(w => ({ value: w, label: w.charAt(0).toUpperCase() + w.slice(1) }))}
                  />
                ) : (
                  <NumInput id="trans-bw" label="Transition Bandwidth" value={transitionBw}
                    onChange={v => { setTransitionBw(v); setApplySuccess(false); }}
                    min={10} max={5000} step={10} unit="Hz" />
                )}
              </div>
            )}
          </section>
        </div>

        {/* ── Right: viz + education ── */}
        <div className="flex flex-col gap-6">

          {/* Frequency response chart */}
          <section className="rounded-3xl border border-white/[0.06] bg-white/[0.02] p-6 lg:p-8 shadow-sm">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-[11px] font-bold tracking-widest text-slate-500 uppercase ml-1">
                Theoretical Frequency Response
              </h2>
              {isFetching && <Loader2 className="h-3 w-3 animate-spin text-indigo-400" />}
            </div>

            {respError ? (
              <div className="flex h-80 items-center justify-center rounded-xl border border-red-500/10 bg-red-500/5">
                <p className="text-xs text-red-400">{respError}</p>
              </div>
            ) : freqResp ? (
              <div className="h-80 w-full">
                <FrequencyResponseChart data={freqResp} sampleRateHz={sampleRateHz} />
              </div>
            ) : (
              <div className="flex h-80 items-center justify-center">
                <Loader2 className="h-5 w-5 animate-spin text-indigo-400/40" />
              </div>
            )}

            <div className="mt-2 flex flex-wrap gap-3 text-[10px] text-slate-500 font-mono">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-px w-5 bg-indigo-400/70"></span>Response
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-px w-5 border-t border-dashed border-violet-400/50"></span>Cutoff / edge
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-px w-5 border-t border-dashed border-yellow-400/30"></span>−3 dB
              </span>
            </div>
          </section>

          {/* Educational card — dynamic family guide */}
          <section className="rounded-3xl border border-white/[0.08] bg-[#161625] p-6 lg:p-8 text-sm text-slate-300 leading-relaxed shadow-sm">
            <p className="mb-4 text-[11px] font-bold tracking-widest text-slate-500 uppercase border-b border-white/[0.05] pb-4">Filter Family Guide</p>
            <div className="min-h-[4rem]">
              {!isPeaking && family === "butterworth" && <p><strong className="text-white">Butterworth:</strong> Maximally flat magnitude response in the passband, no passband ripple. The most common choice for a clean, general-purpose filter without coloration, but has a slower transition to the stopband.</p>}
              {!isPeaking && family === "chebyshev1" && <p><strong className="text-white">Chebyshev Type I:</strong> Introduces passband ripple to achieve a sharper transition to the stopband than Butterworth for a given order. Useful when separation is needed and slight magnitude ripple is acceptable.</p>}
              {!isPeaking && family === "chebyshev2" && <p><strong className="text-white">Chebyshev Type II:</strong> Keeps a flat passband but introduces equiripple in the stopband. Achieves a sharper transition than Butterworth. Good when you want to avoid passband coloration but need tight filtering.</p>}
              {!isPeaking && family === "elliptic" && <p><strong className="text-white">Elliptic (Cauer):</strong> The sharpest possible transition between passband and stopband for a given order. It allows ripples in both bands. Used when the absolute narrowest transition bandwidth is required.</p>}
              {!isPeaking && family === "bessel" && <p><strong className="text-white">Bessel:</strong> Optimized for maximally flat group delay, preserving the wave shape of signals in the passband. Excellent for time-domain transients, but has a very gradual frequency roll-off.</p>}
              {!isPeaking && family === "fir_window" && <p><strong className="text-white">FIR Window:</strong> Windowed Finite Impulse Response design. FIR filters are unconditionally stable and provide exact linear phase. The choice of window trades off transition width vs stopband attenuation.</p>}
              {!isPeaking && family === "fir_remez" && <p><strong className="text-white">Parks-McClellan:</strong> Optimal equiripple FIR filter design using the Remez exchange algorithm. It minimizes the maximum error (minimax) in the specified bands, providing the most efficient FIR filter for given specifications.</p>}
              {isPeaking && <p><strong className="text-white">Peaking EQ (Biquad):</strong> An Infinite Impulse Response biquad filter that boosts or cuts a specific centre frequency while leaving the rest of the spectrum unchanged. Common in parametric equalizers.</p>}
            </div>
          </section>

          {/* SOS note */}
          {!isFIR && (
            <section className="rounded-3xl border border-white/[0.08] bg-[#161625] p-6 lg:p-8 text-[13px] text-slate-400 leading-relaxed shadow-sm">
              <p className="text-slate-500 font-bold uppercase tracking-widest text-[11px] mb-3">Why SOS for IIR?</p>
              <p>
                Direct-form polynomials of high-order IIR filters cause catastrophic floating-point cancellation. Rigel uses Second-Order Sections (SOS) factorisation to cascade well-conditioned biquads. All IIR families are processed with <code className="text-indigo-300/70">sosfiltfilt</code> for zero-phase offline output.
              </p>
            </section>
          )}
          
          {/* Group 4: Actions (Moved from left column) */}
          <div className="flex flex-col gap-3 mt-auto pt-2">
            {applyError && (
              <div className="flex items-start gap-2 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                {applyError}
              </div>
            )}
            {applySuccess && !isApplying && (
              <div className="mb-4 flex items-start gap-3 rounded-xl bg-cyan-500/10 p-4 border border-cyan-500/20">
                <CheckCircle className="h-5 w-5 text-cyan-400 shrink-0 mt-0.5" />
                <p className="text-cyan-200">Filter Applied.</p>
              </div>
            )}
            <div className="flex flex-col sm:flex-row gap-4 w-full">
              <button
                onClick={handleApply}
                disabled={isApplying || state.status !== "success"}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl
                  border border-indigo-500/30 bg-gradient-to-r from-indigo-600/20 to-violet-600/20
                  px-4 py-4 text-[15px] font-bold text-indigo-100
                  hover:border-indigo-500/50 hover:from-indigo-600/30 hover:to-violet-600/30 hover:text-white
                  disabled:cursor-not-allowed disabled:opacity-40 transition-all duration-200 uppercase tracking-widest shadow-[0_0_20px_rgba(79,70,229,0.15)]"
              >
                {isApplying
                  ? <><Loader2 className="h-5 w-5 animate-spin" /> Filtering…</>
                  : <><SlidersHorizontal className="h-5 w-5" /> Apply Filter</>}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── SIGNAL RESULT ── */}
      {state.status === "success" && state.waveform && (
        <SignalComparison
          originalWaveform={state.waveform}
          processedWaveform={filteredWaveform}
          originalSpectrum={state.spectrum!}
          processedSpectrum={filteredSpectrum}
          originalAudioUrl={state.objectUrl!}
          processedAudioUrl={processedUrl}
          originalDuration={state.metadata.duration_seconds}
          processedDuration={state.metadata.duration_seconds}
          processedLabel="Filtered"
        />
      )}
    </div>
  );
}
