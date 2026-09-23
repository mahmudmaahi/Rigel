"use client";

import { useEffect, useRef, useState } from "react";
import { Waves, Loader2 } from "lucide-react";
import { usePlayground } from "@/contexts/playground-context";
import { WaveformViewer } from "@/components/audio/waveform-viewer";
import { AudioPlayer } from "@/components/audio/audio-player";
import { SpectrumViewer } from "@/components/audio/spectrum-viewer";
import { SpectrogramViewer } from "@/components/audio/spectrogram-viewer";

const WINDOW_OPTIONS = [
  { value: "rectangular", label: "Rectangular" },
  { value: "hamming", label: "Hamming" },
  { value: "hann", label: "Hann" },
  { value: "blackman", label: "Blackman" },
  { value: "kaiser", label: "Kaiser" },
  { value: "bartlett", label: "Bartlett" },
  { value: "welch", label: "Welch" },
];

function MetaBadge({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="glass flex items-center gap-2 rounded-xl px-3.5 py-2 border border-white/5 text-xs shadow-sm">
      <span className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
        {label}
      </span>
      <span className="h-3 w-px bg-white/10" />
      <span className={mono ? "font-mono font-medium text-indigo-200" : "font-medium text-white"}>
        {value}
      </span>
    </div>
  );
}

export default function AnalysisPage() {
  const { state, updateSpectrogramWindow } = usePlayground();
  
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTarget, setSeekTarget] = useState<number | undefined>(undefined);
  const [isSpectrogramLoading, setIsSpectrogramLoading] = useState(false);
  
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (state.status !== "success") {
    return null; // Handled by layout
  }

  const { metadata, waveform, spectrum, spectrogram, objectUrl, spectrogramWindow } = state;

  function handleWaveformSeek(timeSeconds: number) {
    setCurrentTime(timeSeconds);
    setSeekTarget(timeSeconds);
  }

  function handleTimeUpdate(time: number) {
    setCurrentTime(time);
  }

  function handleAudioEnded() {
    setCurrentTime(0);
  }

  return (
    <div className="glass-strong relative overflow-hidden rounded-3xl p-8 sm:p-10 shadow-2xl shadow-black/40 border border-white/10 animate-fade-in-up">
      {/* Header */}
      <div className="mb-10 text-center border-b border-white/5 pb-8">
        <h1 className="flex items-center justify-center gap-3 text-3xl font-bold tracking-tight sm:text-4xl text-white">
          <Waves className="h-8 w-8 text-indigo-400" />
          Audio Analysis
        </h1>
        <div className="mt-4 flex items-center justify-center gap-4">
          <p className="font-mono text-sm text-slate-400">{metadata.filename}</p>
          <span className="h-4 w-px bg-white/10" />
          <div className="font-mono text-[11px] uppercase tracking-wider text-indigo-300 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-2.5 py-1">
            {metadata.channels === 1 ? "MONO" : "STEREO L/R"} • {waveform.n_bins} BINS
          </div>
        </div>
      </div>

      {/* Metadata Badges */}
      <div className="mb-8 flex flex-wrap gap-2.5">
        <MetaBadge label="Format" value={metadata.format.toUpperCase()} />
        <MetaBadge
          label="Sample Rate"
          value={`${metadata.sample_rate_hz.toLocaleString()} Hz`}
          mono
        />
        <MetaBadge
          label="Channels"
          value={metadata.channels === 1 ? "Mono" : "Stereo (L/R)"}
        />
        <MetaBadge
          label="Duration"
          value={`${metadata.duration_seconds.toFixed(3)} s`}
          mono
        />
        {metadata.bit_depth != null && (
          <MetaBadge label="Bit Depth" value={`${metadata.bit_depth}-bit`} mono />
        )}
        <MetaBadge label="Data Type" value={metadata.sample_summary.dtype} mono />
      </div>

      {/* Waveform Visualization Canvas */}
      <div className="mb-6 rounded-2xl border border-white/[0.08] bg-[#161625] p-6 shadow-sm">
        <WaveformViewer
          waveform={waveform}
          currentTime={currentTime}
          onSeek={handleWaveformSeek}
        />
      </div>

      {/* Audio Player Transport */}
      <div className="mb-8">
        <AudioPlayer
          src={objectUrl}
          duration={metadata.duration_seconds}
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleAudioEnded}
          seekTarget={seekTarget}
        />
      </div>

      {/* Advanced Analysis Grid */}
      <div className="flex flex-col gap-6">
        {/* Frequency Spectrum */}
        {spectrum && (
          <div className="rounded-2xl border border-white/[0.08] bg-[#161625] p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h4 className="font-mono text-xs uppercase tracking-widest text-slate-400">
                  Frequency Spectrum
                </h4>
                <p className="mt-0.5 font-mono text-[10px] text-slate-500">
                  Magnitude (dBFS) &nbsp;·&nbsp;
                  Δf = {spectrum.frequency_resolution_hz.toFixed(3)} Hz &nbsp;·&nbsp;
                  Nyquist = {(spectrum.nyquist_hz / 1000).toFixed(1)} kHz
                </p>
              </div>
              <span className="font-mono text-[10px] uppercase tracking-wider text-indigo-400 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-2 py-1">
                {spectrum.n_display_bins} bins
              </span>
            </div>
            <SpectrumViewer spectrum={spectrum} />
          </div>
        )}

        {/* Spectrogram / STFT */}
        {spectrogram && (
          <div className="rounded-2xl border border-white/[0.08] bg-[#161625] p-6 shadow-sm relative">
            <div className="mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h4 className="font-mono text-xs uppercase tracking-widest text-slate-400">
                  Spectrogram
                </h4>
                <p className="mt-0.5 font-mono text-[10px] text-slate-500">
                  STFT · {spectrogram.window} window · L = {spectrogram.frame_length} · H = {spectrogram.hop_length} &nbsp;·&nbsp;
                  Δf = {spectrogram.frequency_resolution_hz.toFixed(1)} Hz · Δt = {(spectrogram.time_resolution_seconds * 1000).toFixed(1)} ms
                </p>
              </div>
              
              <div className="flex items-center gap-3">
                <div className="relative" ref={dropdownRef}>
                  <button
                    type="button"
                    onClick={() => !isSpectrogramLoading && setIsDropdownOpen(!isDropdownOpen)}
                    disabled={isSpectrogramLoading}
                    className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-400 focus:bg-indigo-500/20 disabled:opacity-50 transition-colors"
                  >
                    Window = {WINDOW_OPTIONS.find((o) => o.value === spectrogramWindow)?.label || "Hann"}
                    <svg className="h-3 w-3 fill-current text-indigo-400" viewBox="0 0 20 20">
                      <path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" fillRule="evenodd" />
                    </svg>
                  </button>
                  {isDropdownOpen && (
                    <div className="absolute right-0 mt-2 w-40 origin-top-right rounded-xl border border-white/10 bg-[#0a0a14]/90 backdrop-blur-xl shadow-2xl shadow-black ring-1 ring-black ring-opacity-5 focus:outline-none z-50 overflow-hidden">
                      <div className="py-1">
                        {WINDOW_OPTIONS.map((option) => (
                          <button
                            key={option.value}
                            type="button"
                            className={[
                              "block w-full text-left px-4 py-2 text-xs font-mono tracking-wide transition-colors",
                              option.value === spectrogramWindow
                                ? "bg-indigo-500/20 text-indigo-300"
                                : "text-slate-300 hover:bg-white/10 hover:text-white"
                            ].join(" ")}
                            onClick={async () => {
                              setIsDropdownOpen(false);
                              setIsSpectrogramLoading(true);
                              try {
                                await updateSpectrogramWindow(option.value);
                              } finally {
                                setIsSpectrogramLoading(false);
                              }
                            }}
                          >
                            {option.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <span className="font-mono text-[10px] uppercase tracking-wider text-indigo-400 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-2 py-1 shrink-0">
                  {spectrogram.n_time_display}×{spectrogram.n_freq_display}
                </span>
              </div>
            </div>
            
            <div className={isSpectrogramLoading ? "opacity-50 transition-opacity" : "transition-opacity"}>
              <SpectrogramViewer spectrogram={spectrogram} />
            </div>
            {isSpectrogramLoading && (
              <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
                <Loader2 className="h-8 w-8 text-indigo-400 animate-spin" />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Signal Metrics Accordion */}
      <details className="mt-8 border-t border-white/5 pt-6 group">
        <summary className="flex cursor-pointer items-center justify-between font-mono text-xs uppercase tracking-widest text-slate-400 hover:text-white transition-colors">
          <span className="flex items-center gap-2">
            <span className="transition-transform duration-200 group-open:rotate-90 text-indigo-400">
              ▸
            </span>
            <span>Discrete-Time Signal Metrics</span>
          </span>
          <span className="text-[11px] text-slate-500">x[n] Characteristics</span>
        </summary>
        <dl className="mt-4 grid gap-3 sm:grid-cols-2">
          {[
            ["Samples / Channel", metadata.samples_per_channel.toLocaleString()],
            ["Total Samples", metadata.total_samples.toLocaleString()],
            ["Array Shape", `[${metadata.sample_summary.array_shape.join(", ")}]`],
            ["Min Value", metadata.sample_summary.min_value.toString()],
            ["Max Value", metadata.sample_summary.max_value.toString()],
            ["Mean Value", metadata.sample_summary.mean_value.toFixed(4)],
            ["Waveform Bins", `${waveform.n_bins.toLocaleString()} (peak-envelope)`],
            ["Mathematical Model", "x[n] discrete-time signal"],
          ].map(([label, value]) => (
            <div
              key={label}
              className="bg-[#161625] rounded-xl p-4 border border-white/[0.08] hover:border-white/[0.15] transition-colors"
            >
              <dt className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                {label}
              </dt>
              <dd className="mt-1.5 font-mono text-sm font-semibold text-indigo-200">
                {value}
              </dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}
