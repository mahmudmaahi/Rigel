"use client";

import { useEffect, useRef, useState } from "react";
import { Waves, Loader2 } from "lucide-react";
import { usePlayground } from "@/contexts/playground-context";
import { WaveformViewer } from "@/components/audio/waveform-viewer";
import { AudioPlayer } from "@/components/audio/audio-player";
import { SpectrumComparison } from "@/components/audio/spectrum-comparison";
import { AdvancedSpectrogramViewer } from "@/components/audio/advanced-spectrogram-viewer";

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
  const { state } = usePlayground();
  
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTarget, setSeekTarget] = useState<number | undefined>(undefined);

  if (state.status !== "success") {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center gap-6 text-center">
        <div className="relative">
          <div className="absolute inset-0 rounded-full bg-indigo-500/20 blur-2xl scale-150" />
          <div className="relative rounded-full bg-indigo-900/30 border border-indigo-500/20 p-6">
            <Waves className="h-12 w-12 text-indigo-400" />
          </div>
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Audio Analysis</h1>
          <p className="text-slate-400 mt-2 text-sm max-w-md mx-auto">
            Inspect waveforms, analyze frequency content, and view spectrograms.
          </p>
          <p className="text-slate-500 mt-4 text-xs">Upload a WAV file from the sidebar to begin.</p>
        </div>
      </div>
    );
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
            <SpectrumComparison original={spectrum} filtered={null} />
          </div>
        )}

        {spectrogram && (
          <AdvancedSpectrogramViewer spectrogram={spectrogram} />
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
