import React, { useState, useRef, useEffect } from "react";
import { SpectrogramViewer } from "./spectrogram-viewer";
import { usePlayground } from "@/contexts/playground-context";
import { Loader2 } from "lucide-react";
import type { SpectrogramData } from "@/lib/api";

const WINDOW_OPTIONS = [
  { value: "rectangular", label: "Rectangular" },
  { value: "hamming", label: "Hamming" },
  { value: "hann", label: "Hann" },
  { value: "blackman", label: "Blackman" },
  { value: "kaiser", label: "Kaiser" },
  { value: "bartlett", label: "Bartlett" },
  { value: "welch", label: "Welch" },
];

export interface AdvancedSpectrogramViewerProps {
  spectrogram: SpectrogramData;
  label?: string;
  isProcessed?: boolean;
}

export function AdvancedSpectrogramViewer({ spectrogram, label = "Spectrogram", isProcessed = false }: AdvancedSpectrogramViewerProps) {
  const { state, updateSpectrogramWindow } = usePlayground();
  const [isSpectrogramLoading, setIsSpectrogramLoading] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const spectrogramWindow = state.status === "success" ? state.spectrogramWindow : "hann";

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Use indigo for processed, or standard styling for original
  const accentColor = isProcessed ? "indigo" : "slate";
  const badgeClass = isProcessed
    ? "text-indigo-400 border-indigo-500/20 bg-indigo-500/10 focus:ring-indigo-400 focus:bg-indigo-500/20"
    : "text-slate-300 border-white/10 bg-white/5 focus:ring-slate-400 focus:bg-white/10";
  const labelColorClass = isProcessed ? "text-indigo-400" : "text-slate-400";

  return (
    <div className={`rounded-2xl border ${isProcessed ? 'border-indigo-500/20 bg-[#161625]' : 'border-white/[0.08] bg-[#161625]'} p-6 shadow-sm relative w-full flex flex-col gap-4 min-w-0`}>
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h4 className={`font-mono text-xs uppercase tracking-widest ${labelColorClass}`}>
            {label}
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
              className={`flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider rounded-full border px-3 py-1.5 focus:outline-none focus:ring-1 disabled:opacity-50 transition-colors ${badgeClass}`}
            >
              Window = {WINDOW_OPTIONS.find((o) => o.value === spectrogramWindow)?.label || "Hann"}
              <svg className={`h-3 w-3 fill-current ${isProcessed ? 'text-indigo-400' : 'text-slate-400'}`} viewBox="0 0 20 20">
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

          <span className={`font-mono text-[10px] uppercase tracking-wider rounded-full border px-2 py-1 shrink-0 ${badgeClass}`}>
            {spectrogram.n_time_display}×{spectrogram.n_freq_display}
          </span>
        </div>
      </div>
      
      <div className={`w-full ${isSpectrogramLoading ? "opacity-50 transition-opacity" : "transition-opacity"}`}>
        <SpectrogramViewer spectrogram={spectrogram} />
      </div>
      {isSpectrogramLoading && (
        <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
          <div className="rounded-full bg-[#0a0a14]/80 p-3 backdrop-blur-sm">
            <Loader2 className="h-6 w-6 animate-spin text-indigo-400" />
          </div>
        </div>
      )}
    </div>
  );
}
