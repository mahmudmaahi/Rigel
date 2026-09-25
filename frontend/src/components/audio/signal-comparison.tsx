import React, { useState, useRef } from "react";
import { WaveformComparison } from "./waveform-comparison";
import { SpectrumComparison } from "./spectrum-comparison";
import { AdvancedSpectrogramViewer } from "./advanced-spectrogram-viewer";
import { AudioPlayer, AudioPlayerRef } from "./audio-player";

export interface SignalComparisonProps {
  originalWaveform: any;
  processedWaveform: any | null;
  originalSpectrum: any; // Using any for simplicity here to avoid full type imports if they differ
  processedSpectrum: any | null;
  originalSpectrogram?: any;
  processedSpectrogram?: any | null;
  
  originalAudioUrl: string;
  processedAudioUrl: string | null;
  
  originalDuration: number;
  processedDuration: number;
  
  processedLabel?: string;
}

export function SignalComparison({
  originalWaveform,
  processedWaveform,
  originalSpectrum,
  processedSpectrum,
  originalSpectrogram,
  processedSpectrogram,
  originalAudioUrl,
  processedAudioUrl,
  originalDuration,
  processedDuration,
  processedLabel = "Processed"
}: SignalComparisonProps) {
  const originalAudioRef = useRef<AudioPlayerRef>(null);
  const processedAudioRef = useRef<AudioPlayerRef>(null);

  const [originalTime, setOriginalTime] = useState(0);
  const [processedTime, setProcessedTime] = useState(0);
  
  const [originalSeekTarget, setOriginalSeekTarget] = useState<number | undefined>();
  const [processedSeekTarget, setProcessedSeekTarget] = useState<number | undefined>();
  
  const [activePlayer, setActivePlayer] = useState<"original" | "processed" | null>(null);

  const handlePlayOriginal = () => {
    setActivePlayer("original");
    if (processedAudioRef.current) {
      processedAudioRef.current.pause();
    }
  };

  const handlePlayProcessed = () => {
    setActivePlayer("processed");
    if (originalAudioRef.current) {
      originalAudioRef.current.pause();
    }
  };

  return (
    <div className="mt-8 flex flex-col gap-8 border-t border-white/5 pt-12 w-full">
      {/* 1. WAVEFORM COMPARISON */}
      <div className="flex flex-col gap-4 w-full min-w-0">
        <div className="flex items-center justify-between">
          <h2 className="text-[11px] font-bold tracking-widest text-slate-400 uppercase">
            {processedWaveform ? "Waveform Comparison" : "Original Waveform"}
          </h2>
        </div>

        <div className="rounded-3xl border border-white/[0.08] bg-[#161625] p-4 shadow-sm overflow-hidden w-full min-w-0">
          <WaveformComparison
            original={originalWaveform}
            filtered={processedWaveform}
            filteredLabel={processedLabel}
            currentTime={activePlayer === "processed" ? processedTime : originalTime}
            onSeek={(time) => {
              setOriginalSeekTarget(time);
              setProcessedSeekTarget(time);
            }}
          />
        </div>

        {/* Compact audio players */}
        <div className="flex flex-col gap-3">
          {originalAudioUrl && (
            <div className="flex flex-col gap-1">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 pl-1">Original</p>
              <AudioPlayer
                ref={originalAudioRef}
                src={originalAudioUrl}
                duration={originalDuration}
                onTimeUpdate={setOriginalTime}
                seekTarget={originalSeekTarget}
                onPlay={handlePlayOriginal}
              />
            </div>
          )}
          {processedAudioUrl && processedWaveform && (
            <div className="flex flex-col gap-1">
              <p className="text-[10px] font-bold uppercase tracking-widest text-cyan-500 pl-1">{processedLabel}</p>
              <AudioPlayer
                ref={processedAudioRef}
                src={processedAudioUrl}
                duration={processedDuration}
                onTimeUpdate={setProcessedTime}
                seekTarget={processedSeekTarget}
                onPlay={handlePlayProcessed}
              />
            </div>
          )}
        </div>
      </div>

      {/* 2. FREQUENCY SPECTRUM COMPARISON */}
      {originalSpectrum && (
        <div className="flex flex-col gap-4 w-full min-w-0">
          <h2 className="text-[11px] font-bold tracking-widest text-slate-400 uppercase">
            {processedSpectrum ? "Frequency Spectrum Comparison" : "Original Audio Spectrum"}
          </h2>
          <div className="rounded-3xl border border-white/[0.08] bg-[#161625] p-4 shadow-sm overflow-hidden w-full min-w-0">
            <SpectrumComparison 
              original={originalSpectrum} 
              filtered={processedSpectrum || null} 
              filteredLabel={processedLabel}
            />
          </div>
        </div>
      )}
      
      {/* 3. SPECTROGRAM (if available, mostly for Voice Lab) */}
      {originalSpectrogram && (
        <div className="flex flex-col gap-4 w-full min-w-0 mt-4">
          <h2 className="text-[11px] font-bold tracking-widest text-slate-400 uppercase">
            {processedSpectrogram ? "Spectrograms" : "Original Spectrogram"}
          </h2>
          <div className="flex flex-col gap-4">
            <AdvancedSpectrogramViewer 
              spectrogram={originalSpectrogram} 
              label="Original" 
              isProcessed={false} 
            />
            
            {processedSpectrogram && (
              <AdvancedSpectrogramViewer 
                spectrogram={processedSpectrogram} 
                label={processedLabel} 
                isProcessed={true} 
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
