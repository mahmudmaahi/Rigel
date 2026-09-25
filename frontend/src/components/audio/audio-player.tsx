"use client";

import { useCallback, useEffect, useRef, useState, forwardRef, useImperativeHandle } from "react";
import { Pause, Play, Volume2, VolumeX } from "lucide-react";

interface AudioPlayerProps {
  src: string;
  duration: number;
  onTimeUpdate?: (currentTime: number) => void;
  onEnded?: () => void;
  onPlay?: () => void;
  onPause?: () => void;
  onSeek?: (time: number) => void;
  seekTarget?: number;
}

export interface AudioPlayerRef {
  pause: () => void;
  reset: () => void;
  getCurrentTime: () => number;
}

function formatTime(seconds: number): string {
  const clamped = Math.max(0, seconds);
  const m = Math.floor(clamped / 60);
  const s = Math.floor(clamped % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export const AudioPlayer = forwardRef<AudioPlayerRef, AudioPlayerProps>(({
  src,
  duration,
  onTimeUpdate,
  onEnded,
  onPlay,
  onPause,
  onSeek,
  seekTarget,
}, ref) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const prevSeekTarget = useRef<number | undefined>(undefined);

  const handleTimeUpdate = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    setCurrentTime(audio.currentTime);
    onTimeUpdate?.(audio.currentTime);
  }, [onTimeUpdate]);

  const handleEnded = useCallback(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    if (audioRef.current) audioRef.current.currentTime = 0;
    onSeek?.(0);
    onEnded?.();
  }, [onEnded, onSeek]);

  const handleCanPlay = useCallback(() => {
    setIsReady(true);
  }, []);

  useImperativeHandle(ref, () => ({
    pause: () => {
      if (audioRef.current && !audioRef.current.paused) {
        audioRef.current.pause();
        setIsPlaying(false);
        onPause?.();
      }
    },
    reset: () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        setIsPlaying(false);
        setCurrentTime(0);
        onTimeUpdate?.(0);
        onPause?.();
      }
    },
    getCurrentTime: () => {
      return audioRef.current ? audioRef.current.currentTime : 0;
    }
  }));

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || seekTarget === undefined) return;
    if (seekTarget === prevSeekTarget.current) return;

    prevSeekTarget.current = seekTarget;
    audio.currentTime = seekTarget;
    setCurrentTime(seekTarget);
    onTimeUpdate?.(seekTarget);
    onSeek?.(seekTarget);
  }, [seekTarget, onTimeUpdate]);

  async function togglePlay() {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
      onPause?.();
    } else {
      try {
        await audio.play();
        setIsPlaying(true);
        onPlay?.();
      } catch {
        setIsPlaying(false);
      }
    }
  }

  function toggleMute() {
    const audio = audioRef.current;
    if (!audio) return;
    audio.muted = !isMuted;
    setIsMuted(!isMuted);
  }

  function handleSliderInput(event: React.FormEvent<HTMLInputElement>) {
    const audio = audioRef.current;
    if (!audio) return;
    const t = parseFloat((event.target as HTMLInputElement).value);
    audio.currentTime = t;
    setCurrentTime(t);
    onTimeUpdate?.(t);
    onSeek?.(t);
  }

  const progress = duration > 0 ? currentTime / duration : 0;

  return (
    <div className="flex flex-col gap-4">
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        onCanPlay={handleCanPlay}
      />

      {/* Seek Progress Bar */}
      <div className="group relative flex h-6 w-full cursor-pointer items-center">
        <div className="relative h-2 w-full rounded-full bg-white/10 backdrop-blur-sm transition-all group-hover:h-2.5">
          <div
            className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400 shadow-[0_0_12px_rgba(99,102,241,0.5)] transition-all duration-200 ease-linear"
            style={{ width: `${progress * 100}%` }}
          />
          <div 
            className="absolute top-1/2 h-3.5 w-3.5 -translate-y-1/2 -translate-x-1/2 rounded-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)] transition-all duration-200 ease-linear scale-75 group-hover:scale-100"
            style={{ left: `${progress * 100}%` }}
          />
        </div>
        <input
          type="range"
          min={0}
          max={duration}
          step={0.01}
          value={currentTime}
          disabled={!isReady}
          onInput={handleSliderInput}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
          aria-label="Seek audio"
        />
      </div>

      {/* Control Buttons */}
      <div className="flex items-center gap-4">
        <button
          type="button"
          disabled={!isReady}
          onClick={() => void togglePlay()}
          className="flex h-11 w-11 items-center justify-center rounded-xl border border-white/20 bg-white/10 backdrop-blur-md text-white shadow-lg transition-all hover:bg-white/20 hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <Pause className="h-5 w-5" />
          ) : (
            <Play className="h-5 w-5 translate-x-0.5" />
          )}
        </button>

        <span className="font-mono text-xs tabular-nums text-slate-300">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        <div className="flex-1" />

        <button
          type="button"
          onClick={toggleMute}
          className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/5 bg-white/5 text-slate-300 transition-all hover:border-white/10 hover:bg-white/10 hover:text-white"
          aria-label={isMuted ? "Unmute" : "Mute"}
        >
          {isMuted ? (
            <VolumeX className="h-4 w-4 text-red-400" />
          ) : (
            <Volume2 className="h-4 w-4" />
          )}
        </button>
      </div>
    </div>
  );
});

AudioPlayer.displayName = "AudioPlayer";
