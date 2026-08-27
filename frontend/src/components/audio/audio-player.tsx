"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Pause, Play, Volume2, VolumeX } from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AudioPlayerProps {
  /** Object URL created by the caller via URL.createObjectURL(file). */
  src: string;
  duration: number;
  /** Called on every timeupdate event so the parent can sync the waveform. */
  onTimeUpdate?: (currentTime: number) => void;
  /** Called when the audio ends. */
  onEnded?: () => void;
  /**
   * Programmatic seek target.  When this changes the player will seek to the
   * given time.  Use this to wire click-to-seek from WaveformViewer.
   */
  seekTarget?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTime(seconds: number): string {
  const clamped = Math.max(0, seconds);
  const m = Math.floor(clamped / 60);
  const s = Math.floor(clamped % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AudioPlayer({
  src,
  duration,
  onTimeUpdate,
  onEnded,
  seekTarget,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const prevSeekTarget = useRef<number | undefined>(undefined);

  // ---- Audio element event handlers ----

  const handleTimeUpdate = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    setCurrentTime(audio.currentTime);
    onTimeUpdate?.(audio.currentTime);
  }, [onTimeUpdate]);

  const handleEnded = useCallback(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    onEnded?.();
  }, [onEnded]);

  const handleCanPlay = useCallback(() => {
    setIsReady(true);
  }, []);

  // ---- Programmatic seek from waveform click ----
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || seekTarget === undefined) return;
    if (seekTarget === prevSeekTarget.current) return;

    prevSeekTarget.current = seekTarget;
    audio.currentTime = seekTarget;
    setCurrentTime(seekTarget);
    onTimeUpdate?.(seekTarget);
  }, [seekTarget, onTimeUpdate]);

  // ---- Play / pause ----
  async function togglePlay() {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      try {
        await audio.play();
        setIsPlaying(true);
      } catch {
        // Autoplay may be blocked; the user must interact first.
        setIsPlaying(false);
      }
    }
  }

  // ---- Mute toggle ----
  function toggleMute() {
    const audio = audioRef.current;
    if (!audio) return;
    audio.muted = !isMuted;
    setIsMuted(!isMuted);
  }

  // ---- Seek slider ----
  function handleSliderChange(event: React.ChangeEvent<HTMLInputElement>) {
    const audio = audioRef.current;
    if (!audio) return;
    const t = parseFloat(event.target.value);
    audio.currentTime = t;
    setCurrentTime(t);
    onTimeUpdate?.(t);
  }

  // ---- Derived values ----
  const progress = duration > 0 ? currentTime / duration : 0;

  return (
    <div className="flex flex-col gap-3">
      {/* Hidden native audio element */}
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        onCanPlay={handleCanPlay}
      />

      {/* Progress bar / seek slider */}
      <div className="relative h-1 w-full bg-border">
        <div
          className="absolute left-0 top-0 h-full bg-foreground transition-none"
          style={{ width: `${progress * 100}%` }}
        />
        <input
          type="range"
          min={0}
          max={duration}
          step={0.01}
          value={currentTime}
          disabled={!isReady}
          onChange={handleSliderChange}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
          aria-label="Seek"
        />
      </div>

      {/* Controls row */}
      <div className="flex items-center gap-4">
        {/* Play / Pause */}
        <button
          type="button"
          disabled={!isReady}
          onClick={() => void togglePlay()}
          className="flex h-9 w-9 items-center justify-center border border-border bg-card transition hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
          aria-label={isPlaying ? "Pause" : "Play"}
          id="audio-player-play-pause"
        >
          {isPlaying ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </button>

        {/* Time display */}
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Mute */}
        <button
          type="button"
          onClick={toggleMute}
          className="flex h-7 w-7 items-center justify-center text-muted-foreground transition hover:text-foreground"
          aria-label={isMuted ? "Unmute" : "Mute"}
          id="audio-player-mute"
        >
          {isMuted ? (
            <VolumeX className="h-4 w-4" />
          ) : (
            <Volume2 className="h-4 w-4" />
          )}
        </button>
      </div>
    </div>
  );
}
