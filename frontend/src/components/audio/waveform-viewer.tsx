"use client";

import { useCallback, useEffect, useRef } from "react";

import type { WaveformData } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WaveformViewerProps {
  waveform: WaveformData;
  /** Current playback time in seconds, for the playhead. */
  currentTime: number;
  /** Called when the user clicks to seek. Argument is time in seconds. */
  onSeek?: (timeSeconds: number) => void;
}

// ---------------------------------------------------------------------------
// Design constants (all values in pixels unless noted)
// ---------------------------------------------------------------------------

/** Height of the time-axis tick area below each channel waveform. */
const AXIS_HEIGHT = 24;
/** Height of each channel's waveform canvas area. */
const CHANNEL_HEIGHT = 80;
/** Vertical gap between two channel rows (stereo). */
const CHANNEL_GAP = 8;
/** Horizontal padding inside the canvas. */
const H_PAD = 0;

/** Maximum number of time-axis tick labels to draw. */
const MAX_TICKS = 8;

// ---------------------------------------------------------------------------
// Colour helpers
// We read computed CSS custom properties at draw time so the waveform respects
// the active Rigel theme (light / future dark mode) without hard-coding values.
// ---------------------------------------------------------------------------

function getCssVar(el: HTMLElement, name: string): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------------------
// Draw helpers
// ---------------------------------------------------------------------------

/**
 * Format a time value (seconds) as mm:ss or ss.d depending on duration.
 */
function formatTime(seconds: number, totalDuration: number): string {
  if (totalDuration >= 60) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }
  return `${seconds.toFixed(1)}s`;
}

/**
 * Choose a sensible tick interval (in seconds) given the duration and the
 * maximum number of ticks we want to show.
 */
function chooseTick(duration: number): number {
  const candidates = [
    0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600,
  ];
  for (const c of candidates) {
    if (duration / c <= MAX_TICKS) return c;
  }
  return Math.ceil(duration / MAX_TICKS);
}

// ---------------------------------------------------------------------------
// Core draw function
// ---------------------------------------------------------------------------

function drawWaveform(
  canvas: HTMLCanvasElement,
  waveform: WaveformData,
  currentTime: number,
) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const nChannels = waveform.n_channels;
  const cssH =
    nChannels * CHANNEL_HEIGHT +
    (nChannels - 1) * CHANNEL_GAP +
    AXIS_HEIGHT;

  // Resize backing buffer if needed.
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.height = `${cssH}px`;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  // ------ Colour tokens (read from CSS custom properties) ------
  const fg = getCssVar(canvas, "--foreground") || "hsl(0 0% 3.9%)";
  const muted = getCssVar(canvas, "--muted-foreground") || "hsl(0 0% 45.1%)";
  const border = getCssVar(canvas, "--border") || "hsl(0 0% 89.8%)";
  const accent = getCssVar(canvas, "--accent") || "hsl(0 0% 96.1%)";

  const drawW = cssW - H_PAD * 2;
  const duration = waveform.duration_seconds;

  // Find global amplitude range across all channels for consistent scaling.
  let globalMin = Infinity;
  let globalMax = -Infinity;
  for (const ch of waveform.channels) {
    for (const bin of ch.bins) {
      if (bin.min_amplitude < globalMin) globalMin = bin.min_amplitude;
      if (bin.max_amplitude > globalMax) globalMax = bin.max_amplitude;
    }
  }
  // Guard: flat / silent audio.
  const absMax = Math.max(Math.abs(globalMin), Math.abs(globalMax), 1);

  // ------ Draw each channel ------
  waveform.channels.forEach((ch, chIdx) => {
    const rowTop = chIdx * (CHANNEL_HEIGHT + CHANNEL_GAP);
    const midY = rowTop + CHANNEL_HEIGHT / 2;
    const halfH = CHANNEL_HEIGHT / 2 - 4; // 4 px breathing room

    // Centre line (zero amplitude reference).
    ctx.strokeStyle = border;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(H_PAD, midY);
    ctx.lineTo(H_PAD + drawW, midY);
    ctx.stroke();

    // Channel label for stereo.
    if (nChannels > 1) {
      ctx.fillStyle = muted;
      ctx.font = `10px system-ui, sans-serif`;
      ctx.textAlign = "left";
      ctx.fillText(chIdx === 0 ? "L" : "R", H_PAD + 4, rowTop + 12);
    }

    // Peak-envelope bars.
    const bins = ch.bins;
    const nBins = bins.length;

    ctx.fillStyle = fg;

    for (let i = 0; i < nBins; i++) {
      const bin = bins[i];
      const x = H_PAD + (i / nBins) * drawW;
      const binW = Math.max(1, drawW / nBins);

      const minNorm = bin.min_amplitude / absMax; // −1 … +1
      const maxNorm = bin.max_amplitude / absMax;

      const yTop = midY - maxNorm * halfH;
      const yBot = midY - minNorm * halfH;
      const barH = Math.max(1, yBot - yTop);

      ctx.fillRect(x, yTop, binW - 0.5, barH);
    }
  });

  // ------ Time axis (below all channels) ------
  const axisTop =
    nChannels * CHANNEL_HEIGHT + (nChannels - 1) * CHANNEL_GAP;
  const tickInterval = chooseTick(duration);

  ctx.fillStyle = muted;
  ctx.font = `10px system-ui, sans-serif`;
  ctx.textAlign = "center";

  // Axis baseline.
  ctx.strokeStyle = border;
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(H_PAD, axisTop);
  ctx.lineTo(H_PAD + drawW, axisTop);
  ctx.stroke();

  for (let t = 0; t <= duration; t += tickInterval) {
    const x = H_PAD + (t / duration) * drawW;

    ctx.strokeStyle = border;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, axisTop);
    ctx.lineTo(x, axisTop + 4);
    ctx.stroke();

    ctx.fillStyle = muted;
    ctx.fillText(formatTime(t, duration), x, axisTop + AXIS_HEIGHT - 4);
  }

  // ------ Playhead ------
  if (duration > 0) {
    const playX = H_PAD + (currentTime / duration) * drawW;
    const totalH = nChannels * CHANNEL_HEIGHT + (nChannels - 1) * CHANNEL_GAP;

    ctx.strokeStyle = fg;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, totalH);
    ctx.stroke();

    // Small triangle handle at the top.
    ctx.fillStyle = fg;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX - 5, -8);
    ctx.lineTo(playX + 5, -8);
    ctx.closePath();
    ctx.fill();
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function WaveformViewer({
  waveform,
  currentTime,
  onSeek,
}: WaveformViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Redraw whenever the waveform data or playback position changes.
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawWaveform(canvas, waveform, currentTime);
  }, [waveform, currentTime]);

  // Redraw on initial mount and whenever redraw changes.
  useEffect(() => {
    redraw();
  }, [redraw]);

  // Handle canvas resize via ResizeObserver.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const observer = new ResizeObserver(() => {
      redraw();
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [redraw]);

  // Convert a click position to a seek time and call onSeek.
  function handleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!onSeek || waveform.duration_seconds <= 0) return;
    const rect = (event.target as HTMLCanvasElement).getBoundingClientRect();
    const x = event.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    onSeek(ratio * waveform.duration_seconds);
  }

  const nChannels = waveform.n_channels;
  const cssH =
    nChannels * CHANNEL_HEIGHT +
    (nChannels - 1) * CHANNEL_GAP +
    AXIS_HEIGHT;

  return (
    <div className="w-full select-none overflow-hidden">
      <canvas
        ref={canvasRef}
        style={{ height: cssH, cursor: onSeek ? "pointer" : "default" }}
        className="w-full"
        onClick={handleClick}
        aria-label="Audio waveform"
      />
    </div>
  );
}
