"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { WaveformData } from "@/lib/api";
import { createSymmetricColorMapGradient } from "@/lib/colors";

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
const CHANNEL_HEIGHT = 140;
/** Vertical gap between two channel rows (stereo). */
const CHANNEL_GAP = 8;
/** Horizontal padding inside the canvas. */
const H_PAD = 0;
/** Top padding for playhead handle. */
const TOP_PAD = 10;

/** Maximum number of time-axis tick labels to draw. */
const MAX_TICKS = 8;

// ---------------------------------------------------------------------------
// Colour palette (matches SpectrumViewer)
// ---------------------------------------------------------------------------

const COLOURS = {
  background:    "#0F0F1E",
  grid:          "rgba(255,255,255,0.08)",
  axisLabel:     "rgba(255,255,255,0.65)",
  channelLabel:  "rgba(255,255,255,0.75)",
  playhead:      "rgba(129,140,248,0.85)", // indigo-400
  waveform:      "#6366f1", // indigo-500
} as const;

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
    TOP_PAD +
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

  // Fill background
  ctx.fillStyle = COLOURS.background;
  ctx.fillRect(0, 0, cssW, cssH);

  // ------ Colour tokens ------
  const fg = COLOURS.playhead;
  const muted = COLOURS.axisLabel;
  const border = COLOURS.grid;

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
    const rowTop = TOP_PAD + chIdx * (CHANNEL_HEIGHT + CHANNEL_GAP);
    const midY = rowTop + CHANNEL_HEIGHT / 2;
    const halfH = CHANNEL_HEIGHT / 2 - 4; // 4 px breathing room

    // Centre line (zero amplitude reference).
    ctx.strokeStyle = border;
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(H_PAD, midY);
    ctx.lineTo(H_PAD + drawW, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Channel label for stereo + technical reference marks.
    ctx.fillStyle = muted;
    ctx.font = `11px monospace`;
    ctx.textAlign = "left";
    if (nChannels > 1) {
      ctx.fillText(chIdx === 0 ? "CH_L" : "CH_R", H_PAD + 6, rowTop + 14);
    } else {
      ctx.fillText("CH_01 (MONO)", H_PAD + 6, rowTop + 14);
    }

    // Amplitude scale indicators
    ctx.textAlign = "right";
    ctx.fillText("+1.0", H_PAD + drawW - 6, rowTop + 12);
    ctx.fillText("0.0", H_PAD + drawW - 6, midY + 3);
    ctx.fillText("-1.0", H_PAD + drawW - 6, rowTop + CHANNEL_HEIGHT - 6);

    // Peak-envelope bars.
    const bins = ch.bins;
    const nBins = bins.length;

    // Use a solid color for the waveform to improve clarity
    ctx.fillStyle = COLOURS.waveform;

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
    TOP_PAD + nChannels * CHANNEL_HEIGHT + (nChannels - 1) * CHANNEL_GAP;
  const tickInterval = chooseTick(duration);

  ctx.fillStyle = muted;
  ctx.font = `11px system-ui, sans-serif`;
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
    const totalH =
      TOP_PAD + nChannels * CHANNEL_HEIGHT + (nChannels - 1) * CHANNEL_GAP;

    // Subtle glow effect behind playhead
    ctx.strokeStyle = fg;
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.1;
    ctx.beginPath();
    ctx.moveTo(playX, TOP_PAD);
    ctx.lineTo(playX, totalH);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Sharp playhead line
    ctx.strokeStyle = fg;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, TOP_PAD);
    ctx.lineTo(playX, totalH);
    ctx.stroke();

    // Precise triangle handle at the top within canvas bounds.
    ctx.fillStyle = fg;
    ctx.beginPath();
    ctx.moveTo(playX, TOP_PAD);
    ctx.lineTo(playX - 4, 2);
    ctx.lineTo(playX + 4, 2);
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
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    time: number;
  } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

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

  // Convert a click/drag position to a seek time and call onSeek.
  function handleSeek(clientX: number) {
    if (!onSeek || waveform.duration_seconds <= 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    onSeek(ratio * waveform.duration_seconds);
  }

  function handleMouseDown(event: React.MouseEvent<HTMLCanvasElement>) {
    setIsDragging(true);
    handleSeek(event.clientX);
  }

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = event.target as HTMLCanvasElement;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const time = (x / rect.width) * waveform.duration_seconds;
    setHoverInfo({ x, time });

    // If dragging, continuously seek
    if (isDragging) {
      handleSeek(event.clientX);
    }
  }

  function handleMouseUp() {
    setIsDragging(false);
  }

  function handleMouseLeave() {
    setHoverInfo(null);
    setIsDragging(false);
  }

  // Handle global mouse up to stop dragging even if mouse leaves canvas
  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, []);

  const nChannels = waveform.n_channels;
  const cssH =
    TOP_PAD +
    nChannels * CHANNEL_HEIGHT +
    (nChannels - 1) * CHANNEL_GAP +
    AXIS_HEIGHT;

  return (
    <div className="relative w-full select-none overflow-hidden">
      <canvas
        ref={canvasRef}
        style={{ height: cssH, cursor: onSeek ? (isDragging ? "grabbing" : "pointer") : "default" }}
        className="w-full"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        aria-label="Audio waveform"
      />
      {/* Hover scrubber line */}
      {hoverInfo !== null && (
        <div
          className="pointer-events-none absolute top-0 h-full w-px bg-foreground/30 transition-opacity"
          style={{
            left: `${hoverInfo.x}px`,
            height: `${cssH}px`,
          }}
        />
      )}
    </div>
  );
}
