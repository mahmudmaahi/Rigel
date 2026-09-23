"use client";

/**
 * WaveformComparison — Filtering workspace visualization
 *
 * Renders original and filtered waveforms overlaid on identical
 * time and amplitude axes.
 *
 * Design invariants:
 *  - OVERLAPPED: Both waveforms drawn in the same row. Original in background, filtered on top.
 *  - SHARED globalAbsMax: computed from BOTH signals; NEVER independently normalized.
 *  - DISPLAY decimation only (~900 bins). DSP/audio is untouched.
 *  - COLORS: Original → Rigel indigo; Filtered → Rigel cyan.
 *  - TOGGLE VISIBILITY: legend items are clickable to show/hide each signal.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { WaveformData } from "@/lib/api";

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const AXIS_HEIGHT    = 26;   // px — time-tick row
const CHANNEL_HEIGHT = 160;  // px — per-channel waveform area (restored to original size)
const CHANNEL_GAP    = 16;   // px — vertical gap between stereo rows
const H_PAD          = 0;
const TOP_PAD        = 12;

const DISPLAY_BINS = 900;
const MAX_TICKS    = 8;

// ---------------------------------------------------------------------------
// Colour tokens
// ---------------------------------------------------------------------------

const COLOURS = {
  background:   "#0F0F1E",
  grid:         "rgba(255,255,255,0.08)",
  axisLabel:    "rgba(255,255,255,0.65)",
  channelLabel: "rgba(255,255,255,0.75)",
  original:     "rgba(99,102,241,0.80)",   // indigo-500 @ 80% (more visible underneath)
  // Filtered color is now computed dynamically based on amplitude ratio
  legendColors: {
    original: "#6366f1",
    filtered: "#22d3ee",
  },
} as const;

// ---------------------------------------------------------------------------
// Peak-preserving display decimation (VISUALIZATION ONLY)
// ---------------------------------------------------------------------------

interface DisplayBin {
  min_amplitude: number;
  max_amplitude: number;
}

function decimateBins(
  bins: { min_amplitude: number; max_amplitude: number }[],
  targetCount: number,
): DisplayBin[] {
  const n = bins.length;
  if (n <= targetCount) return bins;

  const result: DisplayBin[] = [];
  const step = n / targetCount;
  for (let i = 0; i < targetCount; i++) {
    const start = Math.floor(i * step);
    const end   = Math.min(Math.ceil((i + 1) * step), n);
    let bMin = Infinity;
    let bMax = -Infinity;
    for (let j = start; j < end; j++) {
      if (bins[j].min_amplitude < bMin) bMin = bins[j].min_amplitude;
      if (bins[j].max_amplitude > bMax) bMax = bins[j].max_amplitude;
    }
    result.push({ min_amplitude: bMin, max_amplitude: bMax });
  }
  return result;
}

// ---------------------------------------------------------------------------
// Axis helpers
// ---------------------------------------------------------------------------

function formatTime(seconds: number, totalDuration: number): string {
  if (totalDuration >= 60) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }
  return `${seconds.toFixed(1)}s`;
}

function chooseTick(duration: number): number {
  const candidates = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  for (const c of candidates) {
    if (duration / c <= MAX_TICKS) return c;
  }
  return Math.ceil(duration / MAX_TICKS);
}

// ---------------------------------------------------------------------------
// Draw one trace
// ---------------------------------------------------------------------------

function drawTrace(
  ctx: CanvasRenderingContext2D,
  bins: DisplayBin[],
  globalAbsMax: number,
  rowTop: number,
  drawW: number,
  color: string,
) {
  const midY  = rowTop + CHANNEL_HEIGHT / 2;
  const halfH = CHANNEL_HEIGHT / 2 - 4;
  const n = bins.length;

  ctx.fillStyle = color;
  for (let i = 0; i < n; i++) {
    const bin  = bins[i];
    const x    = H_PAD + (i / n) * drawW;
    const binW = Math.max(1, drawW / n);
    
    // Scale by globalAbsMax. 
    // If the signal is very weak compared to the original, it will naturally be small.
    // If it's literally a straight line, it means it's extremely attenuated.
    const yTop = midY - (bin.max_amplitude / globalAbsMax) * halfH;
    const yBot = midY - (bin.min_amplitude / globalAbsMax) * halfH;
    ctx.fillRect(x, yTop, binW - 0.5, Math.max(1, yBot - yTop));
  }
}

// ---------------------------------------------------------------------------
// Core canvas draw
// ---------------------------------------------------------------------------

function drawComparison(
  canvas: HTMLCanvasElement,
  original: WaveformData,
  filtered: WaveformData | null,
  currentTime: number,
  showOriginal: boolean,
  showFiltered: boolean,
) {
  const dpr       = window.devicePixelRatio || 1;
  const cssW      = canvas.clientWidth;
  const nChannels = original.n_channels;

  const cssH = TOP_PAD + nChannels * CHANNEL_HEIGHT + (nChannels - 1) * CHANNEL_GAP + AXIS_HEIGHT;

  if (
    canvas.width  !== Math.round(cssW * dpr) ||
    canvas.height !== Math.round(cssH * dpr)
  ) {
    canvas.width        = Math.round(cssW * dpr);
    canvas.height       = Math.round(cssH * dpr);
    canvas.style.height = `${cssH}px`;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  ctx.fillStyle = COLOURS.background;
  ctx.fillRect(0, 0, cssW, cssH);

  const drawW    = cssW - H_PAD * 2;
  const duration = original.duration_seconds;

  // ── Shared amplitude scale & Dynamic Opacity ─────────────────────────
  let origAbsMax = 1e-9;
  for (const ch of original.channels) {
    for (const b of ch.bins) {
      const a = Math.max(Math.abs(b.min_amplitude), Math.abs(b.max_amplitude));
      if (a > origAbsMax) origAbsMax = a;
    }
  }

  let filtAbsMax = 1e-9;
  if (filtered) {
    for (const ch of filtered.channels) {
      for (const b of ch.bins) {
        const a = Math.max(Math.abs(b.min_amplitude), Math.abs(b.max_amplitude));
        if (a > filtAbsMax) filtAbsMax = a;
      }
    }
  }

  let globalAbsMax = Math.max(origAbsMax, filtAbsMax);
  if (globalAbsMax < 0.001) {
    globalAbsMax = 0.001; 
  }

  // Dynamic Opacity: 
  // If filtered is small compared to original (e.g. heavy cut-off), make it very opaque (0.95) so it's visible.
  // If filtered is almost the same size as original (ratio ~ 1), make it translucent (0.45) so original shows through.
  const ratio = Math.min(1.0, filtAbsMax / origAbsMax);
  const dynamicFiltAlpha = 0.95 - ratio * (0.95 - 0.45);
  const dynamicFilteredColor = `rgba(34,211,238,${dynamicFiltAlpha.toFixed(2)})`;

  // ── Per-channel rendering ──────────────────────────────────────────────
  for (let chIdx = 0; chIdx < nChannels; chIdx++) {
    const rowTop = TOP_PAD + chIdx * (CHANNEL_HEIGHT + CHANNEL_GAP);
    const midY   = rowTop + CHANNEL_HEIGHT / 2;

    // Zero-line
    ctx.strokeStyle = COLOURS.grid;
    ctx.lineWidth   = 0.5;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(H_PAD, midY);
    ctx.lineTo(H_PAD + drawW, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Channel label
    const chLabel = nChannels > 1 ? (chIdx === 0 ? "CH_L" : "CH_R") : "CH (MONO)";
    ctx.fillStyle = COLOURS.channelLabel;
    ctx.font      = "bold 11px monospace";
    ctx.textAlign = "left";
    ctx.fillText(chLabel, H_PAD + 6, rowTop + 14);

    // Amplitude scale markers
    ctx.fillStyle = COLOURS.axisLabel;
    ctx.font      = "10px monospace";
    ctx.textAlign = "right";
    ctx.fillText("+1.0", H_PAD + drawW - 6, rowTop + 12);
    ctx.fillText("0.0",  H_PAD + drawW - 6, midY + 3);
    ctx.fillText("-1.0", H_PAD + drawW - 6, rowTop + CHANNEL_HEIGHT - 6);

    // Draw Original (underneath)
    if (showOriginal) {
      const origCh = original.channels[chIdx];
      if (origCh) {
        const origBins = decimateBins(origCh.bins, DISPLAY_BINS);
        drawTrace(ctx, origBins, globalAbsMax, rowTop, drawW, COLOURS.original);
      }
    }

    // Draw Filtered (on top)
    if (showFiltered && filtered) {
      const filtCh = filtered.channels[chIdx];
      if (filtCh) {
        const filtBins = decimateBins(filtCh.bins, DISPLAY_BINS);
        drawTrace(ctx, filtBins, globalAbsMax, rowTop, drawW, dynamicFilteredColor);
      }
    }
  }

  // ── Time axis ──────────────────────────────────────────────────────────
  const axisTop      = TOP_PAD + nChannels * CHANNEL_HEIGHT + (nChannels - 1) * CHANNEL_GAP;
  const tickInterval = chooseTick(duration);

  ctx.strokeStyle = COLOURS.grid;
  ctx.lineWidth   = 0.5;
  ctx.beginPath();
  ctx.moveTo(H_PAD, axisTop);
  ctx.lineTo(H_PAD + drawW, axisTop);
  ctx.stroke();

  ctx.fillStyle = COLOURS.axisLabel;
  ctx.font      = "11px system-ui, sans-serif";
  ctx.textAlign = "center";

  for (let t = 0; t <= duration; t += tickInterval) {
    const x = H_PAD + (t / duration) * drawW;
    ctx.strokeStyle = COLOURS.grid;
    ctx.lineWidth   = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, axisTop);
    ctx.lineTo(x, axisTop + 4);
    ctx.stroke();
    ctx.fillStyle = COLOURS.axisLabel;
    ctx.fillText(formatTime(t, duration), x, axisTop + AXIS_HEIGHT - 4);
  }

  // ── Playhead ───────────────────────────────────────────────────────────
  if (duration > 0 && currentTime >= 0) {
    const playX = H_PAD + (currentTime / duration) * drawW;
    const totalH = axisTop; // spans all channel blocks

    // Subtle glow
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.1;
    ctx.beginPath();
    ctx.moveTo(playX, TOP_PAD);
    ctx.lineTo(playX, totalH);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Sharp playhead line
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, TOP_PAD);
    ctx.lineTo(playX, totalH);
    ctx.stroke();

    // Triangle handle at top
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.beginPath();
    ctx.moveTo(playX, TOP_PAD);
    ctx.lineTo(playX - 4, 2);
    ctx.lineTo(playX + 4, 2);
    ctx.fill();
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface WaveformComparisonProps {
  original: WaveformData;
  filtered: WaveformData | null;
  currentTime?: number;
  onSeek?: (timeSeconds: number) => void;
  filteredLabel?: string;
}

export function WaveformComparison({ original, filtered, currentTime = 0, onSeek, filteredLabel = "Filtered" }: WaveformComparisonProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDragging = useRef(false);
  const [showOriginal, setShowOriginal] = useState(true);
  const [showFiltered, setShowFiltered] = useState(true);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawComparison(canvas, original, filtered, currentTime, showOriginal, showFiltered);
  }, [original, filtered, currentTime, showOriginal, showFiltered]);

  useEffect(() => { redraw(); }, [redraw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => { redraw(); });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [redraw]);

  const { original: origColor, filtered: filtColor } = COLOURS.legendColors;

  const handlePointer = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!onSeek) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left - H_PAD, rect.width - H_PAD * 2));
      const ratio = x / (rect.width - H_PAD * 2);
      onSeek(ratio * original.duration_seconds);
    },
    [onSeek, original.duration_seconds],
  );

  return (
    <div className="w-full select-none">
      {/* Legend header — sits ABOVE the canvas, no overlap */}
      <div className="flex items-center justify-end gap-4 px-2 pb-2">
        <button
          type="button"
          onClick={() => setShowOriginal((v) => !v)}
          className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest transition-opacity cursor-pointer"
          style={{ color: origColor, opacity: showOriginal ? 1 : 0.3 }}
        >
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: origColor }} />
          Original
        </button>
        {filtered && (
          <button
            type="button"
            onClick={() => setShowFiltered((v) => !v)}
            className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest transition-opacity cursor-pointer"
            style={{ color: filtColor, opacity: showFiltered ? 1 : 0.3 }}
          >
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: filtColor }} />
            {filteredLabel}
          </button>
        )}
      </div>

      {/* Canvas */}
      <div
        className="relative w-full overflow-hidden cursor-crosshair"
        onMouseDown={(e) => {
          isDragging.current = true;
          handlePointer(e);
        }}
        onMouseMove={(e) => {
          if (isDragging.current) handlePointer(e);
        }}
        onMouseUp={() => { isDragging.current = false; }}
        onMouseLeave={() => { isDragging.current = false; }}
      >
        <canvas
          ref={canvasRef}
          className="w-full"
          aria-label="Waveform comparison: original vs filtered audio signal"
        />
      </div>
    </div>
  );
}
