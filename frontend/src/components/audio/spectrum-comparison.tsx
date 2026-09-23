"use client";

/**
 * SpectrumComparison — Filtering workspace visualization
 *
 * Renders two SpectrumData objects (original + filtered) overlaid on
 * the same frequency/dBFS axes, one canvas row per audio channel.
 *
 * Design invariants:
 *  - IDENTICAL axes: same nyquist_hz, same DB_MIN/DB_MAX. No rescaling.
 *  - STROKES carry the comparison: fills are subtle; strokes are primary.
 *  - COLORS: Original → Rigel indigo/violet; Filtered → Rigel cyan.
 *  - TOGGLE VISIBILITY: legend items are clickable.
 *  - CHART SIZE: CHANNEL_HEIGHT = 220px — large presentation.
 */

import { useEffect, useRef, useState } from "react";
import type { SpectrumData } from "@/lib/api";
import { DB_MIN, DB_MAX } from "@/lib/colors";

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const CHANNEL_HEIGHT = 220;
const CHANNEL_GAP    = 16;
const PADDING = { top: 36, right: 24, bottom: 48, left: 60 };

const DB_GRID_LINES = [-20, -40, -60, -80];
const FREQ_TICK_TARGET = 8;

// ---------------------------------------------------------------------------
// Colour tokens
// ---------------------------------------------------------------------------

const ORIGINAL_FILL   = "rgba(99,102,241,0.18)";
const ORIGINAL_STROKE = "rgba(129,140,248,0.80)";
const FILTERED_FILL   = "rgba(34,211,238,0.12)";
const FILTERED_STROKE = "rgba(34,211,238,0.85)";

const COLOURS = {
  background:   "#0F0F1E",
  grid:         "rgba(255,255,255,0.10)",
  axisLabel:    "rgba(255,255,255,0.65)",
  channelLabel: "rgba(255,255,255,0.75)",
  zeroDB:       "rgba(255,255,255,0.15)",
  legendColors: {
    original: "#818cf8",
    filtered: "#22d3ee",
  },
} as const;

// ---------------------------------------------------------------------------
// Axis helpers
// ---------------------------------------------------------------------------

function niceFreqInterval(nyquistHz: number): number {
  const raw  = nyquistHz / FREQ_TICK_TARGET;
  const nice = [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000];
  return nice.find((n) => n >= raw) ?? nice[nice.length - 1];
}

function freqToX(freqHz: number, nyquistHz: number, plotLeft: number, plotWidth: number): number {
  return plotLeft + (freqHz / nyquistHz) * plotWidth;
}

function dbToY(db: number, rowTop: number, rowHeight: number): number {
  const fraction = (DB_MAX - db) / (DB_MAX - DB_MIN);
  return rowTop + fraction * rowHeight;
}

// ---------------------------------------------------------------------------
// Draw one spectrum trace
// ---------------------------------------------------------------------------

function drawTrace(
  ctx: CanvasRenderingContext2D,
  bins: { frequency_hz: number; magnitude_dbfs: number }[],
  nyquistHz: number,
  rowTop: number,
  plotLeft: number,
  plotWidth: number,
  plotHeight: number,
  fillColor: string,
  strokeColor: string,
) {
  if (bins.length === 0) return;

  const bottomY = rowTop + plotHeight;
  const firstX  = freqToX(bins[0].frequency_hz, nyquistHz, plotLeft, plotWidth);
  const lastX   = freqToX(bins[bins.length - 1].frequency_hz, nyquistHz, plotLeft, plotWidth);

  // Area fill (subtle)
  ctx.beginPath();
  ctx.moveTo(firstX, bottomY);
  for (const bin of bins) {
    const x = freqToX(bin.frequency_hz, nyquistHz, plotLeft, plotWidth);
    const y = dbToY(Math.max(DB_MIN, Math.min(DB_MAX, bin.magnitude_dbfs)), rowTop, plotHeight);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(lastX, bottomY);
  ctx.closePath();
  ctx.fillStyle   = fillColor;
  ctx.globalAlpha = 1.0;
  ctx.fill();

  // Top-edge stroke (primary comparison signal)
  ctx.beginPath();
  ctx.moveTo(firstX, dbToY(Math.max(DB_MIN, Math.min(DB_MAX, bins[0].magnitude_dbfs)), rowTop, plotHeight));
  for (const bin of bins) {
    const x = freqToX(bin.frequency_hz, nyquistHz, plotLeft, plotWidth);
    const y = dbToY(Math.max(DB_MIN, Math.min(DB_MAX, bin.magnitude_dbfs)), rowTop, plotHeight);
    ctx.lineTo(x, y);
  }
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth   = 1.5;
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Core canvas draw
// ---------------------------------------------------------------------------

function drawSpectrumComparison(
  canvas: HTMLCanvasElement,
  original: SpectrumData,
  filtered: SpectrumData | null,
  showOriginal: boolean,
  showFiltered: boolean,
) {
  const dpr      = window.devicePixelRatio || 1;
  const cssWidth = canvas.offsetWidth;
  const nChannels = original.channels.length;

  const cssHeight =
    PADDING.top +
    nChannels * CHANNEL_HEIGHT +
    (nChannels - 1) * CHANNEL_GAP +
    PADDING.bottom;

  canvas.width        = cssWidth * dpr;
  canvas.height       = cssHeight * dpr;
  canvas.style.height = `${cssHeight}px`;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  ctx.fillStyle = COLOURS.background;
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const plotLeft  = PADDING.left;
  const plotWidth = cssWidth - PADDING.left - PADDING.right;
  const nyquistHz = original.nyquist_hz;

  const channelLabels = nChannels === 1 ? ["CH"] : ["CH_L", "CH_R", "CH_3", "CH_4"];

  for (let ch = 0; ch < nChannels; ch++) {
    const rowTop     = PADDING.top + ch * (CHANNEL_HEIGHT + CHANNEL_GAP);
    const plotHeight = CHANNEL_HEIGHT;

    ctx.fillStyle = COLOURS.background;
    ctx.fillRect(plotLeft, rowTop, plotWidth, plotHeight);

    // dBFS grid lines
    ctx.strokeStyle = COLOURS.grid;
    ctx.lineWidth   = 1;

    for (const db of DB_GRID_LINES) {
      const y = dbToY(db, rowTop, plotHeight);
      ctx.beginPath();
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(plotLeft + plotWidth, y);
      ctx.stroke();

      ctx.fillStyle  = COLOURS.axisLabel;
      ctx.font       = "10px monospace";
      ctx.textAlign  = "right";
      ctx.fillText(`${db}`, plotLeft - 8, y + 3);
    }

    // 0 dBFS reference line
    const yZero = dbToY(0, rowTop, plotHeight);
    ctx.strokeStyle = COLOURS.zeroDB;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(plotLeft, yZero);
    ctx.lineTo(plotLeft + plotWidth, yZero);
    ctx.stroke();

    ctx.fillStyle  = COLOURS.axisLabel;
    ctx.font       = "11px monospace";
    ctx.textAlign  = "right";
    ctx.fillText("0", plotLeft - 8, yZero + 3);

    // ORIGINAL trace
    if (showOriginal) {
      const origBins = original.channels[ch]?.bins ?? [];
      drawTrace(ctx, origBins, nyquistHz, rowTop, plotLeft, plotWidth, plotHeight,
        ORIGINAL_FILL, ORIGINAL_STROKE);
    }

    // FILTERED trace
    if (showFiltered && filtered) {
      const filtBins = filtered.channels[ch]?.bins ?? [];
      drawTrace(ctx, filtBins, nyquistHz, rowTop, plotLeft, plotWidth, plotHeight,
        FILTERED_FILL, FILTERED_STROKE);
    }

    // Left axis border
    ctx.strokeStyle = COLOURS.grid;
    ctx.lineWidth   = 0.5;
    ctx.beginPath();
    ctx.moveTo(plotLeft, rowTop);
    ctx.lineTo(plotLeft, rowTop + plotHeight);
    ctx.stroke();

    // Channel label
    ctx.fillStyle  = COLOURS.channelLabel;
    ctx.font       = "bold 11px monospace";
    ctx.textAlign  = "left";
    ctx.fillText(channelLabels[ch] ?? `CH_${ch + 1}`, plotLeft + 8, rowTop + 16);
  }

  // Frequency axis
  const axisY        = cssHeight - PADDING.bottom + 14;
  const tickInterval = niceFreqInterval(nyquistHz);

  ctx.fillStyle  = COLOURS.axisLabel;
  ctx.font       = "11px monospace";
  ctx.textAlign  = "center";

  for (let f = 0; f <= nyquistHz; f += tickInterval) {
    const x     = freqToX(f, nyquistHz, plotLeft, plotWidth);
    const label = f >= 1000 ? `${f / 1000}k` : `${f}`;
    ctx.fillText(label, x, axisY);
  }

  ctx.fillStyle  = COLOURS.axisLabel;
  ctx.font       = "11px monospace";
  ctx.textAlign  = "center";
  ctx.fillText("Frequency (Hz)", plotLeft + plotWidth / 2, cssHeight - 4);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface SpectrumComparisonProps {
  original: SpectrumData;
  filtered: SpectrumData | null;
  filteredLabel?: string;
}

export function SpectrumComparison({ original, filtered, filteredLabel = "Filtered" }: SpectrumComparisonProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showOriginal, setShowOriginal] = useState(true);
  const [showFiltered, setShowFiltered] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawSpectrumComparison(canvas, original, filtered, showOriginal, showFiltered);
  }, [original, filtered, showOriginal, showFiltered]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => {
      drawSpectrumComparison(canvas, original, filtered, showOriginal, showFiltered);
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [original, filtered, showOriginal, showFiltered]);

  const { original: origColor, filtered: filtColor } = COLOURS.legendColors;

  return (
    <div className="w-full">
      {/* Legend header — sits ABOVE the canvas, no axis overlap */}
      <div className="flex items-center justify-end gap-5 px-2 pb-2">
        <button
          type="button"
          onClick={() => setShowOriginal((v) => !v)}
          className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-widest transition-opacity cursor-pointer"
          style={{ color: origColor, opacity: showOriginal ? 1 : 0.3 }}
        >
          <span className="inline-block w-5" style={{ background: origColor, height: 2 }} />
          Original
        </button>
        {filtered && (
          <button
            type="button"
            onClick={() => setShowFiltered((v) => !v)}
            className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-widest transition-opacity cursor-pointer"
            style={{ color: filtColor, opacity: showFiltered ? 1 : 0.3 }}
          >
            <span className="inline-block w-5" style={{ background: filtColor, height: 2 }} />
            {filteredLabel}
          </button>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ display: "block" }}
        aria-label="Frequency spectrum comparison: original vs filtered audio"
      />
    </div>
  );
}
