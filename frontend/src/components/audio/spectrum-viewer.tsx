"use client";

/**
 * SpectrumViewer — Module 04: Frequency Spectrum Display
 *
 * Renders the one-sided amplitude magnitude spectrum (dBFS) returned by
 * POST /api/audio/spectrum on an HTML5 Canvas.
 *
 * Axes:
 *   X — linear frequency (Hz), 0 to Nyquist.
 *       Linear scale directly maps to DFT bin mathematics: f[k] = k·Fs/N.
 *   Y — magnitude in dBFS (0 dBFS at top, more negative at bottom).
 *
 * One canvas row is rendered per audio channel (stereo: L above R).
 */

import { useEffect, useRef } from "react";
import type { SpectrumData } from "@/lib/api";
import { DB_MIN, DB_MAX, createColorMapGradient } from "@/lib/colors";

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const CHANNEL_HEIGHT = 160; // px per channel row
const PADDING = { top: 24, right: 20, bottom: 36, left: 56 }; // px

// Grid line positions (dBFS)
const DB_GRID_LINES = [-20, -40, -60, -80];

// Approximate number of frequency tick marks on the X axis
const FREQ_TICK_TARGET = 8;

const COLOURS = {
  background:    "#0d0d18",
  grid:          "rgba(255,255,255,0.07)",
  axisLabel:     "rgba(255,255,255,0.40)",
  channelLabel:  "rgba(255,255,255,0.55)",
  zeroDB:        "rgba(255,255,255,0.12)",  // 0 dBFS reference line
} as const;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Pick a "nice" frequency tick interval (in Hz) so that approximately
 * FREQ_TICK_TARGET ticks appear across the given Nyquist frequency range.
 */
function niceFreqInterval(nyquistHz: number): number {
  const raw = nyquistHz / FREQ_TICK_TARGET;
  const nice = [
    100, 200, 500,
    1_000, 2_000, 5_000,
    10_000, 20_000,
  ];
  return nice.find((n) => n >= raw) ?? nice[nice.length - 1];
}

/** Map a frequency value to a canvas X coordinate within the plot area. */
function freqToX(
  freqHz: number,
  nyquistHz: number,
  plotLeft: number,
  plotWidth: number,
): number {
  return plotLeft + (freqHz / nyquistHz) * plotWidth;
}

/** Map a dBFS value to a canvas Y coordinate within one channel row. */
function dbToY(
  db: number,
  rowTop: number,
  rowHeight: number,
): number {
  // DB_MAX (0 dBFS) → rowTop;  DB_MIN (e.g. -90) → rowTop + rowHeight
  const fraction = (DB_MAX - db) / (DB_MAX - DB_MIN);
  return rowTop + fraction * rowHeight;
}

// ---------------------------------------------------------------------------
// Drawing functions
// ---------------------------------------------------------------------------

function drawChannelRow(
  ctx: CanvasRenderingContext2D,
  bins: { frequency_hz: number; magnitude_dbfs: number }[],
  nyquistHz: number,
  rowTop: number,
  plotLeft: number,
  plotWidth: number,
  plotHeight: number,
  channelLabel: string,
  dpr: number,
) {
  const rowHeight = plotHeight;

  // --- Background ---
  ctx.fillStyle = COLOURS.background;
  ctx.fillRect(plotLeft, rowTop, plotWidth, rowHeight);

  // --- dBFS grid lines ---
  ctx.strokeStyle = COLOURS.grid;
  ctx.lineWidth = 1;

  for (const db of DB_GRID_LINES) {
    const y = dbToY(db, rowTop, rowHeight);
    ctx.beginPath();
    ctx.moveTo(plotLeft, y);
    ctx.lineTo(plotLeft + plotWidth, y);
    ctx.stroke();

    // dBFS label on the left
    ctx.fillStyle = COLOURS.axisLabel;
    ctx.font = `9px monospace`;
    ctx.textAlign = "right";
    ctx.fillText(`${db}`, plotLeft - 6, y + 3);
  }

  // 0 dBFS reference line (top)
  const yZero = dbToY(0, rowTop, rowHeight);
  ctx.strokeStyle = COLOURS.zeroDB;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plotLeft, yZero);
  ctx.lineTo(plotLeft + plotWidth, yZero);
  ctx.stroke();

  ctx.fillStyle = COLOURS.axisLabel;
  ctx.font = `9px monospace`;
  ctx.textAlign = "right";
  ctx.fillText("0", plotLeft - 6, yZero + 3);

  // --- Spectrum area fill + line ---
  if (bins.length === 0) return;

  const gradient = createColorMapGradient(ctx, 0, rowTop, 0, rowTop + rowHeight);

  ctx.beginPath();

  // Start from bottom-left corner of the plot area
  const firstX = freqToX(bins[0].frequency_hz, nyquistHz, plotLeft, plotWidth);
  const bottomY = rowTop + rowHeight;
  ctx.moveTo(firstX, bottomY);

  // Trace the top edge of the spectrum
  for (const bin of bins) {
    const x = freqToX(bin.frequency_hz, nyquistHz, plotLeft, plotWidth);
    const y = dbToY(
      Math.max(DB_MIN, Math.min(DB_MAX, bin.magnitude_dbfs)),
      rowTop,
      rowHeight,
    );
    ctx.lineTo(x, y);
  }

  // Close back to the bottom-right corner
  const lastX = freqToX(bins[bins.length - 1].frequency_hz, nyquistHz, plotLeft, plotWidth);
  ctx.lineTo(lastX, bottomY);
  ctx.closePath();

  ctx.fillStyle = gradient;
  ctx.globalAlpha = 0.5;
  ctx.fill();
  ctx.globalAlpha = 1.0;

  // Stroke the top edge only
  ctx.beginPath();
  ctx.moveTo(firstX, dbToY(
    Math.max(DB_MIN, Math.min(DB_MAX, bins[0].magnitude_dbfs)),
    rowTop, rowHeight,
  ));
  for (const bin of bins) {
    const x = freqToX(bin.frequency_hz, nyquistHz, plotLeft, plotWidth);
    const y = dbToY(
      Math.max(DB_MIN, Math.min(DB_MAX, bin.magnitude_dbfs)),
      rowTop,
      rowHeight,
    );
    ctx.lineTo(x, y);
  }

  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // --- Channel label ---
  ctx.fillStyle = COLOURS.channelLabel;
  ctx.font = `bold 10px monospace`;
  ctx.textAlign = "left";
  ctx.fillText(channelLabel, plotLeft + 8, rowTop + 16);
}

function drawFrequencyAxis(
  ctx: CanvasRenderingContext2D,
  nyquistHz: number,
  canvasHeight: number,
  plotLeft: number,
  plotWidth: number,
  dpr: number,
) {
  const tickInterval = niceFreqInterval(nyquistHz);
  const axisY = canvasHeight - PADDING.bottom + 14;

  ctx.fillStyle = COLOURS.axisLabel;
  ctx.font = `9px monospace`;
  ctx.textAlign = "center";

  for (let f = 0; f <= nyquistHz; f += tickInterval) {
    const x = freqToX(f, nyquistHz, plotLeft, plotWidth);
    const label = f >= 1000 ? `${f / 1000}k` : `${f}`;
    ctx.fillText(label, x, axisY);
  }

  // Axis title
  ctx.fillStyle = COLOURS.axisLabel;
  ctx.font = `9px monospace`;
  ctx.textAlign = "center";
  ctx.fillText("Frequency (Hz)", plotLeft + plotWidth / 2, canvasHeight - 4);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface SpectrumViewerProps {
  spectrum: SpectrumData;
}

export function SpectrumViewer({ spectrum }: SpectrumViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.offsetWidth;
    const nChannels = spectrum.channels.length;
    const channelGap = 12; // px gap between channel rows
    const cssHeight =
      PADDING.top +
      nChannels * CHANNEL_HEIGHT +
      (nChannels - 1) * channelGap +
      PADDING.bottom;

    // Set physical canvas size
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    canvas.style.height = `${cssHeight}px`;

    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const plotLeft = PADDING.left;
    const plotWidth = cssWidth - PADDING.left - PADDING.right;
    const nyquistHz = spectrum.nyquist_hz;

    const channelLabels =
      nChannels === 1 ? ["CH"] : ["CH_L", "CH_R", "CH_3", "CH_4"];

    for (let ch = 0; ch < spectrum.channels.length; ch++) {
      const rowTop =
        PADDING.top + ch * (CHANNEL_HEIGHT + channelGap);

      drawChannelRow(
        ctx,
        spectrum.channels[ch].bins,
        nyquistHz,
        rowTop,
        plotLeft,
        plotWidth,
        CHANNEL_HEIGHT,
        channelLabels[ch] ?? `CH_${ch + 1}`,
        dpr,
      );
    }

    drawFrequencyAxis(ctx, nyquistHz, cssHeight, plotLeft, plotWidth, dpr);
  }, [spectrum]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ display: "block" }}
      aria-label="Frequency magnitude spectrum"
    />
  );
}
