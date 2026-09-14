"use client";

/**
 * SpectrogramViewer — Module 05: STFT Spectrogram Display
 *
 * Renders an STFT spectrogram heat-map on an HTML5 Canvas.
 *
 * Layout:
 *   X axis — time (seconds), left to right.
 *   Y axis — frequency (Hz), bottom to top (low freq at bottom, matching
 *             physical intuition: bass at the bottom, treble at the top).
 *
 * Colour map:
 *   Maps the dBFS range [DB_MIN, 0] to a perceptually-ordered gradient:
 *     DB_MIN (silent) → deep indigo/navy
 *     −60 dBFS       → indigo/purple
 *     −30 dBFS       → cyan/teal
 *     0  dBFS (loud) → bright white/yellow
 *
 * One canvas row is rendered per audio channel (stereo: L above R),
 * using the same panel layout as WaveformViewer and SpectrumViewer.
 */

import { useEffect, useRef } from "react";
import type { SpectrogramData } from "@/lib/api";
import { DB_MIN, DB_MAX, getRgbForT, dbToT } from "@/lib/colors";

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const CHANNEL_HEIGHT = 200;  // px per channel panel
const PADDING = { top: 24, right: 20, bottom: 40, left: 60 }; // px

// dBFS display range (imported from lib/colors)

// Approximate number of time tick marks on the X axis
const TIME_TICK_TARGET = 8;
// Approximate number of frequency tick marks on the Y axis
const FREQ_TICK_TARGET = 6;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Pick a "nice" time tick interval so that ~TIME_TICK_TARGET ticks appear. */
function niceTimeInterval(durationS: number): number {
  const raw = durationS / TIME_TICK_TARGET;
  const candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300];
  return candidates.find((c) => c >= raw) ?? candidates[candidates.length - 1];
}

/** Pick a "nice" frequency tick interval so that ~FREQ_TICK_TARGET ticks appear. */
function niceFreqInterval(nyquistHz: number): number {
  const raw = nyquistHz / FREQ_TICK_TARGET;
  const candidates = [100, 200, 500, 1000, 2000, 5000, 10000, 20000];
  return candidates.find((c) => c >= raw) ?? candidates[candidates.length - 1];
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function drawSpectrogramPanel(
  ctx: CanvasRenderingContext2D,
  channel: { time_frames: number[]; freq_bins: number[]; magnitudes: number[][] },
  rowTop: number,
  plotLeft: number,
  plotWidth: number,
  plotHeight: number,
  channelLabel: string,
  dpr: number,
) {
  const { time_frames, freq_bins, magnitudes } = channel;
  const nTime = time_frames.length;
  const nFreq = freq_bins.length;

  if (nTime === 0 || nFreq === 0) return;

  const durationS = time_frames[time_frames.length - 1];
  const nyquistHz = freq_bins[freq_bins.length - 1];

  const cellW = plotWidth / nTime;
  const cellH = plotHeight / nFreq;

  // --- Background ---
  ctx.fillStyle = "#0d0d18";
  ctx.fillRect(plotLeft, rowTop, plotWidth, plotHeight);

  // --- Draw spectrogram cells ---
  // freq_bins is ascending (index 0 = low freq), but canvas Y is top-down.
  // So freq index 0 (low freq) → bottom of panel, last freq → top of panel.
  for (let t = 0; t < nTime; t++) {
    for (let f = 0; f < nFreq; f++) {
      const db = magnitudes[t][f];
      const [r, g, b] = getRgbForT(dbToT(db));
      ctx.fillStyle = `rgb(${r},${g},${b})`;

      const x = plotLeft + t * cellW;
      // Flip Y: freq index 0 is at the bottom
      const y = rowTop + plotHeight - (f + 1) * cellH;
      ctx.fillRect(x, y, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  // --- Subtle grid overlay ---
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 0.5;

  // Time grid lines
  const timeInterval = niceTimeInterval(durationS);
  for (let t = 0; t <= durationS; t += timeInterval) {
    const x = plotLeft + (t / durationS) * plotWidth;
    ctx.beginPath();
    ctx.moveTo(x, rowTop);
    ctx.lineTo(x, rowTop + plotHeight);
    ctx.stroke();
  }

  // Frequency grid lines
  const freqInterval = niceFreqInterval(nyquistHz);
  for (let f = 0; f <= nyquistHz; f += freqInterval) {
    const y = rowTop + plotHeight - (f / nyquistHz) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(plotLeft, y);
    ctx.lineTo(plotLeft + plotWidth, y);
    ctx.stroke();
  }

  // --- Frequency axis labels (left) ---
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "9px monospace";
  ctx.textAlign = "right";

  for (let f = 0; f <= nyquistHz; f += freqInterval) {
    const y = rowTop + plotHeight - (f / nyquistHz) * plotHeight;
    const label = f >= 1000 ? `${f / 1000}k` : `${f}`;
    ctx.fillText(label, plotLeft - 6, y + 3);
  }

  // Y-axis unit label
  ctx.save();
  ctx.translate(plotLeft - 44, rowTop + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.font = "9px monospace";
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.fillText("Hz", 0, 0);
  ctx.restore();

  // --- Channel label ---
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "left";
  ctx.fillText(channelLabel, plotLeft + 8, rowTop + 16);
}

function drawTimeAxis(
  ctx: CanvasRenderingContext2D,
  durationS: number,
  canvasHeight: number,
  plotLeft: number,
  plotWidth: number,
) {
  const interval = niceTimeInterval(durationS);
  const axisY = canvasHeight - PADDING.bottom + 14;

  ctx.fillStyle = "rgba(255,255,255,0.40)";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";

  for (let t = 0; t <= durationS; t += interval) {
    const x = plotLeft + (t / durationS) * plotWidth;
    const label = t >= 60
      ? `${Math.floor(t / 60)}m${(t % 60).toFixed(0)}s`
      : `${t.toFixed(1)}s`;
    ctx.fillText(label, x, axisY);
  }

  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.font = "9px monospace";
  ctx.fillText("Time (s)", plotLeft + plotWidth / 2, canvasHeight - 4);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface SpectrogramViewerProps {
  spectrogram: SpectrogramData;
}

export function SpectrogramViewer({ spectrogram }: SpectrogramViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.offsetWidth;
    const nChannels = spectrogram.channels.length;
    const channelGap = 12;
    const cssHeight =
      PADDING.top +
      nChannels * CHANNEL_HEIGHT +
      (nChannels - 1) * channelGap +
      PADDING.bottom;

    // Physical canvas size
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    canvas.style.height = `${cssHeight}px`;

    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const plotLeft = PADDING.left;
    const plotWidth = cssWidth - PADDING.left - PADDING.right;

    const channelLabels =
      nChannels === 1 ? ["CH"] : ["CH_L", "CH_R", "CH_3", "CH_4"];

    const durationS =
      spectrogram.channels[0]?.time_frames.slice(-1)[0] ?? spectrogram.duration_seconds;

    for (let ch = 0; ch < nChannels; ch++) {
      const rowTop = PADDING.top + ch * (CHANNEL_HEIGHT + channelGap);
      drawSpectrogramPanel(
        ctx,
        spectrogram.channels[ch],
        rowTop,
        plotLeft,
        plotWidth,
        CHANNEL_HEIGHT,
        channelLabels[ch] ?? `CH_${ch + 1}`,
        dpr,
      );
    }

    drawTimeAxis(ctx, durationS, cssHeight, plotLeft, plotWidth);
  }, [spectrogram]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ display: "block" }}
      aria-label="STFT spectrogram — time-frequency magnitude"
    />
  );
}
