"use client";

import React, { useRef, useEffect, useState, useMemo } from "react";
import { VadResult, SpeechSegment } from "@/hooks/use-vad-history";

interface VadVisualizationProps {
  history: VadResult[];
  segments: SpeechSegment[];
  currentTime: number; // in seconds
  duration: number; // in seconds, total audio duration (0 if microphone)
  mode: "shared" | "microphone";
}

export function VadVisualization({ history, segments, currentTime, duration, mode }: VadVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 160 });

  // Handle Resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      setDimensions({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Time window bounds
  const windowSec = mode === "microphone" ? 10 : Math.max(duration, 1);
  
  const timeMin = useMemo(() => {
    if (mode === "shared") return 0;
    // For microphone, keep the current time at the right edge
    return Math.max(0, currentTime - windowSec);
  }, [mode, currentTime, windowSec]);

  const timeMax = useMemo(() => {
    if (mode === "shared") return windowSec;
    return Math.max(windowSec, currentTime);
  }, [mode, currentTime, windowSec]);

  // Render Canvas (Activity Curve)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = dimensions;
    // Handle High DPI displays
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    if (history.length === 0) return;

    const timeRange = timeMax - timeMin;
    if (timeRange <= 0) return;

    const getX = (t: number) => ((t - timeMin) / timeRange) * width;
    const getY = (val: number) => height - (val * height);

    // Draw Probability Curve
    ctx.beginPath();
    ctx.strokeStyle = "rgba(167, 139, 250, 0.8)"; // primary/accent color (purple-400)
    ctx.lineWidth = 1.5;

    let started = false;
    for (const res of history) {
      if (res.timestamp_seconds < timeMin) continue;
      const x = getX(res.timestamp_seconds);
      const y = getY(res.activity_score);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Fill under the curve
    if (started) {
      ctx.lineTo(getX(history[history.length - 1].timestamp_seconds), height);
      ctx.lineTo(getX(Math.max(timeMin, history[0].timestamp_seconds)), height);
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, "rgba(167, 139, 250, 0.2)");
      gradient.addColorStop(1, "rgba(167, 139, 250, 0.0)");
      ctx.fillStyle = gradient;
      ctx.fill();
    }
  }, [history, timeMin, timeMax, dimensions]);

  return (
    <div className="flex flex-col gap-2 w-full font-sans">
      {/* Legend / Header */}
      <div className="flex items-center gap-4 text-xs tracking-wider uppercase text-white/50 px-1">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-green-500/50" />
          <span>Speech Segments</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-purple-400" />
          <span>VAD Probability</span>
        </div>
      </div>

      <div 
        ref={containerRef} 
        className="relative w-full rounded-lg bg-black/40 border border-white/5 overflow-hidden"
        style={{ height: "160px" }}
      >
        {/* Canvas for Activity Score Curve */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 pointer-events-none"
          style={{ width: "100%", height: "100%" }}
        />

        {/* SVG overlay for Speech Segments and Playhead */}
        <svg 
          className="absolute inset-0 pointer-events-none" 
          width="100%" 
          height="100%"
          preserveAspectRatio="none"
        >
          {/* Segments */}
          {segments.map((seg, i) => {
            const startT = Math.max(timeMin, seg.start_time);
            const endT = Math.min(timeMax, seg.end_time !== null ? seg.end_time : currentTime);
            if (endT < timeMin || startT > timeMax) return null;

            const x = ((startT - timeMin) / (timeMax - timeMin)) * 100;
            const width = (((endT - startT) / (timeMax - timeMin)) * 100);

            return (
              <rect
                key={i}
                x={`${x}%`}
                y="0"
                width={`${Math.max(0, width)}%`}
                height="100%"
                fill={seg.is_active ? "rgba(34, 197, 94, 0.15)" : "rgba(255, 255, 255, 0.1)"}
                stroke={seg.is_active ? "rgba(34, 197, 94, 0.4)" : "rgba(255, 255, 255, 0.2)"}
                strokeWidth="1"
              />
            );
          })}

          {/* Playhead Cursor */}
          {currentTime >= timeMin && currentTime <= timeMax && (
            <line
              x1={`${((currentTime - timeMin) / (timeMax - timeMin)) * 100}%`}
              y1="0"
              x2={`${((currentTime - timeMin) / (timeMax - timeMin)) * 100}%`}
              y2="100%"
              stroke="white"
              strokeWidth="2"
              className="drop-shadow-[0_0_4px_rgba(255,255,255,0.8)]"
            />
          )}
        </svg>
      </div>
      
      {/* Time Axis Labels */}
      <div className="flex justify-between text-[10px] text-white/40 px-1 tabular-nums">
        <span>{formatTime(timeMin)}</span>
        <span>{formatTime(timeMax)}</span>
      </div>
    </div>
  );
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 10);
  if (m > 0) return `${m}:${s.toString().padStart(2, "0")}.${ms}`;
  return `${s}.${ms}s`;
}
