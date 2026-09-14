"use client";

import { useEffect, useRef } from "react";

/**
 * Animated idle signal oscilloscope inside the dropzone when no audio is loaded.
 * Gently oscillates to invite interaction.
 */
export function DropzoneIdleWaveform() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let time = 0;

    function resize() {
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx?.scale(dpr, dpr);
    }

    resize();
    window.addEventListener("resize", resize);

    function animate() {
      if (!canvas || !ctx) return;

      const w = canvas.width / (window.devicePixelRatio || 1);
      const h = canvas.height / (window.devicePixelRatio || 1);

      // Read current theme colors directly from computed styles
      const computedStyle = getComputedStyle(canvas);
      const strokeColor = computedStyle.getPropertyValue("--foreground").trim() || "#000000";

      ctx.clearRect(0, 0, w, h);

      const centerY = h / 2;
      const points = 80;

      // Draw faint center guide
      ctx.strokeStyle = strokeColor;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 0.5;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(w, centerY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // Draw gently oscillating idle wave
      ctx.lineWidth = 1;
      ctx.beginPath();

      for (let i = 0; i <= points; i++) {
        const x = (i / points) * w;
        const progress = i / points;
        // Envelope so edges taper to center line
        const envelope = Math.sin(progress * Math.PI);
        const t = prefersReducedMotion ? 0 : time * 0.03;

        const wave =
          Math.sin(progress * 8 + t) * 12 * envelope +
          Math.sin(progress * 16 - t * 1.2) * 5 * envelope;

        if (i === 0) {
          ctx.moveTo(x, centerY + wave);
        } else {
          ctx.lineTo(x, centerY + wave);
        }
      }

      ctx.stroke();

      if (!prefersReducedMotion) {
        time++;
        frameRef.current = requestAnimationFrame(animate);
      }
    }

    animate();

    return () => {
      window.removeEventListener("resize", resize);
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-x-4 top-1/2 h-16 -translate-y-1/2 opacity-25 transition-opacity duration-300 group-hover:opacity-45 dark:opacity-35 dark:group-hover:opacity-55"
      aria-hidden="true"
    />
  );
}
