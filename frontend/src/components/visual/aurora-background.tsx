"use client";

import { useEffect, useRef } from "react";

export function AuroraBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let time = 0;
    let frameId: number;

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

      ctx.clearRect(0, 0, w, h);

      // Waveform removed per user request

      if (!prefersReducedMotion) {
        time++;
        frameId = requestAnimationFrame(animate);
      }
    }

    animate();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(frameId);
    };
  }, []);

  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden" style={{ zIndex: 0 }}>
      {/* Aurora glowing gradient blobs */}
      <div className="absolute -top-[20%] left-1/2 h-[500px] w-[800px] -translate-x-1/2 rounded-full bg-gradient-to-tr from-indigo-600/20 via-purple-600/20 to-cyan-500/10 blur-[120px] animate-aurora-pulse" />
      <div className="absolute top-[30%] -left-[10%] h-[450px] w-[550px] rounded-full bg-gradient-to-tr from-purple-800/15 to-indigo-600/10 blur-[100px] animate-float" />
      <div className="absolute top-[50%] -right-[10%] h-[400px] w-[500px] rounded-full bg-gradient-to-br from-cyan-600/15 via-indigo-700/10 to-transparent blur-[100px] animate-float-delayed" />
      <div className="absolute -bottom-[10%] left-1/3 h-[350px] w-[600px] rounded-full bg-gradient-to-tr from-indigo-900/15 via-purple-900/15 to-transparent blur-[120px]" />

      {/* Tiny shining stars */}
      <div 
        className="absolute inset-0 opacity-[0.25] animate-aurora-pulse" 
        style={{
          backgroundImage: `
            radial-gradient(1.5px 1.5px at 15% 25%, white, transparent),
            radial-gradient(1.5px 1.5px at 75% 15%, white, transparent),
            radial-gradient(2px 2px at 85% 65%, white, transparent),
            radial-gradient(1.5px 1.5px at 50% 45%, white, transparent),
            radial-gradient(1px 1px at 30% 10%, white, transparent),
            radial-gradient(1.5px 1.5px at 80% 80%, white, transparent),
            radial-gradient(1px 1px at 5% 50%, white, transparent),
            radial-gradient(1.5px 1.5px at 45% 90%, white, transparent),
            radial-gradient(1px 1px at 95% 20%, white, transparent),
            radial-gradient(1px 1px at 20% 95%, white, transparent),
            radial-gradient(1.5px 1.5px at 60% 5%, white, transparent),
            radial-gradient(1px 1px at 40% 70%, white, transparent)
          `
        }} 
      />
      <div 
        className="absolute inset-0 z-0 animate-[pulse_6s_ease-in-out_infinite_alternate]"
        style={{
          backgroundImage: `
            radial-gradient(1.5px 1.5px at 25% 75%, white, transparent),
            radial-gradient(2px 2px at 10% 85%, white, transparent),
            radial-gradient(1.5px 1.5px at 90% 35%, white, transparent),
            radial-gradient(1px 1px at 60% 90%, white, transparent),
            radial-gradient(1.5px 1.5px at 40% 30%, white, transparent),
            radial-gradient(1.5px 1.5px at 70% 50%, white, transparent),
            radial-gradient(2px 2px at 20% 60%, white, transparent),
            radial-gradient(1px 1px at 85% 10%, white, transparent),
            radial-gradient(1.5px 1.5px at 5% 25%, white, transparent),
            radial-gradient(1px 1px at 55% 15%, white, transparent),
            radial-gradient(1.5px 1.5px at 75% 85%, white, transparent),
            radial-gradient(1px 1px at 35% 55%, white, transparent)
          `
        }} 
      />

      {/* Interactive canvas waveform */}
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
    </div>
  );
}
