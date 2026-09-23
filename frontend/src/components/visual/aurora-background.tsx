"use client";

import { useEffect, useRef, useState } from "react";

export function AuroraBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stars, setStars] = useState<Array<{x: number, y: number, r: number, delay: number, duration: number}>>([]);

  useEffect(() => {
    // Generate stars for the SVG twinkling effect after hydration
    const newStars = Array.from({ length: 45 }).map(() => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      r: Math.random() * 1.2 + 0.5, // 0.5px to 1.7px
      delay: Math.random() * 5,     // 0 to 5s delay
      duration: Math.random() * 3 + 3, // 3 to 6s duration
    }));
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStars(newStars);
  }, []);

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

      {/* Tiny twinkling stars */}
      <svg className="absolute inset-0 h-full w-full opacity-60">
        <defs>
          <style>
            {`
              @keyframes twinkle {
                0%, 100% { opacity: 0.1; }
                50% { opacity: 0.9; }
              }
              .star {
                animation: twinkle var(--duration) ease-in-out infinite;
                animation-delay: var(--delay);
              }
              @media (prefers-reduced-motion: reduce) {
                .star { animation: none; opacity: 0.5; }
              }
            `}
          </style>
        </defs>
        {stars.map((star, i) => (
          <circle
            key={i}
            cx={`${star.x}%`}
            cy={`${star.y}%`}
            r={star.r}
            fill="#ffffff"
            className="star"
            style={{
              '--delay': `${star.delay}s`,
              '--duration': `${star.duration}s`,
            } as React.CSSProperties}
          />
        ))}
      </svg>

      {/* Interactive canvas waveform */}
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
    </div>
  );
}
