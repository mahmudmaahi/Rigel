"use client";

import { useEffect, useState } from "react";

let hasShownSplash = false;

export function SplashScreen() {
  // Initialize directly from the module-level flag so we never call setState
  // synchronously inside an effect (which triggers a lint error).
  const [stage, setStage] = useState<"initial" | "slide-up" | "expand" | "done">(
    hasShownSplash ? "done" : "initial"
  );
  const [shouldRender, setShouldRender] = useState(!hasShownSplash);

  useEffect(() => {
    if (hasShownSplash) {
      // Already done — nothing to run.
      return;
    }

    const slideTimer = setTimeout(() => {
      setStage("slide-up");
    }, 200);

    const expandTimer = setTimeout(() => {
      setStage("expand");
    }, 1200);

    const doneTimer = setTimeout(() => {
      setStage("done");
      hasShownSplash = true;
    }, 2500);

    return () => {
      clearTimeout(slideTimer);
      clearTimeout(expandTimer);
      clearTimeout(doneTimer);
    };
  }, []);

  if (stage === "done" || !shouldRender) return null;

  return (
    <div className="fixed inset-0 z-[100] pointer-events-none overflow-hidden">
      {/* Expanding mask reveals the background underneath */}
      <svg className="absolute inset-0 h-full w-full">
        <defs>
          <mask id="splash-mask">
            <rect width="100%" height="100%" fill="white" />
            <circle
              cx="50%" cy="50%"
              r={stage === "expand" ? "150%" : "0%"}
              fill="black"
              style={{ transition: "r 1.2s cubic-bezier(0.65, 0, 0.35, 1)" }}
            />
          </mask>
        </defs>
        <rect width="100%" height="100%" fill="white" mask="url(#splash-mask)" />
      </svg>

      {/* Rigel Text */}
      <div className="absolute inset-0 flex items-center justify-center">
        <h1
          className={`text-6xl font-bold tracking-[0.2em] transition-all duration-1000 ease-[cubic-bezier(0.16,1,0.3,1)] ${
            stage === "initial"
              ? "translate-y-[15vh] text-black opacity-100"
              : stage === "slide-up"
              ? "translate-y-0 text-black opacity-100"
              : "translate-y-0 text-white opacity-0"
          }`}
          style={{ fontFamily: "'Rivage', serif" }}
        >
          RIGEL
        </h1>
      </div>
    </div>
  );
}
