import React from "react";

export function RigelLogo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <div className={`relative flex items-center justify-center ${className}`}>
      {/* Outer deep blue glow */}
      <div className="absolute inset-0 rounded-full bg-blue-600 blur-[8px] opacity-70 animate-pulse"></div>
      
      {/* Inner bright blue glow */}
      <div className="absolute inset-1 rounded-full bg-blue-400 blur-[4px]"></div>
      
      {/* Core bright white-cyan */}
      <div className="absolute inset-2 rounded-full bg-cyan-100 shadow-[0_0_10px_#fff]"></div>
      
      {/* Texture details mimicking the stellar surface */}
      <div className="absolute inset-1 rounded-full bg-[radial-gradient(circle_at_center,transparent_30%,rgba(0,100,255,0.4)_100%)] mix-blend-overlay"></div>
      
      {/* Spikes / Rays */}
      <svg 
        className="absolute inset-[-50%] w-[200%] h-[200%] text-blue-300 opacity-60 animate-[spin_20s_linear_infinite]" 
        viewBox="0 0 100 100" 
        fill="currentColor"
      >
        <path d="M50 10 L52 45 L85 50 L52 55 L50 90 L48 55 L15 50 L48 45 Z" className="blur-[1px]" />
        <path d="M50 20 L51 47 L75 50 L51 53 L50 80 L49 53 L25 50 L49 47 Z" className="text-white opacity-80" />
      </svg>
    </div>
  );
}
