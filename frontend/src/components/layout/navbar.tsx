"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [hash, setHash] = useState(() =>
    typeof window !== "undefined" ? window.location.hash : ""
  );
  const pathname = usePathname();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    
    const handleHashChange = () => {
      setHash(window.location.hash);
    };

    window.addEventListener("scroll", handleScroll);
    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, [pathname]);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="fixed inset-x-0 top-0 z-50 flex justify-center p-0 md:p-4 transition-all duration-500 pointer-events-none">
      <nav
        className={`pointer-events-auto flex w-full items-center justify-between transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          scrolled
            ? "h-16 max-w-5xl rounded-full bg-slate-700/30 px-6 backdrop-blur-sm shadow-lg shadow-black/20 translate-y-0"
            : "h-20 max-w-[1400px] rounded-none bg-transparent px-8 backdrop-blur-none shadow-none -translate-y-0"
        }`}
      >
        {/* Left: Logo + Brand */}
        <Link
          href="/"
          onClick={(e) => {
            if (pathname === "/") {
              e.preventDefault();
              window.scrollTo({ top: 0, behavior: "smooth" });
            }
          }}
          className="flex flex-1 items-center justify-start gap-3"
        >
          <img 
            src="/logo.png" 
            alt="Rigel Logo" 
            className="h-8 w-8 object-contain filter drop-shadow-[0_0_12px_rgba(255,255,255,0.5)]" 
          />
          <p 
            className={`font-bold text-white transition-all duration-700 leading-none mt-1 ${
              scrolled 
                ? "text-xl tracking-[0.2em]" 
                : "text-2xl tracking-[0.5em]"
            }`}
            style={{ fontFamily: "'Rivage', serif" }}          >
            RIGEL
          </p>
        </Link>

        {/* Center: Nav Links */}
        <div className="hidden items-center justify-center gap-2 md:flex p-1 rounded-full bg-white/5 border border-white/10" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
          {[
            { name: "Home", href: "/" },
            { name: "Desktop", href: "/#workspace" },
            { name: "Docs", href: "/#docs" }
          ].map((tab) => {
            const isActive = tab.name === "Home" ? pathname === "/" : tab.name === "Docs" ? pathname === "/docs" : false;
            
            return (
              <Link
                key={tab.name}
                href={tab.href}
                onClick={(e) => {
                  if (tab.href === "/" && pathname === "/") {
                    e.preventDefault();
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  } else if (tab.href.startsWith("/#") && pathname === "/") {
                    e.preventDefault();
                    scrollToSection(tab.href.substring(2));
                  }
                }}
                className={`relative px-6 py-2 text-sm font-medium transition-colors ${
                  isActive ? "text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                {isActive && (
                  <motion.div
                    layoutId="navbar-bubble"
                    className="absolute inset-0 z-[-1] rounded-full bg-white/10 shadow-[0_0_15px_rgba(255,255,255,0.1)] border border-white/20 backdrop-blur-md"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <span className="relative z-10">{tab.name}</span>
              </Link>
            );
          })}
        </div>

        {/* Right: Login/Download */}
        <div className="hidden flex-1 items-center justify-end gap-6 text-lg font-medium text-slate-300 md:flex" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
          <button className="transition-colors hover:text-white">Login</button>
          <button 
            className="group animate relative flex cursor-pointer items-center justify-center text-white bg-slate-600/80 border border-slate-500 h-10 px-6 hover:border-slate-500 hover:bg-slate-300/50 rounded-full overflow-hidden transition-colors"
          >
            <div
              className="before:absolute before:inset-[0] before:h-full before:w-full before:rounded-full before:p-[1px] before:will-change-[background-position] before:content-[''] before:![-webkit-mask-composite:xor] before:[background-image:var(--background-radial-gradient)] before:[background-size:300%_300%] before:![mask-composite:exclude] before:[mask:var(--mask-linear-gradient)] motion-safe:before:animate-[shine-pulse_var(--shine-pulse-duration)_infinite_linear]"
              style={{
                "--shine-pulse-duration": "14s",
                "--mask-linear-gradient": "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
                "--background-radial-gradient": "radial-gradient(transparent,transparent, #A07CFE,#FE8FB5,#FFBE7B,transparent,transparent)",
              } as React.CSSProperties}
            ></div>
            <span className="relative z-[1] flex items-center justify-center gap-2 text-base font-semibold tracking-wide text-white">
              Download
            </span>
          </button>
        </div>
      </nav>
    </div>
  );
}
