"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import Link from "next/link";
import { Footer } from "@/components/layout/footer";
import { Navbar } from "@/components/layout/navbar";
import { AuroraBackground } from "@/components/visual/aurora-background";
import { SplashScreen } from "@/components/visual/splash-screen";
import { Activity, AudioLines, FileAudio, Music, SlidersHorizontal, Waves } from "lucide-react";

const emptySubscribe = () => () => {};

let hasPlayedHomeAnimations = false;

// Dummy cards for the showcase
const SHOWCASE_ROW_1 = [
  { title: "Waveform Analysis", icon: Waves },
  { title: "Frequency Spectrum", icon: Activity },
  { title: "STFT Spectrogram", icon: AudioLines },
  { title: "Peak Envelope", icon: FileAudio },
  { title: "Digital Filters", icon: SlidersHorizontal },
  { title: "Audio Enhancement", icon: Music },
];

const SHOWCASE_ROW_2 = [
  { title: "Voice Activity Detection", icon: Activity },
  { title: "Parametric EQ", icon: SlidersHorizontal },
  { title: "Spectral Subtraction", icon: Waves },
  { title: "Phase Processing", icon: AudioLines },
  { title: "Wiener Filtering", icon: FileAudio },
  { title: "High-Resolution FFT", icon: Music },
];

function ShowcaseCard({ title, icon: Icon }: { title: string; icon: any }) {
  return (
    <div className="flex shrink-0 items-center gap-3 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md px-6 py-4 transition-colors hover:border-indigo-500/30 hover:bg-indigo-500/10">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-indigo-500/20 bg-indigo-500/10">
        <Icon className="h-5 w-5 text-indigo-400" />
      </div>
      <span className="font-medium text-white">{title}</span>
    </div>
  );
}

export default function Home() {
  const [animate, setAnimate] = useState(!hasPlayedHomeAnimations);

  useEffect(() => {
    window.scrollTo(0, 0);
    
    if (animate) {
      const timer = setTimeout(() => {
        hasPlayedHomeAnimations = true;
        setAnimate(false);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [animate]);

  const isClient = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );

  return (
    <>
      {isClient && <SplashScreen />}

      <div className="relative min-h-screen overflow-x-hidden bg-[#080811] text-white">
        {/* Aurora gradient background */}
        <AuroraBackground />

        {/* Navbar */}
        <Navbar />

        <main className="relative z-10">
          {/* Hero Section */}
          <section className="relative px-5 pt-32 pb-20 sm:px-8 lg:px-12">
            <div className="mx-auto max-w-6xl">
              {/* Headline */}
              <div
                className={`mt-12 text-center ${animate ? "animate-fade-in-up opacity-0" : ""}`}
                style={animate ? { animationDelay: "0.2s" } : {}}
              >
                <h1 
                  className="text-4xl md:text-5xl lg:text-[4.25rem] font-medium leading-[1.1] tracking-tight flex flex-col items-center justify-center gap-y-4"
                >
                  <div className={`flex gap-x-3 items-center ${animate ? "opacity-0 animate-slide-in-left" : ""}`} style={animate ? { animationDelay: "1.5s" } : {}}>
                    <span className="text-slate-400 font-trirong font-semibold">From</span>
                    <span className="text-white tracking-normal font-trirong font-semibold">Raw Signals</span>
                  </div>
                  <div className={`flex gap-x-3 items-center ${animate ? "opacity-0 animate-slide-in-right" : ""}`} style={animate ? { animationDelay: "1.8s" } : {}}>
                    <span className="text-slate-400 font-trirong font-semibold">to</span>
                    <span className="text-white tracking-normal font-trirong font-semibold">Pure Clarity</span>
                  </div>
                </h1>
                <p 
                  className={`mx-auto mt-8 max-w-2xl text-lg font-medium text-slate-400 sm:text-xl leading-relaxed ${animate ? "opacity-0 animate-fade-in-up" : ""}`}
                  style={{ animationDelay: animate ? "2.1s" : "0s", fontFamily: "'Century Gothic', sans-serif" }}
                >
                  Supercharging your audio analysis with Digital Signal Processing.
                </p>
              </div>

              {/* Explore the future CTA */}
              <div
                className={`mt-16 flex w-full items-center justify-center gap-2 ${animate ? "animate-fade-in-up opacity-0" : ""}`}
                style={animate ? { animationDelay: "2.4s" } : {}}
                id="cta"
              >
                <div className="hidden sm:block h-[1px] w-12 sm:w-40 rounded-full bg-gradient-to-l from-slate-500 to-transparent"></div>
                <div className="flex shrink-0 grow-0 flex-row items-center justify-between gap-5 rounded-full border border-slate-600 bg-slate-700/30 py-2 pl-4 pr-2 backdrop-blur-sm">
                  <p className="text-white/70 font-medium px-2" style={{fontFamily: "'Century Gothic', sans-serif" }}>Explore the future</p>
                  <button 
                    onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })} 
                    className="group block"
                  >
                      <div
                        className="animate relative flex cursor-pointer items-center justify-center text-white bg-slate-600/80 rounded-full border border-slate-500 h-10 px-6 hover:border-slate-500 hover:bg-slate-300/50 overflow-hidden transition-colors"
                      >
                        <div
                          className="before:absolute before:inset-[0] before:h-full before:w-full before:rounded-full before:p-[1px] before:will-change-[background-position] before:content-[''] before:![-webkit-mask-composite:xor] before:[background-image:var(--background-radial-gradient)] before:[background-size:300%_300%] before:![mask-composite:exclude] before:[mask:var(--mask-linear-gradient)] motion-safe:before:animate-[shine-pulse_var(--shine-pulse-duration)_infinite_linear]"
                          style={{
                            "--shine-pulse-duration": "14s",
                            "--mask-linear-gradient": "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
                            "--background-radial-gradient": "radial-gradient(transparent,transparent, #A07CFE,#FE8FB5,#FFBE7B,transparent,transparent)",
                          } as React.CSSProperties}
                        ></div>
                        <div className="relative z-[1] flex items-center justify-center gap-2 text-sm font-bold tracking-wide text-white">
                          Get Started
                        </div>
                      </div>
                  </button>
                </div>
                <div className="hidden sm:block h-[1px] w-12 sm:w-40 rounded-full bg-gradient-to-r from-slate-500 to-transparent"></div>
              </div>
            </div>
          </section>

          {/* Feature Showcase / Product Preview Section */}
          <section id="features" className="relative overflow-hidden py-20 pb-32">
            {/* Fades on the sides */}
            <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-32 bg-gradient-to-r from-[#0A0A14] to-transparent"></div>
            <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-32 bg-gradient-to-l from-[#0A0A14] to-transparent"></div>

            <div className="flex flex-col gap-6 mt-8">
              {/* Row 1 - moves left */}
              <div className="flex w-max animate-marquee-left items-center gap-6">
                {[...SHOWCASE_ROW_1, ...SHOWCASE_ROW_1, ...SHOWCASE_ROW_1].map((item, i) => (
                  <ShowcaseCard key={`r1-${i}`} title={item.title} icon={item.icon} />
                ))}
              </div>

              {/* Row 2 - moves right */}
              <div className="flex w-max animate-marquee-right items-center gap-6 -ml-[50%]">
                {[...SHOWCASE_ROW_2, ...SHOWCASE_ROW_2, ...SHOWCASE_ROW_2].map((item, i) => (
                  <ShowcaseCard key={`r2-${i}`} title={item.title} icon={item.icon} />
                ))}
              </div>
            </div>
            
            <div className="mx-auto max-w-2xl text-center mt-32 relative z-20">
              <h2 className="text-sm font-bold tracking-widest text-slate-400 uppercase mb-6" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
                EXPLORE WHAT&apos;S HIDDEN
              </h2>
              <p className="text-xl md:text-2xl text-white/90 leading-relaxed font-medium" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
                Go beyond the waveform. Explore the structures, patterns, and possibilities hidden inside your audio.
              </p>
              <div className="mt-10">
                <Link 
                  href="/playground/analysis" 
                  className="inline-flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-400 hover:text-white transition-colors group"
                >
                  Explore the Playground 
                  <span className="transition-transform group-hover:translate-x-1">→</span>
                </Link>
              </div>
            </div>
          </section>

          {/* Why Rigel Section */}
          <section
            id="docs"
            className="relative px-5 py-24 sm:px-8 lg:px-12 border-t border-white/5"
          >
            <div className="mx-auto max-w-2xl text-center">
              <h2 className="text-sm font-bold tracking-widest text-slate-400 uppercase mb-6" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
                Why Rigel
              </h2>
              <p className="text-xl md:text-2xl text-white/90 leading-relaxed font-medium" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
                Named for the brilliant blue star of Orion, Rigel is an audio laboratory built around the same principle: make the invisible visible.
              </p>
              <div className="mt-10">
                <Link 
                  href="/docs" 
                  className="inline-flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-400 hover:text-white transition-colors group"
                >
                  Explore the story 
                  <span className="transition-transform group-hover:translate-x-1">→</span>
                </Link>
              </div>
            </div>
          </section>
        </main>

        <Footer />
      </div>
    </>
  );
}
