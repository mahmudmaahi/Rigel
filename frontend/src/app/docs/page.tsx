import Link from "next/link";
import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";

export default function DocsPage() {
  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#080811] text-white transition-colors duration-300">
      <Navbar />

      <main className="relative z-10 pt-32 pb-24">
        {/* Header Section */}
        <section className="px-5 sm:px-8 lg:px-12 text-center max-w-4xl mx-auto mb-20">
          <h1 
            className="text-4xl md:text-5xl lg:text-6xl font-medium tracking-tight mb-4"
            style={{ fontFamily: "'Marlino Regular', serif" }}
          >
            RIGEL
          </h1>
          <p className="text-xl text-slate-400" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
            The signal laboratory.
          </p>
        </section>

        {/* The Story of Rigel & Orion */}
        <section className="px-5 sm:px-8 lg:px-12 max-w-6xl mx-auto mb-32">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-16 lg:gap-24 items-start">
            
            {/* Left Column: Context (Orion) */}
            <div className="lg:col-span-7 flex flex-col gap-8">
              <div className="relative overflow-hidden border border-white/10 bg-[#0c0c16] p-3">
                <div className="relative aspect-[16/9] w-full overflow-hidden">
                  <img 
                    src="/orion.jpg" 
                    alt="Orion constellation" 
                    className="h-full w-full object-cover grayscale-[40%] hover:grayscale-0 transition-all duration-1000"
                  />
                </div>
              </div>
              <div className="space-y-5 max-w-xl">
                <h2 className="text-sm font-bold tracking-widest text-slate-500 uppercase" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
                  Orion — The Constellation
                </h2>
                <p className="text-lg leading-relaxed text-white/90">
                  Orion is one of the most recognizable constellations in the night sky. Its distinctive arrangement of stars is traditionally imagined as a hunter, with a raised arm and weapon. 
                </p>
                <p className="text-lg leading-relaxed text-slate-400">
                  We use Orion as the symbol of the larger vision: many individual elements coming together to form something recognizable.
                </p>
              </div>
            </div>

            {/* Right Column: Focus (Rigel) */}
            <div className="lg:col-span-5 flex flex-col gap-8 lg:mt-32">
              <div className="relative overflow-hidden border border-white/10 bg-[#0c0c16] p-3">
                <div className="relative aspect-[4/5] w-full overflow-hidden">
                  <img 
                    src="/rigel.jpg" 
                    alt="Rigel star" 
                    className="h-full w-full object-cover grayscale-[20%] hover:grayscale-0 hover:scale-105 transition-all duration-1000"
                  />
                </div>
              </div>
              <div className="space-y-5">
                <h2 className="text-sm font-bold tracking-widest text-slate-500 uppercase" style={{ fontFamily: "'Century Gothic', sans-serif" }}>
                  Rigel — The Star
                </h2>
                <p className="text-lg leading-relaxed text-white/90">
                  Rigel is the brilliant blue-white star marking Orion&apos;s foot and is the brightest star in the constellation by apparent magnitude.
                </p>
                <p className="text-lg leading-relaxed text-slate-400">
                  Rigel represents our focus within the larger system.
                </p>
              </div>
            </div>

          </div>

          {/* The DSP Connection */}
          <div className="mt-40 max-w-3xl mx-auto text-center border-t border-white/10 pt-24">
            <h3 className="text-3xl md:text-4xl font-medium mb-10 text-white tracking-tight" style={{ fontFamily: "'Marlino Regular', serif" }}>
              From Orion&apos;s stars to the structure of a signal
            </h3>
            <div className="space-y-8 text-lg text-slate-400 leading-relaxed">
              <p>
                A recording initially appears to be a single stream of samples. Rigel explores what happens when we look closer—from <span className="font-mono text-indigo-300">x[n]</span> in the time domain to its frequency components, time-frequency structure, and eventually the transformations that allow us to understand and manipulate sound.
              </p>
              <p className="text-xl text-white/90 font-medium tracking-wide pt-4" style={{ fontFamily: "'Marlino Regular', serif" }}>
                Team Orion builds Rigel from the signal outward
              </p>
            </div>
          </div>
        </section>


      </main>

      <Footer />
    </div>
  );
}
