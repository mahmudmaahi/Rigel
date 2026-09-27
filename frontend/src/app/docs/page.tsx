import Link from "next/link";
import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";
import { Card } from "@/components/ui/card";
import { Waves, SlidersHorizontal, Music, Activity, Mic, Image as ImageIcon, ArrowRight } from "lucide-react";

export const WORKSPACES = [
  {
    slug: "analysis",
    name: "Audio Analysis",
    path: "/playground/analysis",
    icon: Waves,
    description:
      "See a recording in both the time and frequency domains at once — waveform, frequency spectrum, and STFT spectrogram — computed directly from the samples you upload.",
  },
  {
    slug: "filtering",
    name: "Filtering",
    path: "/playground/filtering",
    icon: SlidersHorizontal,
    description:
      "Design classical Butterworth and Elliptic digital filters — low-pass, high-pass, band-pass, band-stop — and hear the result immediately, alongside a live theoretical frequency-response curve.",
  },
  {
    slug: "noise-removal",
    name: "Noise Removal",
    path: "/playground/enhancement",
    icon: Music,
    description:
      "Estimate background noise and suppress it with four classical statistical methods — Improved Spectral Subtraction, Decision-Directed Wiener, Log-MMSE, and IMCRA + OM-LSA — each a different point on the tradeoff between suppression and speech distortion.",
  },
  {
    slug: "vad",
    name: "Voice Activity Detection",
    path: "/playground/speech",
    icon: Activity,
    description:
      "Stream microphone or recorded audio through a VAD engine in real time, and watch speech/silence decisions arrive frame by frame over a live WebSocket connection.",
  },
  {
    slug: "voice-lab",
    name: "Voice Lab",
    path: "/playground/voice-lab",
    icon: Mic,
    description:
      "Measure a voice's pitch (via the YIN algorithm), loudness, and rhythm, then reshape it with an ordered effect chain — Gain, Pitch Shift, Speed, Time Stretch, Echo, Reverb, Chorus, Distortion — applied exactly in the order you arrange it.",
  },
  {
    slug: "audio-image",
    name: "Audio ↔ Image",
    path: "/playground/image",
    icon: ImageIcon,
    description:
      "Encode audio losslessly into a PNG data image using the RGL1 protocol, and decode that same image back into the original audio, sample for sample.",
  },
];

export default function DocsPage() {
  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#080811] text-white transition-colors duration-300">
      <Navbar />

      <main className="relative z-10 pt-32 pb-24">
        {/* Header Section */}
        <section className="px-5 sm:px-8 lg:px-12 text-center max-w-4xl mx-auto mb-16">
          <h1
            className="text-4xl md:text-5xl lg:text-6xl font-medium tracking-tight mb-4"
            style={{ fontFamily: "'Marlino Regular', serif" }}
          >
            RIGEL
          </h1>
          <p className="text-xl text-slate-400">
            Intelligent audio, built from first principles.
          </p>
          <p className="mt-6 max-w-2xl mx-auto text-base text-slate-400 leading-relaxed">
            Rigel is a signal-processing laboratory: every effect and measurement in the
            Playground is a classical, understandable algorithm — not a black box — so you can
            see exactly how a recording is being analyzed or transformed.
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
                <h2 className="text-sm font-bold tracking-widest text-slate-500 uppercase">
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
                <h2 className="text-sm font-bold tracking-widest text-slate-500 uppercase">
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

        {/* Playground Workspaces */}
        <section className="px-5 sm:px-8 lg:px-12 max-w-6xl mx-auto mb-32 border-t border-white/5 pt-24">
          <div className="text-center mb-14">
            <h2 className="text-sm font-bold tracking-widest text-slate-500 uppercase mb-3">
              The Playground
            </h2>
            <p className="text-lg text-white/80 max-w-2xl mx-auto">
              Six workspaces, following the same signal from raw waveform to full transformation.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {WORKSPACES.map((ws) => {
              const Icon = ws.icon;
              return (
                <Card key={ws.slug} className="h-full p-6 flex flex-col gap-4">
                  <Link href={ws.path} className="group flex flex-col gap-4 flex-1">
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-indigo-500/20 bg-indigo-500/10 transition-colors group-hover:bg-indigo-500/20">
                      <Icon className="h-5 w-5 text-indigo-400" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-white mb-2">{ws.name}</h3>
                      <p className="text-sm text-slate-400 leading-relaxed">{ws.description}</p>
                    </div>
                    <span className="inline-flex items-center gap-1 text-xs font-semibold tracking-wide text-indigo-400 opacity-0 group-hover:opacity-100 transition-opacity">
                      Open workspace
                      <ArrowRight className="h-3 w-3" />
                    </span>
                  </Link>
                  <Link
                    href={`/docs/${ws.slug}`}
                    className="inline-flex items-center gap-1 text-xs font-semibold tracking-wide text-slate-400 hover:text-white border-t border-white/5 pt-4 transition-colors"
                  >
                    Learn more — architecture &amp; formulas
                    <ArrowRight className="h-3 w-3" />
                  </Link>
                </Card>
              );
            })}
          </div>
        </section>

      </main>

      <Footer />
    </div>
  );
}
