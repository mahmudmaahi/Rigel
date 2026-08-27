import { ArrowUpRight, AudioLines, CircleDot } from "lucide-react";

import { AudioUploader } from "@/components/audio/audio-uploader";
import { Card } from "@/components/ui/card";

export const metadata = {
  title: "Rigel — Intelligent Audio",
  description:
    "Analyze, enhance, and transform audio through transparent signal processing. Built from first principles by Team Orion.",
};

const capabilities = ["Analyze", "Enhance", "Transform", "About"];

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-5 py-5 sm:px-8 lg:px-10">
        <header className="flex items-center justify-between border-b border-border pb-5">
          <div className="flex items-end gap-4">
            <h1 className="text-5xl font-semibold leading-none tracking-normal sm:text-6xl">
              Rigel
            </h1>
            <p className="hidden pb-1 text-xs uppercase tracking-[0.22em] text-muted-foreground sm:block">
              Team Orion
            </p>
          </div>
          <nav className="hidden items-center gap-6 text-sm text-muted-foreground md:flex">
            {capabilities.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </nav>
        </header>

        {/* Hero section — editorial headline + feature cards */}
        <section className="grid gap-8 py-10 lg:grid-cols-[0.92fr_1.08fr] lg:items-start">
          <div className="max-w-3xl">
            <div className="mb-6 inline-flex items-center gap-2 border border-border bg-card px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
              <CircleDot className="h-3.5 w-3.5" />
              Transparent signal processing
            </div>
            <h2 className="max-w-3xl text-5xl font-semibold leading-[0.96] tracking-normal sm:text-6xl lg:text-7xl">
              Intelligent audio, built from first principles.
            </h2>
            <p className="mt-6 max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg">
              Analyze, enhance, and transform audio through transparent signal
              processing. Upload a WAV file and Rigel will decode the digital
              signal, visualize the waveform, and let you play it back.
            </p>

            <div className="mt-10 grid max-w-xl gap-px overflow-hidden border border-border bg-border sm:grid-cols-3">
              <Card className="border-0 p-5">
                <AudioLines className="mb-6 h-5 w-5" />
                <p className="text-sm font-medium">Load audio</p>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  Decode the file and read its digital samples.
                </p>
              </Card>
              <Card className="border-0 p-5">
                <ArrowUpRight className="mb-6 h-5 w-5" />
                <p className="text-sm font-medium">Visualize signal</p>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  See the waveform rendered from the real sample values.
                </p>
              </Card>
              <Card className="border-0 p-5">
                <CircleDot className="mb-6 h-5 w-5" />
                <p className="text-sm font-medium">Playback</p>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  Listen while the playhead tracks the waveform in real time.
                </p>
              </Card>
            </div>
          </div>

          {/* Upload + waveform panel */}
          <div>
            <AudioUploader />
          </div>
        </section>
      </div>
    </main>
  );
}
