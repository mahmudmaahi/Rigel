"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { usePlayground, PlaygroundProvider } from "@/contexts/playground-context";
import { AuroraBackground } from "@/components/visual/aurora-background";
import { FileAudio, Loader2, Music, SlidersHorizontal, Activity, Waves, Download, Menu, ChevronLeft, ChevronRight } from "lucide-react";
import { AudioUploader } from "@/components/audio/audio-uploader";
import { useState } from "react";

function formatBytes(bytes: number, decimals = 2) {
  if (!+bytes) return "0 Bytes";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

const PLAYGROUND_NAV = [
  { name: "Audio Analysis", path: "/playground/analysis", icon: Waves, enabled: true },
  { name: "Filtering", path: "/playground/filtering", icon: SlidersHorizontal, enabled: true },
  { name: "Noise Removal", path: "/playground/enhancement", icon: Music, enabled: false },
  { name: "Speech / VAD", path: "/playground/speech", icon: Activity, enabled: false },
];

function PlaygroundSidebar() {
  const pathname = usePathname();
  const { state, clearAudio } = usePlayground();

  return (
    <div className="flex h-full w-full flex-col border-r border-white/5 bg-[#080811]/50 backdrop-blur-xl lg:w-72">
      <div className="p-8 pb-4">
        <Link href="/" className="flex items-center gap-3">
          <img 
            src="/logo.png" 
            alt="Rigel Logo" 
            className="h-7 w-7 object-contain filter drop-shadow-[0_0_20px_rgba(255,255,255,0.7)]" 
          />
          <p 
            className="font-bold text-white text-xl tracking-[0.2em] mt-1"
            style={{ fontFamily: "'Rivage', serif" }}
          >
            RIGEL
          </p>
        </Link>
      </div>

      {/* Shared Audio Area */}
      <div className="px-6 py-4 border-b border-white/5">
        <h3 className="mb-4 text-xs font-bold tracking-widest text-slate-500 uppercase">Current Audio</h3>
        {state.status === "success" ? (
          <div className="rounded-2xl border border-indigo-500/20 bg-indigo-500/10 p-4">
            <div className="flex items-center gap-3 mb-3">
              <FileAudio className="h-5 w-5 text-indigo-400 shrink-0" />
              <div className="overflow-hidden">
                <p className="truncate text-sm font-medium text-white">{state.file.name}</p>
                <p className="text-xs text-indigo-300/70 font-mono">
                  {formatBytes(state.file.size)}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 mb-4">
              <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] font-mono text-indigo-300">
                {state.metadata.sample_rate_hz}Hz
              </span>
              <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] font-mono text-indigo-300">
                {state.metadata.channels === 1 ? 'Mono' : 'Stereo'}
              </span>
              <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] font-mono text-indigo-300">
                {state.metadata.duration_seconds.toFixed(2)}s
              </span>
            </div>
            <button
              onClick={clearAudio}
              className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-white/10 hover:text-white transition-colors"
            >
              Change File ↺
            </button>
          </div>
        ) : state.status === "loading" ? (
          <div className="rounded-2xl border border-white/5 bg-white/5 p-6 flex flex-col items-center justify-center text-center">
             <Loader2 className="h-6 w-6 animate-spin text-indigo-400 mb-2" />
             <p className="text-sm text-slate-300">Processing Audio...</p>
          </div>
        ) : (
          <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-4 text-center">
            <p className="text-xs text-slate-400 mb-2">No audio loaded</p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <h3 className="mb-3 px-2 text-xs font-bold tracking-widest text-slate-500 uppercase">Workspaces</h3>
        <nav className="flex flex-col gap-1">
          {PLAYGROUND_NAV.map((item) => {
            const isActive = pathname === item.path;
            const Icon = item.icon;
            
            return item.enabled ? (
              <Link
                key={item.path}
                href={item.path}
                className={[
                  "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-indigo-500/10 text-indigo-300"
                    : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                ].join(" ")}
              >
                <Icon className="h-4 w-4" />
                {item.name}
              </Link>
            ) : (
              <div
                key={item.path}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-600 cursor-not-allowed"
                title="Future Feature"
              >
                <Icon className="h-4 w-4 opacity-50" />
                {item.name}
                <span className="ml-auto text-[9px] uppercase tracking-wider bg-slate-800/50 px-1.5 py-0.5 rounded text-slate-500">Soon</span>
              </div>
            );
          })}
        </nav>
      </div>

      {/* Download Section */}
      <div className="p-6 border-t border-white/5">
        <h3 className="mb-3 text-xs font-bold tracking-widest text-slate-500 uppercase">Download</h3>
        
        {state.status === "success" && state.processedAudio ? (
          <button 
            onClick={() => {
              const url = URL.createObjectURL(state.processedAudio!);
              const a = document.createElement('a');
              a.href = url;
              a.download = 'rigel_processed.wav';
              a.click();
              URL.revokeObjectURL(url);
            }}
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-indigo-600/20 border border-indigo-500/30 px-3 py-2.5 text-sm font-medium text-indigo-300 hover:bg-indigo-600/40 hover:text-white transition-colors"
          >
            <Download className="h-4 w-4" />
            Download WAV
          </button>
        ) : (
          <button 
            disabled
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-white/[0.02] border border-white/5 px-3 py-2.5 text-sm font-medium text-slate-600 cursor-not-allowed"
          >
            <Download className="h-4 w-4 opacity-50" />
            No processed audio
          </button>
        )}
      </div>
    </div>
  );
}

function PlaygroundShell({ children }: { children: React.ReactNode }) {
  const { state } = usePlayground();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className="relative flex h-screen w-full overflow-hidden bg-[#080811] text-white">
      <AuroraBackground />
      
      {/* Sidebar with Drawer Animation */}
      <div className={`transition-all duration-300 ease-in-out relative z-30 flex-shrink-0 overflow-hidden ${isSidebarOpen ? 'w-64 lg:w-72' : 'w-0'}`}>
        <div className="w-64 lg:w-72 h-full absolute top-0 left-0">
          <PlaygroundSidebar />
        </div>
      </div>

      {/* Main Workspace Area */}
      <main className="relative z-10 flex flex-1 flex-col overflow-y-auto overflow-x-hidden">
        {/* Drawer Toggle */}
        <div className="sticky top-0 z-40 p-4 lg:p-6 pb-0 w-full flex justify-start">
          <button 
             onClick={() => setIsSidebarOpen(!isSidebarOpen)}
             className="flex items-center justify-center rounded-xl p-2 bg-white/5 border border-white/10 hover:bg-white/10 text-slate-400 hover:text-white transition-colors backdrop-blur-md"
             title="Toggle Sidebar"
          >
             {isSidebarOpen ? <ChevronLeft className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
        <div className="p-5 pt-2 sm:p-8 sm:pt-4 lg:p-12 lg:pt-6">
          <div className="mx-auto max-w-[100rem]">
            {state.status === "idle" || state.status === "loading" || state.status === "error" ? (
              <div className="mt-10 lg:mt-20 max-w-5xl mx-auto">
                <AudioUploader />
              </div>
            ) : (
              children
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default function PlaygroundLayout({ children }: { children: React.ReactNode }) {
  return (
    <PlaygroundProvider>
      <PlaygroundShell>{children}</PlaygroundShell>
    </PlaygroundProvider>
  );
}
