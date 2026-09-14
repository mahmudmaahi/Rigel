"use client";

import { useRef, useState } from "react";
import { AlertCircle, AudioLines, FileAudio, Loader2, Upload } from "lucide-react";
import { usePlayground } from "@/contexts/playground-context";
import { DropzoneIdleWaveform } from "@/components/visual/dropzone-idle-waveform";

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function AudioUploader() {
  const inputRef = useRef<HTMLInputElement>(null);
  const { state, uploadFile } = usePlayground();
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  function chooseFile(file: File | undefined) {
    if (!file) return;
    setSelectedFile(file);
  }

  async function submitUpload(file: File | null = selectedFile) {
    if (!file) {
      return; // Or show error locally
    }
    await uploadFile(file);
  }

  // If successfully loaded, AudioUploader handles its disappearance in layout or 
  // could return null if it's meant to hide automatically. Layout handles this though.
  if (state.status === "success") {
    return null;
  }

  return (
    <div className="glass-strong relative overflow-hidden rounded-3xl p-8 sm:p-10 shadow-2xl shadow-black/40 animate-fade-in-up">
      <div className="flex flex-col justify-between gap-8">
        <div className="flex items-start justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-3 py-1">
              <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-pulse" />
              <span className="font-mono text-xs uppercase tracking-wider text-indigo-300">
                Rigel DSP Engine
              </span>
            </div>
            <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-4xl text-white">
              Process Your Audio
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              Supports uncompressed PCM WAV files up to 24-bit / 96kHz.
            </p>
          </div>
          <div className="hidden h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md shadow-lg sm:flex">
            <AudioLines className="h-6 w-6 text-indigo-400" />
          </div>
        </div>

        <button
          type="button"
          className={[
            "group/dropzone relative flex min-h-48 w-full cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-12 text-center transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 overflow-hidden",
            isDragging
              ? "border-indigo-500 bg-indigo-500/10 scale-[0.99] shadow-2xl shadow-indigo-500/20"
              : "border-white/10 bg-white/[0.02] hover:border-indigo-500/40 hover:bg-white/[0.04]",
          ].join(" ")}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            chooseFile(event.dataTransfer.files[0]);
          }}
        >
          {/* Animated idle waveform inside dropzone */}
          <DropzoneIdleWaveform />

          <div className="relative z-10 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-gradient-to-br from-indigo-500/20 to-purple-600/20 shadow-lg shadow-indigo-500/20 transition-transform group-hover/dropzone:scale-110">
            <Upload className="h-6 w-6 text-indigo-300" />
          </div>

          <span className="relative z-10 mt-5 text-lg font-semibold text-white tracking-wide">
            Drop your high-fidelity WAV file here
          </span>
          <span className="relative z-10 mt-2 font-mono text-xs text-slate-400">
            or click to browse your local device
          </span>
        </button>

        <input
          ref={inputRef}
          type="file"
          accept=".wav,audio/wav,audio/x-wav,audio/wave"
          className="sr-only"
          onChange={(event) => chooseFile(event.target.files?.[0])}
        />

        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-t border-white/5 pt-6">
          <div>
            {selectedFile ? (
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/5">
                  <FileAudio className="h-5 w-5 text-indigo-400 shrink-0" />
                </div>
                <div>
                  <span className="font-medium text-white">{selectedFile.name}</span>
                  <p className="font-mono text-xs text-slate-400">
                    {formatBytes(selectedFile.size)}
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <span className="h-2 w-2 rounded-full bg-slate-500" />
                <span className="font-mono text-xs uppercase tracking-wider">
                  No file selected
                </span>
              </div>
            )}
          </div>

          <button
            type="button"
            disabled={state.status === "loading" || !selectedFile}
            className="group animate relative flex cursor-pointer items-center justify-center text-white bg-slate-600/80 border border-slate-500 h-12 px-8 hover:border-slate-500 hover:bg-slate-300/50 disabled:cursor-not-allowed disabled:opacity-50 rounded-full overflow-hidden transition-colors"
            onClick={() => void submitUpload()}
          >
            <div
              className="before:absolute before:inset-[0] before:h-full before:w-full before:rounded-full before:p-[1px] before:will-change-[background-position] before:content-[''] before:![-webkit-mask-composite:xor] before:[background-image:var(--background-radial-gradient)] before:[background-size:300%_300%] before:![mask-composite:exclude] before:[mask:var(--mask-linear-gradient)] motion-safe:before:animate-[shine-pulse_var(--shine-pulse-duration)_infinite_linear]"
              style={{
                "--shine-pulse-duration": "14s",
                "--mask-linear-gradient": "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
                "--background-radial-gradient": "radial-gradient(transparent,transparent, #A07CFE,#FE8FB5,#FFBE7B,transparent,transparent)",
              } as React.CSSProperties}
            ></div>
            <div className="relative z-[1] flex items-center justify-center gap-2 text-sm font-semibold tracking-wide text-white">
              {state.status === "loading" ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>Load into Playground</span>
                  <span className="transition-transform group-hover:translate-x-1">→</span>
                </>
              )}
            </div>
          </button>
        </div>

        {state.status === "error" && (
          <div className="flex items-start gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-200">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
            <div>
              <p className="font-medium">Analysis Error</p>
              <p className="mt-1 text-xs text-red-300">{state.message}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
