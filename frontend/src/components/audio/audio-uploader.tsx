"use client";

import { useEffect, useRef, useState } from "react";
import {
  AudioLines,
  CheckCircle2,
  FileAudio,
  Loader2,
  Upload,
} from "lucide-react";

import { analyzeAudio, type AudioAnalyzeResponse, type AudioMetadata, type WaveformData } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { WaveformViewer } from "@/components/audio/waveform-viewer";
import { AudioPlayer } from "@/components/audio/audio-player";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type UploadState =
  | { status: "idle" }
  | { status: "ready"; file: File }
  | { status: "loading"; file: File }
  | {
      status: "success";
      file: File;
      metadata: AudioMetadata;
      waveform: WaveformData;
      /** Object URL for the HTML5 audio element. Revoked on unmount. */
      objectUrl: string;
    }
  | { status: "error"; file?: File; message: string };

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AudioUploader() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<UploadState>({ status: "idle" });
  const [isDragging, setIsDragging] = useState(false);
  /** Current playback time in seconds, shared between AudioPlayer and WaveformViewer. */
  const [currentTime, setCurrentTime] = useState(0);
  /** Seek target written when the waveform is clicked. */
  const [seekTarget, setSeekTarget] = useState<number | undefined>(undefined);

  // Revoke the object URL when the component unmounts or a new file is loaded.
  const prevObjectUrl = useRef<string | null>(null);
  useEffect(() => {
    return () => {
      if (prevObjectUrl.current) {
        URL.revokeObjectURL(prevObjectUrl.current);
      }
    };
  }, []);

  const selectedFile =
    state.status === "ready" ||
    state.status === "loading" ||
    state.status === "success" ||
    state.status === "error"
      ? state.file
      : undefined;

  function chooseFile(file: File | undefined) {
    if (!file) return;

    // Reset playback state when a new file is chosen.
    setCurrentTime(0);
    setSeekTarget(undefined);
    setState({ status: "ready", file });
  }

  async function submitUpload(file: File | undefined = selectedFile) {
    if (!file) {
      setState({ status: "error", message: "Choose a WAV file before uploading." });
      return;
    }

    // Reset playback state.
    setCurrentTime(0);
    setSeekTarget(undefined);
    setState({ status: "loading", file });

    try {
      const response: AudioAnalyzeResponse = await analyzeAudio(file);

      // Create a local object URL so the HTML5 audio element can play the
      // file without requiring a backend streaming endpoint.  The file is
      // already in memory as the browser File object.
      const objectUrl = URL.createObjectURL(file);

      // Revoke the previous URL to avoid memory leaks.
      if (prevObjectUrl.current) {
        URL.revokeObjectURL(prevObjectUrl.current);
      }
      prevObjectUrl.current = objectUrl;

      setState({
        status: "success",
        file,
        metadata: response.metadata,
        waveform: response.waveform,
        objectUrl,
      });
    } catch (error) {
      setState({
        status: "error",
        file,
        message:
          error instanceof Error
            ? error.message
            : "The audio could not be loaded.",
      });
    }
  }

  // Handler for waveform click → seek AudioPlayer.
  function handleWaveformSeek(timeSeconds: number) {
    setCurrentTime(timeSeconds);
    setSeekTarget(timeSeconds);
  }

  // Handler for AudioPlayer timeupdate → update waveform playhead.
  function handleTimeUpdate(time: number) {
    setCurrentTime(time);
  }

  function handleAudioEnded() {
    setCurrentTime(0);
  }

  return (
    <div className="flex flex-col gap-6">
      {/* ---- File intake card ---- */}
      <Card className="overflow-hidden">
        <div
          className={[
            "relative flex min-h-[320px] flex-col justify-between p-6 transition-colors sm:p-8",
            isDragging ? "bg-accent" : "bg-card",
          ].join(" ")}
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
          <div className="flex items-start justify-between gap-6">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Audio Intake
              </p>
              <h2 className="mt-3 max-w-md text-3xl font-semibold leading-tight tracking-normal sm:text-4xl">
                Load a WAV file and inspect its signal.
              </h2>
            </div>
            <div className="hidden h-12 w-12 items-center justify-center border border-border bg-background sm:flex">
              <AudioLines className="h-5 w-5" />
            </div>
          </div>

          <button
            type="button"
            className="mt-8 flex min-h-36 w-full cursor-pointer flex-col items-center justify-center border border-dashed border-border bg-background/70 px-5 py-8 text-center transition hover:border-primary hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-card"
            onClick={() => inputRef.current?.click()}
          >
            <Upload className="h-8 w-8" />
            <span className="mt-4 text-base font-medium">
              Drop audio here, or select a WAV file
            </span>
            <span className="mt-2 max-w-sm text-sm leading-6 text-muted-foreground">
              Rigel decodes the digital samples and renders the signal waveform.
            </span>
          </button>

          <input
            ref={inputRef}
            type="file"
            accept=".wav,audio/wav,audio/x-wav,audio/wave"
            className="sr-only"
            onChange={(event) => chooseFile(event.target.files?.[0])}
          />

          <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-h-10">
              {selectedFile ? (
                <div className="flex items-center gap-3 text-sm">
                  <FileAudio className="h-4 w-4 shrink-0" />
                  <span className="font-medium">{selectedFile.name}</span>
                  <span className="text-muted-foreground">
                    {formatBytes(selectedFile.size)}
                  </span>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No file selected</p>
              )}
            </div>

            <button
              type="button"
              disabled={state.status === "loading"}
              className="inline-flex h-11 items-center justify-center gap-2 border border-primary bg-primary px-5 text-sm font-medium text-primary-foreground transition hover:bg-foreground/90 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={() => void submitUpload()}
              id="audio-upload-submit"
            >
              {state.status === "loading" ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analysing&hellip;
                </>
              ) : (
                "Analyse Audio"
              )}
            </button>
          </div>

          {state.status === "error" && (
            <div className="mt-5 border border-primary bg-primary p-4 text-sm text-primary-foreground">
              {state.message}
            </div>
          )}
        </div>
      </Card>

      {/* ---- Waveform + player card (visible only after a successful analysis) ---- */}
      {state.status === "success" && (
        <AnalysisPanel
          metadata={state.metadata}
          waveform={state.waveform}
          objectUrl={state.objectUrl}
          currentTime={currentTime}
          seekTarget={seekTarget}
          onWaveformSeek={handleWaveformSeek}
          onTimeUpdate={handleTimeUpdate}
          onAudioEnded={handleAudioEnded}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// AnalysisPanel — waveform visualizer + audio player + metadata
// ---------------------------------------------------------------------------

interface AnalysisPanelProps {
  metadata: AudioMetadata;
  waveform: WaveformData;
  objectUrl: string;
  currentTime: number;
  seekTarget: number | undefined;
  onWaveformSeek: (t: number) => void;
  onTimeUpdate: (t: number) => void;
  onAudioEnded: () => void;
}

function AnalysisPanel({
  metadata,
  waveform,
  objectUrl,
  currentTime,
  seekTarget,
  onWaveformSeek,
  onTimeUpdate,
  onAudioEnded,
}: AnalysisPanelProps) {
  return (
    <Card className="overflow-hidden">
      <div className="p-6 sm:p-8">
        {/* Header */}
        <div className="mb-6 flex items-center gap-3">
          <CheckCircle2 className="h-5 w-5 shrink-0" />
          <div>
            <h3 className="text-base font-semibold">Signal loaded</h3>
            <p className="text-sm text-muted-foreground">{metadata.filename}</p>
          </div>
        </div>

        {/* Metadata badges */}
        <div className="mb-6 flex flex-wrap gap-2">
          <MetaBadge label="Format" value={metadata.format.toUpperCase()} />
          <MetaBadge
            label="Sample Rate"
            value={`${metadata.sample_rate_hz.toLocaleString()} Hz`}
          />
          <MetaBadge
            label="Channels"
            value={metadata.channels === 1 ? "Mono" : "Stereo"}
          />
          <MetaBadge
            label="Duration"
            value={`${metadata.duration_seconds.toFixed(3)} s`}
          />
          {metadata.bit_depth != null && (
            <MetaBadge label="Bit Depth" value={`${metadata.bit_depth}-bit`} />
          )}
          <MetaBadge label="Type" value={metadata.sample_summary.dtype} />
        </div>

        {/* Waveform */}
        <div className="mb-4 w-full border border-border bg-background p-3">
          <WaveformViewer
            waveform={waveform}
            currentTime={currentTime}
            onSeek={onWaveformSeek}
          />
        </div>

        {/* Audio player */}
        <AudioPlayer
          src={objectUrl}
          duration={metadata.duration_seconds}
          onTimeUpdate={onTimeUpdate}
          onEnded={onAudioEnded}
          seekTarget={seekTarget}
        />

        {/* Signal detail table */}
        <details className="mt-6">
          <summary className="cursor-pointer text-xs uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground">
            Signal details
          </summary>
          <dl className="mt-3 grid gap-px overflow-hidden border border-border bg-border sm:grid-cols-2">
            {[
              ["Samples / Channel", metadata.samples_per_channel.toLocaleString()],
              ["Total Samples", metadata.total_samples.toLocaleString()],
              ["Array Shape", `[${metadata.sample_summary.array_shape.join(", ")}]`],
              ["Min Value", metadata.sample_summary.min_value.toString()],
              ["Max Value", metadata.sample_summary.max_value.toString()],
              ["Mean Value", metadata.sample_summary.mean_value.toFixed(4)],
              ["Waveform Bins", waveform.n_bins.toLocaleString()],
            ].map(([label, value]) => (
              <div key={label} className="bg-card p-4">
                <dt className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                  {label}
                </dt>
                <dd className="mt-2 text-sm font-medium">{value}</dd>
              </div>
            ))}
          </dl>
        </details>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function MetaBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-1.5 border border-border bg-card px-2.5 py-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-xs font-medium">{value}</span>
    </div>
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
