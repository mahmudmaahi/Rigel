"use client";

import { useState, useRef, useEffect } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { decodeBlobToSamples, encodeSamplesToWavBlob } from "@/lib/api";

const MAX_RECORDING_SECONDS = 30;

export function SidebarAudioRecorder({
  onRecordingComplete,
  className = "",
}: {
  onRecordingComplete: (file: File) => void;
  className?: string;
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(MAX_RECORDING_SECONDS);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const cleanup = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
    mediaRecorderRef.current = null;
    streamRef.current = null;
  };

  useEffect(() => {
    return cleanup;
  }, []);

  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        setIsRecording(false);
        setIsProcessing(true);
        cleanup();
        
        try {
          const rawBlob = new Blob(chunksRef.current, { type: mr.mimeType });
          // Decode browser's native format (webm/mp4) into raw Float32 arrays
          const { samples, sampleRate } = await decodeBlobToSamples(rawBlob);
          // Re-encode as strict 16-bit PCM WAV for the backend
          const wavBlob = encodeSamplesToWavBlob(samples, sampleRate);
          const wavFile = new File([wavBlob], `Recording_${new Date().getTime()}.wav`, { type: "audio/wav" });
          
          onRecordingComplete(wavFile);
        } catch (e) {
          console.error("Error processing recording:", e);
        } finally {
          setIsProcessing(false);
        }
      };

      setSecondsLeft(MAX_RECORDING_SECONDS);
      mr.start();
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setSecondsLeft((prev) => {
          if (prev <= 1) {
            if (mediaRecorderRef.current?.state === "recording") {
              mediaRecorderRef.current.stop();
            }
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

    } catch (e) {
      console.error("Microphone access denied:", e);
      alert("Microphone access is required to record audio.");
    }
  };

  const handleStop = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  };

  if (isProcessing) {
    return (
      <div className={`flex items-center justify-center gap-2 rounded-xl border border-indigo-500/30 bg-indigo-500/10 px-3 py-2 text-xs font-medium text-indigo-300 ${className}`}>
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Processing...
      </div>
    );
  }

  if (isRecording) {
    return (
      <button
        onClick={handleStop}
        className={`flex w-full items-center justify-between rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 px-3 py-2 text-xs font-medium text-red-400 transition-colors shadow-lg shadow-red-500/10 ${className}`}
      >
        <span className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
          </span>
          Recording ({secondsLeft}s)
        </span>
        <Square className="h-3.5 w-3.5 fill-current" />
      </button>
    );
  }

  return (
    <button
      onClick={handleStart}
      className={`flex w-full items-center justify-center gap-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 px-3 py-2 text-xs font-medium text-slate-300 hover:text-white transition-colors ${className}`}
    >
      <Mic className="h-3.5 w-3.5" />
      Record Audio
    </button>
  );
}
