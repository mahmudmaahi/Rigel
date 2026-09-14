"use client";

import React, { createContext, useContext, useState, useRef, useEffect } from "react";
import {
  analyzeAudio,
  fetchSpectrum,
  fetchSpectrogram,
  type AudioMetadata,
  type WaveformData,
  type SpectrumData,
  type SpectrogramData,
} from "@/lib/api";

export type SharedAudioState =
  | { status: "idle" }
  | { status: "loading"; file: File }
  | {
      status: "success";
      file: File;
      processedAudio: Blob | null;
      metadata: AudioMetadata;
      waveform: WaveformData;
      spectrum: SpectrumData | null;
      spectrogram: SpectrogramData | null;
      objectUrl: string;
      spectrogramWindow: string;
    }
  | { status: "error"; file?: File; message: string };

interface PlaygroundContextValue {
  state: SharedAudioState;
  uploadFile: (file: File) => Promise<void>;
  updateSpectrogramWindow: (windowName: string) => Promise<void>;
  clearAudio: () => void;
  setProcessedAudio: (blob: Blob | null) => void;
}

const PlaygroundContext = createContext<PlaygroundContextValue | null>(null);

export function PlaygroundProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<SharedAudioState>({ status: "idle" });
  const prevObjectUrl = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (prevObjectUrl.current) {
        URL.revokeObjectURL(prevObjectUrl.current);
      }
    };
  }, []);

  const uploadFile = async (file: File) => {
    setState({ status: "loading", file });

    try {
      const [analyzeResponse, spectrumResponse, spectrogramResponse] = await Promise.allSettled([
        analyzeAudio(file),
        fetchSpectrum(file),
        fetchSpectrogram(file, "hann"),
      ]);

      if (analyzeResponse.status === "rejected") {
        throw analyzeResponse.reason;
      }

      const response = analyzeResponse.value;
      const spectrum = spectrumResponse.status === "fulfilled" ? spectrumResponse.value.spectrum : null;
      const spectrogram = spectrogramResponse.status === "fulfilled" ? spectrogramResponse.value.spectrogram : null;

      const objectUrl = URL.createObjectURL(file);

      if (prevObjectUrl.current) {
        URL.revokeObjectURL(prevObjectUrl.current);
      }
      prevObjectUrl.current = objectUrl;

      setState({
        status: "success",
        file,
        processedAudio: null,
        metadata: response.metadata,
        waveform: response.waveform,
        spectrum,
        spectrogram,
        objectUrl,
        spectrogramWindow: "hann",
      });
    } catch (error) {
      setState({
        status: "error",
        file,
        message: error instanceof Error ? error.message : "The audio could not be loaded.",
      });
    }
  };

  const updateSpectrogramWindow = async (windowName: string) => {
    if (state.status !== "success") return;
    
    try {
      const response = await fetchSpectrogram(state.file, windowName);
      setState((prev) => 
        prev.status === "success" 
          ? { ...prev, spectrogram: response.spectrogram, spectrogramWindow: windowName }
          : prev
      );
    } catch (error) {
      console.error("Failed to fetch new spectrogram", error);
    }
  };

  const clearAudio = () => {
    if (prevObjectUrl.current) {
      URL.revokeObjectURL(prevObjectUrl.current);
      prevObjectUrl.current = null;
    }
    setState({ status: "idle" });
  };

  const setProcessedAudio = (blob: Blob | null) => {
    setState((prev) =>
      prev.status === "success" ? { ...prev, processedAudio: blob } : prev
    );
  };

  return (
    <PlaygroundContext.Provider value={{ state, uploadFile, updateSpectrogramWindow, clearAudio, setProcessedAudio }}>
      {children}
    </PlaygroundContext.Provider>
  );
}

export function usePlayground() {
  const context = useContext(PlaygroundContext);
  if (!context) {
    throw new Error("usePlayground must be used within a PlaygroundProvider");
  }
  return context;
}
