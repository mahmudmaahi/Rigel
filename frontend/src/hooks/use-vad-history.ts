import { useState, useCallback, useRef } from "react";

export type VadState = "SILENCE" | "ONSET_PENDING" | "SPEECH" | "HANGOVER";

export interface VadResult {
  timestamp_seconds: number;
  state: VadState;
  is_speech: boolean;
  activity_score: number;
  frames_in_state: number;
  speech_duration_frames: number;
  compute_ms: number;
  engine: string;
}

export interface SpeechSegment {
  start_time: number;
  end_time: number | null; // null if still ongoing
  is_active: boolean; // true if currently SPEECH/HANGOVER
}

export function useVadHistory(maxHistorySec: number = 0) {
  const [history, setHistory] = useState<VadResult[]>([]);
  const [segments, setSegments] = useState<SpeechSegment[]>([]);
  const [latestResult, setLatestResult] = useState<VadResult | null>(null);

  const addResult = useCallback((result: VadResult) => {
    setLatestResult(result);
    
    setHistory(prev => {
      const newHistory = [...prev, result];
      // If we want to cap it (e.g. for microphone) - if maxHistorySec > 0
      if (maxHistorySec > 0 && newHistory.length > 0) {
        const latestTime = newHistory[newHistory.length - 1].timestamp_seconds;
        const cutoff = latestTime - maxHistorySec;
        // Keep only frames within maxHistorySec
        return newHistory.filter(r => r.timestamp_seconds >= cutoff);
      }
      return newHistory;
    });

    setSegments(prev => {
      const newSegments = [...prev];
      const isSpeech = result.is_speech;
      
      if (newSegments.length === 0) {
        if (isSpeech) {
          newSegments.push({ start_time: result.timestamp_seconds, end_time: null, is_active: true });
        }
        return newSegments;
      }
      
      const lastSegment = newSegments[newSegments.length - 1];
      
      if (isSpeech && !lastSegment.is_active) {
        // Start new segment
        newSegments.push({ start_time: result.timestamp_seconds, end_time: null, is_active: true });
      } else if (!isSpeech && lastSegment.is_active) {
        // End current segment
        lastSegment.end_time = result.timestamp_seconds;
        lastSegment.is_active = false;
      }
      
      return newSegments;
    });
  }, [maxHistorySec]);

  const clearHistory = useCallback(() => {
    setHistory([]);
    setSegments([]);
    setLatestResult(null);
  }, []);

  return {
    history,
    segments,
    latestResult: latestResult,
    addResult,
    clearHistory
  };
}
