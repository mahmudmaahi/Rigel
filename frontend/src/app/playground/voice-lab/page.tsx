"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { usePlayground } from "@/contexts/playground-context";
import {
  processVoiceLab,
  decodeBlobToSamples,
  encodeSamplesToWavBlob,
  measureAudio,
  analyzeAudio,
  fetchSpectrum,
  fetchSpectrogram,
  type VoiceEffect,
  type VoiceProcessResponse,
  type MeasurementResponse,
  type LiveMeasurementFrame,
  type WaveformData,
  type SpectrumData,
  type SpectrogramData,
} from "@/lib/api";
import {
  Mic, Loader2, CheckCircle,
  Zap, Activity, Music2, MicOff,
  FlaskConical, Play, Square, Plus, Trash2, Power, Settings2, FolderOpen, FileAudio, Info
} from "lucide-react";
import { SignalComparison } from "@/components/audio/signal-comparison";
import { AudioPlayer, AudioPlayerRef } from "@/components/audio/audio-player";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function SectionHeader({ title, icon: Icon, color = "text-indigo-400" }: {
  title: string; icon: React.ComponentType<{ className?: string }>; color?: string;
}) {
  return (
    <div className="flex items-center gap-2 mb-4 border-b border-white/5 pb-2">
      <Icon className={`h-4 w-4 ${color}`} />
      <h3 className="text-xs font-bold tracking-widest text-slate-300 uppercase">{title}</h3>
    </div>
  );
}

function Knob({ label, value, min, max, step = 0.01, unit = "", onChange, precision = 2 }: {
  label: string; value: number; min: number; max: number; step?: number;
  unit?: string; onChange: (v: number) => void; precision?: number;
}) {
  return (
    <div className="flex flex-col gap-1.5 flex-1 min-w-[120px]">
      <label className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">{label}</label>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="flex-1 h-1.5 appearance-none rounded-full bg-white/10 accent-indigo-500 cursor-pointer"
        />
        <span className="text-xs font-mono text-indigo-300 w-16 text-right">
          {value.toFixed(precision)}{unit}
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Add Effect Dropdown Component
// ---------------------------------------------------------------------------
const EFFECT_OPTIONS = [
  { value: "gain", label: "Gain" },
  { value: "pitch_shift", label: "Pitch Shift" },
  { value: "speed", label: "Speed" },
  { value: "time_stretch", label: "Time Stretch" },
  { value: "echo_delay", label: "Echo / Delay" },
  { value: "reverb", label: "Reverb" },
  { value: "chorus", label: "Chorus" },
  { value: "tremolo", label: "Tremolo" },
  { value: "soft_distortion", label: "Distortion" },
];

function AddEffectDropdown({ onAdd }: { onAdd: (v: string) => void }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider rounded-full border px-3 py-1.5 focus:outline-none focus:ring-1 transition-colors text-slate-300 border-white/10 bg-white/5 focus:ring-slate-400 focus:bg-white/10 hover:bg-white/10"
      >
        <Plus className="w-3 h-3" />
        Add Effect
        <svg className="h-3 w-3 fill-current text-slate-400" viewBox="0 0 20 20">
          <path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" fillRule="evenodd" />
        </svg>
      </button>
      {isOpen && (
        <div className="absolute right-0 mt-2 w-40 origin-top-right rounded-xl border border-white/10 bg-[#0a0a14]/95 backdrop-blur-xl shadow-2xl shadow-black ring-1 ring-black ring-opacity-5 focus:outline-none z-50 overflow-hidden">
          <div className="py-1">
            {EFFECT_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className="block w-full text-left px-4 py-2 text-xs font-mono tracking-wide transition-colors text-slate-300 hover:bg-white/10 hover:text-white"
                onClick={() => {
                  setIsOpen(false);
                  onAdd(option.value);
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SVG Timeline Visualization
// ---------------------------------------------------------------------------
function MeasurementVisualization({ 
  frames, 
  currentTime, 
  duration, 
  mode 
}: { 
  frames: { time: number, frame: LiveMeasurementFrame }[], 
  currentTime: number, 
  duration: number, 
  mode: "microphone" | "shared" 
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const height = 140;

  useEffect(() => {
    const observer = new ResizeObserver(entries => {
      if (entries[0]) setWidth(entries[0].contentRect.width);
    });
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // For shared audio, X axis is 0 to duration (max 60s for visual sanity)
  // For microphone, X axis is a rolling 10s window (currentTime - 10 to currentTime)
  const windowDuration = mode === "shared" ? Math.min(Math.max(duration, 1), 60) : 10;
  const timeMin = mode === "shared" ? 0 : Math.max(0, currentTime - windowDuration);
  const timeMax = timeMin + windowDuration;

  const getX = (t: number) => {
    if (t < timeMin || t > timeMax) return -10;
    return ((t - timeMin) / windowDuration) * width;
  };

  const getY = (rms: number) => {
    // RMS is typically -60 to 0
    const clamped = Math.max(-60, Math.min(0, rms));
    return height - ((clamped + 60) / 60) * height;
  };

  // Build SVG path for RMS
  let rmsPath = "";
  const visibleFrames = frames.filter(f => f.time >= timeMin && f.time <= timeMax);
  
  if (visibleFrames.length > 0 && width > 0) {
    rmsPath = `M ${getX(visibleFrames[0].time)},${height} `;
    
    for (let i = 0; i < visibleFrames.length; i++) {
       const x = getX(visibleFrames[i].time);
       const y = getY(visibleFrames[i].time > currentTime ? -60 : visibleFrames[i].frame.rms_dbfs);
       if (i === 0) {
         rmsPath = `M ${x},${y}`;
       } else {
         rmsPath += ` L ${x},${y}`;
       }
    }
  }

  // Current playback/record cursor
  const cursorX = getX(currentTime);

  return (
    <div ref={containerRef} className="w-full h-[140px] relative bg-black/40 rounded-xl border border-white/10 overflow-hidden">
      {width > 0 && (
        <svg width={width} height={height} className="absolute inset-0">
          {/* Grid lines */}
          {[0, -15, -30, -45].map(db => (
            <line key={db} x1={0} y1={getY(db)} x2={width} y2={getY(db)} stroke="rgba(255,255,255,0.05)" strokeWidth={1} strokeDasharray="4 4" />
          ))}

          {/* RMS Path */}
          {rmsPath && (
            <path d={rmsPath} fill="none" stroke="rgba(52, 211, 153, 0.8)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          )}
          
          {/* Peak Points (Red dots for clipping) */}
          {visibleFrames.map((f, i) => {
            if (f.frame.peak_dbfs > -1.0 && f.time <= currentTime) {
              return <circle key={i} cx={getX(f.time)} cy={getY(f.frame.peak_dbfs)} r={2} fill="rgba(244, 63, 94, 0.9)" />
            }
            return null;
          })}

          {/* Fill under RMS */}
          {rmsPath && (
            <path d={`${rmsPath} L ${cursorX > 0 ? Math.min(cursorX, width) : 0},${height} L ${getX(visibleFrames[0].time)},${height} Z`} fill="url(#rmsGrad)" opacity={0.2} />
          )}

          {/* Voiced Regions (Cyan highlights at bottom) */}
          {visibleFrames.map((f, i) => {
            if (f.frame.voiced && f.time <= currentTime) {
              return <rect key={`v${i}`} x={getX(f.time) - 1} y={height - 4} width={3} height={4} fill="rgba(34, 211, 238, 0.8)" />
            }
            return null;
          })}

          {/* Cursor */}
          {cursorX >= 0 && cursorX <= width && (
            <line x1={cursorX} y1={0} x2={cursorX} y2={height} stroke="rgba(255,255,255,0.5)" strokeWidth={1} />
          )}

          <defs>
            <linearGradient id="rmsGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(52, 211, 153, 1)" />
              <stop offset="100%" stopColor="rgba(52, 211, 153, 0)" />
            </linearGradient>
          </defs>
        </svg>
      )}

      {/* Axis Labels */}
      <div className="absolute top-1 left-2 text-[8px] font-mono text-slate-500">0 dBFS</div>
      <div className="absolute bottom-1 right-2 text-[8px] font-mono text-slate-500">{windowDuration}s</div>
    </div>
  );
}

export default function VoiceLabPage() {
  const { state, setProcessedAudio, uploadFile } = usePlayground();

  // ---------------------------------------------------------------------------
  // STATE
  // ---------------------------------------------------------------------------
  const [sourceMode, setSourceMode] = useState<"microphone" | "shared">("shared");
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
  
  // Streaming/Live Measurements
  const [liveFrame, setLiveFrame] = useState<LiveMeasurementFrame | null>(null);
  const [liveFramesTimeline, setLiveFramesTimeline] = useState<{time: number, frame: LiveMeasurementFrame}[]>([]);
  const [sessionStats, setSessionStats] = useState<MeasurementResponse | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const micStartTimeRef = useRef<number>(0);
  
  // Mic state
  const [isRecording, setIsRecording] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  // Shared state
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [isPlayingShared, setIsPlayingShared] = useState(false);
  const audioPlayerRef = useRef<AudioPlayerRef>(null);
  const sampleCursorRef = useRef<number>(0);
  const requestRef = useRef<number | undefined>(undefined);
  const lastDecodedUrlRef = useRef<string | null>(null);

  // Audio/WS Core
  const [sampleRate, setSampleRate] = useState<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Offline stats & Pipeline
  const [offlineMeasure, setOfflineMeasure] = useState<MeasurementResponse | null>(null);
  const [effectChain, setEffectChain] = useState<VoiceEffect[]>([]);
  const [processing, setProcessing] = useState(false);
  const [processError, setProcessError] = useState<string | null>(null);
  const [result, setResult] = useState<VoiceProcessResponse | null>(null);
  const [procAudioUrl, setProcAudioUrl] = useState<string | null>(null);
  const [procWaveform, setProcWaveform] = useState<WaveformData | null>(null);
  const [procSpectrum, setProcSpectrum] = useState<SpectrumData | null>(null);
  const [procSpectrogram, setProcSpectrogram] = useState<SpectrogramData | null>(null);

  // ---------------------------------------------------------------------------
  // LIFECYCLE & CLEANUP
  // ---------------------------------------------------------------------------
  const cleanupAll = useCallback(() => {
    setIsRecording(false);
    setIsPlayingShared(false);
    setWsStatus("disconnected");
    setLiveFrame(null);
    sampleCursorRef.current = 0;
    setCurrentTime(0);
    
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = undefined;
    }
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    return cleanupAll;
  }, [cleanupAll]);

  // Decode audio buffer when shared file changes
  useEffect(() => {
    let active = true;
    if (state.status === "success" && sourceMode === "shared") {
      if (lastDecodedUrlRef.current === state.objectUrl) return;
      lastDecodedUrlRef.current = state.objectUrl;
      
      const ctx = new window.AudioContext();
      state.file.arrayBuffer().then(buffer => {
        ctx.decodeAudioData(buffer).then(decoded => {
          if (!active) return;
          setAudioBuffer(decoded);
          cleanupAll();
          setLiveFramesTimeline([]);
          setSessionStats(null);
          ctx.close();
        }).catch(err => {
          console.error("Failed to decode audio", err);
          ctx.close();
        });
      });
    }
    return () => { active = false; };
  }, [state, sourceMode, cleanupAll]);

  useEffect(() => {
    let active = true;
    if (state.status === "success") {
      fetch(state.objectUrl).then(r => r.blob()).then(async blob => {
        const { samples, sampleRate } = await decodeBlobToSamples(blob);
        if (!active) return;
        try {
          const m = await measureAudio(samples, sampleRate);
          if (active) setOfflineMeasure(m);
        } catch (e) {
          if (active) setOfflineMeasure(null);
        }
      });
    } else {
      setTimeout(() => { if (active) setOfflineMeasure(null); }, 0);
    }
    return () => { active = false; };
  }, [state]);

  // ---------------------------------------------------------------------------
  // WEBSOCKET LOGIC
  // ---------------------------------------------------------------------------
  const connectWebSocket = (): Promise<void> => {
    return new Promise((resolve, reject) => {
      setWsStatus("connecting");
      const wsUrl = process.env.NEXT_PUBLIC_API_BASE_URL
        ? process.env.NEXT_PUBLIC_API_BASE_URL.replace(/^http/, "ws") + "/api/audio/measure/stream"
        : "ws://127.0.0.1:8000/api/audio/measure/stream";
        
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        setWsStatus("connected");
        resolve();
      };
      
      ws.onerror = (e) => {
        console.error("WS Error:", e);
        reject(e);
      };
      
      ws.onclose = () => {
         setWsStatus("disconnected");
         wsRef.current = null;
      }
      
      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
            try {
              const frame = JSON.parse(event.data) as LiveMeasurementFrame;
              setLiveFrame(frame);
              
              setLiveFramesTimeline(prev => {
                let t = 0;
                if (sourceMode === "shared" && audioPlayerRef.current) {
                  t = audioPlayerRef.current.getCurrentTime();
                } else if (sourceMode === "microphone") {
                  t = (Date.now() - micStartTimeRef.current) / 1000;
                }
                
                return [...prev, { time: t, frame }];
              });
            } catch (e) {}
        }
      };
    });
  };

  const handleModeSwitch = (mode: "microphone" | "shared") => {
    if (mode === sourceMode) return;
    cleanupAll();
    setSourceMode(mode);
    setLiveFramesTimeline([]);
    setSessionStats(null);
  };

  // ---------------------------------------------------------------------------
  // MICROPHONE LOGIC
  // ---------------------------------------------------------------------------
  const startRecording = async () => {
    try {
      cleanupAll();
      setMicError(null);
      setLiveFramesTimeline([]);
      setSessionStats(null);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
      streamRef.current = stream;
      micStartTimeRef.current = Date.now();

      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;
      setSampleRate(audioCtx.sampleRate);

      await audioCtx.audioWorklet.addModule("/pcm-processor.js");

      const source = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, "pcm-processor");
      workletNodeRef.current = workletNode;
      
      source.connect(workletNode);
      workletNode.connect(audioCtx.destination);

      await connectWebSocket();
      setIsRecording(true);
      
      const updateMicTime = () => {
         if (streamRef.current && streamRef.current.active) {
            setCurrentTime((Date.now() - micStartTimeRef.current) / 1000);
            requestRef.current = requestAnimationFrame(updateMicTime);
         }
      };
      requestRef.current = requestAnimationFrame(updateMicTime);

      workletNode.port.onmessage = (event) => {
        const buffer = event.data; // Float32Array
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(buffer);
        }
      };

    } catch (err: any) {
      setMicError(err.message || "Microphone access denied.");
      cleanupAll();
    }
  };

  const stopRecording = () => {
    cleanupAll();
    
    // In Microphone mode, compute stats when recording stops
    if (liveFramesTimeline.length > 0) {
      const pitches = liveFramesTimeline.filter(f => f.frame.voiced).map(f => f.frame.pitch_hz);
      const rmsVals = liveFramesTimeline.map(f => f.frame.rms_dbfs);
      const peakVals = liveFramesTimeline.map(f => f.frame.peak_dbfs);
      
      const sessionAvgRms = rmsVals.reduce((a,b)=>a+b, 0) / rmsVals.length;
      const sessionPeak = Math.max(...peakVals);
      
      setSessionStats({
         pitch: {
           average_pitch_hz: pitches.length > 0 ? pitches.reduce((a,b)=>a+b, 0) / pitches.length : 0,
           min_pitch_hz: pitches.length > 0 ? Math.min(...pitches) : 0,
           max_pitch_hz: pitches.length > 0 ? Math.max(...pitches) : 0,
           voiced_percentage: pitches.length / liveFramesTimeline.length,
           confidence: 1.0,
         },
         loudness: {
           average_rms_dbfs: sessionAvgRms,
           peak_dbfs: sessionPeak,
         },
         rhythm: {
           bpm: 0, // BPM not calculated live
           confidence: 0,
           reliable: false
         }
      });
    }
  };

  // ---------------------------------------------------------------------------
  // SHARED AUDIO LOGIC
  // ---------------------------------------------------------------------------
  const sendNextChunk = () => {
    if (!audioBuffer || !audioPlayerRef.current) return;
    
    const t = audioPlayerRef.current.getCurrentTime();
    setCurrentTime(t);
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const targetCursor = Math.floor(t * audioBuffer.sampleRate);
      const currentCursor = sampleCursorRef.current;
      
      if (targetCursor > currentCursor && targetCursor <= audioBuffer.length) {
        const channelData = audioBuffer.getChannelData(0);
        const chunk = channelData.subarray(currentCursor, targetCursor);
        wsRef.current.send(chunk);
        sampleCursorRef.current = targetCursor;
      }
    }
    requestRef.current = requestAnimationFrame(sendNextChunk);
  };

  const handleSharedPlay = () => {
    setIsPlayingShared(true);
    setSessionStats(null); // Clear offline stats while playing
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
       connectWebSocket().then(() => {
           setSampleRate(audioBuffer?.sampleRate || 16000);
           requestRef.current = requestAnimationFrame(sendNextChunk);
       }).catch(() => setIsPlayingShared(false));
    } else {
       requestRef.current = requestAnimationFrame(sendNextChunk);
    }
  };

  const handleSharedPause = () => {
    setIsPlayingShared(false);
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = undefined;
    }
  };

  const handleSharedSeek = (time: number) => {
    if (audioBuffer) {
        sampleCursorRef.current = Math.floor(time * audioBuffer.sampleRate);
        setCurrentTime(time);
        
        // Remove timeline data after the seek point so it redraws cleanly
        setLiveFramesTimeline(prev => prev.filter(f => f.time <= time));
        setLiveFrame(null);
    }
  };

  const handleSharedEnded = () => {
    handleSharedPause();
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsStatus("disconnected");
    // Once finished, show the exact offline measurement stats
    if (offlineMeasure) {
      setSessionStats(offlineMeasure);
    }
  };

  // ---------------------------------------------------------------------------
  // PIPELINE PROCESS
  // ---------------------------------------------------------------------------
  const handleProcess = async () => {
    if (state.status !== "success") return;
    setProcessing(true);
    setProcessError(null);
    try {
      const blob = await fetch(state.objectUrl).then(r => r.blob());
      const { samples, sampleRate } = await decodeBlobToSamples(blob);
      
      const resp = await processVoiceLab({ samples, sample_rate_hz: sampleRate, effect_chain: effectChain });
      
      setResult(resp);
      const wavBlob = encodeSamplesToWavBlob(resp.samples, resp.sample_rate_hz);
      if (procAudioUrl) URL.revokeObjectURL(procAudioUrl);
      const newUrl = URL.createObjectURL(wavBlob);
      setProcAudioUrl(newUrl);
      
      const file = new File([wavBlob], "processed.wav", { type: "audio/wav" });
      const [analyzeRes, specRes, spectroRes] = await Promise.all([
        analyzeAudio(file), fetchSpectrum(file), fetchSpectrogram(file)
      ]);
      setProcWaveform(analyzeRes.waveform);
      setProcSpectrum(specRes.spectrum);
      setProcSpectrogram(spectroRes.spectrogram);
      
    } catch (e: any) {
      setProcessError(e.message || "Processing failed.");
    } finally {
      setProcessing(false);
    }
  };

  const addEffect = (type: string) => {
    let ef: VoiceEffect;
    switch(type) {
      case "gain": ef = { op: "gain", mode: "db", gain_value: 3, enabled: true }; break;
      case "speed": ef = { op: "speed", speed: 1.5, enabled: true }; break;
      case "time_stretch": ef = { op: "time_stretch", stretch: 1.5, enabled: true }; break;
      case "pitch_shift": ef = { op: "pitch_shift", semitones: 4, enabled: true }; break;
      case "echo_delay": ef = { op: "echo_delay", delay_ms: 300, feedback: 0.4, mix: 0.5, enabled: true }; break;
      case "reverb": ef = { op: "reverb", room_size: 0.5, decay: 0.5, wet: 0.3, enabled: true }; break;
      case "chorus": ef = { op: "chorus", rate_hz: 1.5, depth_ms: 3, base_delay_ms: 15, mix: 0.5, enabled: true }; break;
      case "tremolo": ef = { op: "tremolo", rate_hz: 5, depth: 0.5, enabled: true }; break;
      case "soft_distortion": ef = { op: "soft_distortion", drive: 3, enabled: true }; break;
      default: return;
    }
    setEffectChain([...effectChain, ef]);
  };
  const updateEffect = (idx: number, ef: VoiceEffect) => { const c = [...effectChain]; c[idx] = ef; setEffectChain(c); };
  const removeEffect = (idx: number) => { const c = [...effectChain]; c.splice(idx, 1); setEffectChain(c); };
  const moveEffect = (idx: number, dir: 1|-1) => { if(idx+dir < 0 || idx+dir >= effectChain.length) return; const c=[...effectChain]; [c[idx], c[idx+dir]] = [c[idx+dir], c[idx]]; setEffectChain(c); };


  return (
    <div className="max-w-7xl mx-auto space-y-12 pb-24 px-4 text-white font-sans">
      
      {/* HEADER */}
      <div className="flex flex-col items-center justify-center text-center mt-10 mb-8">
        <h1 className="flex items-center justify-center gap-3 text-4xl font-bold tracking-tight text-white">
          <FlaskConical className="h-10 w-10 text-indigo-400" />
          Voice Laboratory
        </h1>
        <p className="mt-4 text-base text-slate-400 max-w-2xl mx-auto">
          Shape the temporal, spectral, and tonal character of a recording.
        </p>
      </div>

      {/* TOP ROW: EXACTLY LIKE VAD PAGE GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Control Panel (col-span-1) */}
        <div className="col-span-1 rounded-2xl border border-white/5 bg-white/5 p-6 backdrop-blur-md flex flex-col gap-6 min-h-[420px]">
          
          {/* Source Toggle */}
          <div className="flex flex-col sm:flex-row bg-black/40 p-1 rounded-xl border border-white/5 gap-1">
            <button
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all ${
                sourceMode === "microphone" 
                  ? "bg-white/10 text-white shadow-lg" 
                  : "text-slate-400 hover:text-slate-300 hover:bg-white/5"
              }`}
              onClick={() => handleModeSwitch("microphone")}
            >
              <Mic className="h-4 w-4" /> Microphone
            </button>
            <button
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all ${
                sourceMode === "shared" 
                  ? "bg-white/10 text-white shadow-lg" 
                  : "text-slate-400 hover:text-slate-300 hover:bg-white/5"
              }`}
              onClick={() => handleModeSwitch("shared")}
            >
              <FileAudio className="h-4 w-4" /> Shared Audio
            </button>
          </div>
          
          <div className="flex-1">
            <h2 className="text-sm font-bold uppercase tracking-widest text-slate-500 mb-4">Capture Control</h2>
            
            {sourceMode === "microphone" && (
              <>
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={wsStatus === "connecting"}
                    className="w-full flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-3 text-sm font-medium text-white transition-all disabled:opacity-50"
                  >
                    <Mic className="h-5 w-5" />
                    {wsStatus === "connecting" ? "Connecting..." : "Start Microphone"}
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="w-full flex items-center justify-center gap-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 px-4 py-3 text-sm font-medium text-red-400 transition-all"
                  >
                    <MicOff className="h-5 w-5" />
                    Stop Recording
                  </button>
                )}
                {micError && <div className="mt-2 text-xs text-red-400">{micError}</div>}
              </>
            )}

            {sourceMode === "shared" && (
              <div className="space-y-4">
                {state.status === "success" ? (
                  <>
                    <AudioPlayer
                      ref={audioPlayerRef}
                      src={state.objectUrl}
                      duration={state.metadata?.duration_seconds || 0}
                      onPlay={handleSharedPlay}
                      onPause={handleSharedPause}
                      onSeek={handleSharedSeek}
                      onEnded={handleSharedEnded}
                    />
                    <div className="flex justify-end">
                      <label className="cursor-pointer flex items-center justify-center gap-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-white/10 px-4 py-2 text-sm font-medium text-white transition-all">
                        <FileAudio className="h-4 w-4" />
                        Change File
                        <input
                          type="file"
                          className="hidden"
                          accept="audio/wav"
                          onChange={async (e) => {
                            if (e.target.files?.[0]) {
                              handleSharedSeek(0);
                              await uploadFile(e.target.files[0]);
                            }
                          }}
                        />
                      </label>
                    </div>
                  </>
                ) : (
                  <div className="p-4 border border-white/10 rounded-xl bg-black/20 text-center flex flex-col items-center gap-4">
                    <p className="text-sm text-slate-400">No shared audio loaded.</p>
                    <p className="text-xs text-slate-500">Upload a WAV file from the sidebar to begin.</p>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="space-y-4 pt-4 border-t border-white/5 mt-auto">
            <h2 className="text-sm font-bold uppercase tracking-widest text-slate-500">Stream Status</h2>
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-400">WebSocket</span>
              <span className={`px-2 py-0.5 rounded font-mono text-xs ${
                wsStatus === "connected" ? "bg-emerald-500/20 text-emerald-300" :
                wsStatus === "connecting" ? "bg-amber-500/20 text-amber-300" :
                "bg-slate-500/20 text-slate-300"
              }`}>
                {wsStatus.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-400">Sample Rate</span>
              <span className="font-mono text-indigo-300">
                {sampleRate ? `${sampleRate} Hz` : "---"}
              </span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-400">Format</span>
              <span className="font-mono text-indigo-300">Float32 PCM</span>
            </div>
          </div>
        </div>

        {/* Live DSP Results / Measurements (col-span-2) */}
        <div className="col-span-1 lg:col-span-2 flex flex-col gap-6">
          <div className="flex-1 rounded-2xl border border-white/5 bg-white/5 p-6 backdrop-blur-md flex flex-col relative overflow-hidden min-h-[420px]">
             
             {/* Header */}
             <div className="flex items-center justify-between z-10 mb-4">
               <h2 className="text-sm font-bold uppercase tracking-widest text-slate-500">Live Voice Measurements</h2>
               <div className="flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-black/40">
                 <div className={`h-2 w-2 rounded-full ${wsStatus === "connected" ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" : "bg-slate-600"}`} />
                 <span className="text-[10px] font-bold tracking-widest uppercase text-slate-400">
                   {wsStatus === "connected" ? "Live Stream Active" : "Waiting for Stream"}
                 </span>
               </div>
             </div>

             <div className="flex-1 flex flex-col justify-end z-10 relative">
                
                {/* Empty State / Initial Message */}
                {liveFramesTimeline.length === 0 && sessionStats === null && (
                   <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-600 pointer-events-none">
                     <h3 className="text-sm font-bold tracking-widest uppercase">Waiting for Audio Stream...</h3>
                     <p className="mt-2 text-xs opacity-60">Live measurements will appear here</p>
                   </div>
                )}
                
                {/* Session Stats (Offline Summary) */}
                {sessionStats !== null && (
                   <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-[#0c0c1a]">
                      <h3 className="text-sm font-bold tracking-widest text-indigo-400 uppercase mb-6">Session Stats Summary</h3>
                      <div className="grid grid-cols-3 gap-4 lg:gap-6 w-full px-4 lg:px-8">
                         <div className="bg-black/60 p-4 lg:p-6 rounded-xl border border-white/10 text-center shadow-lg shadow-black/40">
                            <div className="text-[10px] uppercase tracking-widest text-slate-400 mb-2">Peak dBFS</div>
                            <div className="text-2xl font-mono text-emerald-400">
                               {sessionStats.loudness?.peak_dbfs?.toFixed(1) ?? (sessionStats as any).peak_dbfs?.toFixed(1) ?? "-"}
                            </div>
                         </div>
                         <div className="bg-black/60 p-4 lg:p-6 rounded-xl border border-white/10 text-center shadow-lg shadow-black/40">
                            <div className="text-[10px] uppercase tracking-widest text-slate-400 mb-2">RMS dBFS</div>
                            <div className="text-2xl font-mono text-cyan-400">
                               {sessionStats.loudness?.average_rms_dbfs?.toFixed(1) ?? (sessionStats as any).rms_dbfs?.toFixed(1) ?? "-"}
                            </div>
                         </div>
                         <div className="bg-black/60 p-4 lg:p-6 rounded-xl border border-white/10 text-center shadow-lg shadow-black/40">
                            <div className="text-[10px] uppercase tracking-widest text-slate-400 mb-2">BPM (Rhythm)</div>
                            <div className="text-2xl font-mono text-purple-400">
                               {sessionStats.rhythm?.reliable || (sessionStats as any).rhythm_reliable
                                 ? (sessionStats.rhythm?.bpm ?? (sessionStats as any).bpm)?.toFixed(0) 
                                 : "---"}
                            </div>
                         </div>
                         
                         {/* Pitch Stats row */}
                         <div className="col-span-3 grid grid-cols-3 gap-4 lg:gap-6 mt-2">
                             <div className="bg-black/40 p-4 rounded-xl border border-white/5 text-center">
                                <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Avg Pitch</div>
                                <div className="text-lg font-mono text-slate-300">
                                   {(sessionStats.pitch?.average_pitch_hz ?? (sessionStats as any).average_pitch ?? 0) > 0 
                                      ? `${(sessionStats.pitch?.average_pitch_hz ?? (sessionStats as any).average_pitch).toFixed(1)} Hz` 
                                      : "---"}
                                </div>
                             </div>
                             <div className="bg-black/40 p-4 rounded-xl border border-white/5 text-center">
                                <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Pitch Range</div>
                                <div className="text-lg font-mono text-slate-300">
                                   {(sessionStats.pitch?.min_pitch_hz ?? (sessionStats as any).min_pitch ?? 0) > 0 
                                      ? `${(sessionStats.pitch?.min_pitch_hz ?? (sessionStats as any).min_pitch).toFixed(0)} - ${(sessionStats.pitch?.max_pitch_hz ?? (sessionStats as any).max_pitch).toFixed(0)} Hz` 
                                      : "---"}
                                </div>
                             </div>
                             <div className="bg-black/40 p-4 rounded-xl border border-white/5 text-center">
                                <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Voiced %</div>
                                <div className="text-lg font-mono text-slate-300">
                                   {((sessionStats.pitch?.voiced_percentage ?? (sessionStats as any).voiced_percentage ?? 0) * 100).toFixed(1)}%
                                </div>
                             </div>
                         </div>
                      </div>
                   </div>
                )}

                {/* Live Data (only visible if we have frames and NO session stats overlay) */}
                <div className={`flex-1 flex flex-col justify-end transition-opacity duration-300 ${liveFramesTimeline.length > 0 && sessionStats === null ? "opacity-100" : "opacity-0 pointer-events-none"}`}>
                  <div className="grid grid-cols-2 gap-4 mb-4 mt-auto">
                    <div className="flex flex-col justify-between bg-black/40 p-4 rounded-xl border border-white/10">
                      <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">Live RMS Level</span>
                      <span className="text-2xl font-mono text-emerald-400">{liveFrame?.rms_dbfs.toFixed(1) ?? "-"} dB</span>
                    </div>
                    
                    <div className="flex flex-col justify-between bg-black/40 p-4 rounded-xl border border-white/10">
                      <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">Live Pitch</span>
                      {liveFrame?.voiced ? (
                        <span className="text-2xl font-mono text-cyan-400">{liveFrame.pitch_hz.toFixed(0)} Hz</span>
                      ) : (
                        <span className="text-sm font-mono text-slate-500 mt-2 uppercase">Unvoiced / Idle</span>
                      )}
                    </div>
                  </div>

                  {/* Graph Title and Tooltip explanation */}
                  <div className="flex items-center justify-between mt-auto mb-2">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-emerald-400" />
                        <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">RMS Level (Height)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-cyan-400" />
                        <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">Voiced Limit</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-rose-500">
                        <Info className="h-3 w-3" />
                        <span className="text-[9px] font-mono opacity-80">Red dots indicate transient peak clipping {'>'} -1dBFS</span>
                    </div>
                  </div>
                  
                  {/* SVG Visualizer Graph Box */}
                  <MeasurementVisualization 
                     frames={liveFramesTimeline} 
                     currentTime={currentTime} 
                     duration={state.status === "success" ? state.metadata?.duration_seconds || 0 : 0}
                     mode={sourceMode}
                  />
                </div>
             </div>
          </div>
        </div>
      </div>

      {/* FULL WIDTH: EFFECT CHAIN */}
      <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c1a]/80 backdrop-blur-sm p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-6 border-b border-white/5 pb-4">
          <div className="flex items-center gap-2">
            <Settings2 className="h-4 w-4 text-purple-400" />
            <h3 className="text-xs font-bold tracking-widest text-slate-300 uppercase">Effect Chain</h3>
          </div>
          
          <AddEffectDropdown onAdd={addEffect} />
        </div>

        {effectChain.length === 0 ? (
          <div className="text-center py-16 border border-dashed border-white/10 rounded-xl bg-white/[0.02]">
            <Settings2 className="h-8 w-8 text-slate-600 mx-auto mb-3" />
            <p className="text-xs text-slate-500 uppercase tracking-widest">No effects added</p>
            <p className="text-[10px] text-slate-600 mt-2 font-mono">Audio will pass through unaltered.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {effectChain.map((ef, i) => (
              <div key={i} className={`p-5 flex flex-col rounded-xl border transition-all duration-300 ${ef.enabled ? "bg-[#161625] border-white/10 shadow-lg shadow-black/20" : "bg-black/40 border-white/5 opacity-50"}`}>
                <div className="flex justify-between items-center mb-6">
                  <div className="flex items-center gap-4">
                    <div className="flex flex-col gap-1 items-center justify-center bg-black/60 rounded-lg p-1 border border-white/5">
                      <button onClick={() => moveEffect(i, -1)} disabled={i===0} className="text-slate-500 hover:text-white disabled:opacity-20 py-1 transition"><svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" /></svg></button>
                      <span className="text-[10px] font-mono text-slate-400 bg-white/5 px-1.5 rounded">{String(i+1).padStart(2, '0')}</span>
                      <button onClick={() => moveEffect(i, 1)} disabled={i===effectChain.length-1} className="text-slate-500 hover:text-white disabled:opacity-20 py-1 transition"><svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg></button>
                    </div>
                    <div>
                      <h4 className="text-sm font-bold tracking-widest uppercase text-white">{ef.op.replace("_", " ")}</h4>
                      {ef.op === "pitch_shift" && <p className="text-[10px] text-slate-500 mt-1 font-mono leading-tight">Preserves approx. duration.</p>}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={() => updateEffect(i, { ...ef, enabled: !ef.enabled })} className={`p-2 rounded-lg transition-colors ${ef.enabled ? "bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30" : "bg-white/5 text-slate-500 hover:bg-white/10 hover:text-white"}`} title="Toggle Effect">
                      <Power className="w-4 h-4" />
                    </button>
                    <button onClick={() => removeEffect(i)} className="p-2 rounded-lg bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 transition-colors" title="Remove Effect">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Effect Parameters */}
                <div className="flex-1 mt-4">
                  <div className="grid grid-cols-2 gap-x-6 gap-y-6 w-full">
                    {ef.op === "gain" && (
                      <div className="col-span-2 w-full flex flex-col sm:flex-row gap-6">
                         <div className="flex flex-col gap-1.5 min-w-[120px]">
                           <label className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">Mode</label>
                           <div className="flex gap-1 bg-black/40 p-1 rounded-lg border border-white/5 w-max">
                             {(["db", "linear", "peak", "rms"] as const).map(m => (
                               <button key={m} onClick={() => updateEffect(i, { ...ef, mode: m })} className={`px-3 py-1 text-[10px] font-bold uppercase rounded-md transition-colors ${ef.mode === m ? "bg-indigo-500/20 text-indigo-300" : "text-slate-500 hover:text-white"}`}>{m}</button>
                             ))}
                           </div>
                         </div>
                         <Knob label="Value" value={ef.gain_value} min={-40} max={40} step={0.5} unit={ef.mode==="db"?"dB":""} onChange={v => updateEffect(i, { ...ef, gain_value: v })} />
                      </div>
                    )}
                    {ef.op === "pitch_shift" && (
                      <div className="col-span-2 w-full"><Knob label="Semitones" value={ef.semitones} min={-12} max={12} step={0.5} unit=" st" onChange={v => updateEffect(i, { ...ef, semitones: v })} /></div>
                    )}
                    {ef.op === "speed" && (
                      <div className="col-span-2 w-full"><Knob label="Speed Factor" value={ef.speed} min={0.25} max={4.0} step={0.05} onChange={v => updateEffect(i, { ...ef, speed: v })} /></div>
                    )}
                    {ef.op === "time_stretch" && (
                      <div className="col-span-2 w-full"><Knob label="Stretch Factor" value={ef.stretch} min={0.25} max={4.0} step={0.05} onChange={v => updateEffect(i, { ...ef, stretch: v })} /></div>
                    )}
                    {ef.op === "echo_delay" && (
                      <>
                        <Knob label="Delay Time" value={ef.delay_ms} min={10} max={1000} step={1} unit=" ms" precision={0} onChange={v => updateEffect(i, { ...ef, delay_ms: v })} />
                        <Knob label="Feedback" value={ef.feedback * 100} min={0} max={99} step={1} unit="%" precision={0} onChange={v => updateEffect(i, { ...ef, feedback: v / 100 })} />
                        <div className="col-span-2 w-full"><Knob label="Mix" value={ef.mix * 100} min={0} max={100} step={1} unit="%" precision={0} onChange={v => updateEffect(i, { ...ef, mix: v / 100 })} /></div>
                      </>
                    )}
                    {ef.op === "reverb" && (
                      <>
                        <Knob label="Room Size" value={ef.room_size} min={0.1} max={1.0} step={0.05} onChange={v => updateEffect(i, { ...ef, room_size: v })} />
                        <Knob label="Decay" value={ef.decay} min={0.1} max={0.99} step={0.05} onChange={v => updateEffect(i, { ...ef, decay: v })} />
                        <div className="col-span-2 w-full"><Knob label="Wet Mix" value={ef.wet * 100} min={0} max={100} step={1} unit="%" precision={0} onChange={v => updateEffect(i, { ...ef, wet: v / 100 })} /></div>
                      </>
                    )}
                    {ef.op === "chorus" && (
                      <>
                         <Knob label="Rate" value={ef.rate_hz} min={0.1} max={10} step={0.1} unit=" Hz" onChange={v => updateEffect(i, { ...ef, rate_hz: v })} />
                         <Knob label="Depth" value={ef.depth_ms} min={1} max={20} step={0.5} unit=" ms" onChange={v => updateEffect(i, { ...ef, depth_ms: v })} />
                         <div className="col-span-2 w-full"><Knob label="Mix" value={ef.mix * 100} min={0} max={100} step={1} unit="%" precision={0} onChange={v => updateEffect(i, { ...ef, mix: v / 100 })} /></div>
                      </>
                    )}
                    {ef.op === "tremolo" && (
                      <>
                         <Knob label="Rate" value={ef.rate_hz} min={0.1} max={20} step={0.1} unit=" Hz" onChange={v => updateEffect(i, { ...ef, rate_hz: v })} />
                         <Knob label="Depth" value={ef.depth * 100} min={0} max={100} step={1} unit="%" precision={0} onChange={v => updateEffect(i, { ...ef, depth: v / 100 })} />
                      </>
                    )}
                    {ef.op === "soft_distortion" && (
                      <div className="col-span-2 w-full"><Knob label="Drive" value={ef.drive} min={1} max={100} step={1} precision={0} onChange={v => updateEffect(i, { ...ef, drive: v })} /></div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* FULL WIDTH: PROCESS */}
      <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c1a]/80 backdrop-blur-sm p-6 shadow-2xl flex flex-col items-center">
         {processError && <div className="text-rose-400 text-xs font-mono mb-4 bg-rose-500/10 p-2 rounded-lg w-full text-center border border-rose-500/20">{processError}</div>}
         <button
           disabled={state.status !== "success" || processing || isRecording}
           onClick={handleProcess}
           className="w-full py-4 rounded-xl font-bold tracking-widest uppercase bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white transition flex items-center justify-center gap-2 shadow-lg shadow-indigo-900/20"
         >
           {processing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5 fill-current" />}
           {processing ? "Processing Audio Chain..." : "Process Audio"}
         </button>
      </div>

      {/* FULL WIDTH: OBSERVATORY (PROCESSED AUDIO) */}
      {result && procAudioUrl && state.status === "success" && (
        <div className="mt-12 rounded-2xl border border-indigo-500/30 bg-[#0c0c1a] shadow-2xl overflow-hidden">
          <div className="bg-indigo-900/20 p-6 border-b border-indigo-500/20">
            <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
              <SectionHeader title="Processed Observatory" icon={CheckCircle} color="text-indigo-400" />
              
              <div className="flex items-center gap-4">
                <div className="text-xs text-slate-500 font-mono flex flex-wrap gap-2">
                  {result.operations_applied.map((op, idx) => (
                     <span key={idx} className="bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 px-2 py-1 rounded-full">{op}</span>
                  ))}
                </div>
                
                <a 
                  href={procAudioUrl} 
                  download="processed_voice.wav"
                  className="px-4 py-1.5 rounded-full bg-indigo-600 hover:bg-indigo-500 text-[10px] font-bold uppercase tracking-widest text-white transition shadow-lg shadow-indigo-900/20"
                >
                  Download WAV
                </a>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              <div className="bg-black/40 p-4 rounded-xl border border-white/5 text-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Input Duration</div>
                <div className="text-lg font-mono text-slate-300">{result.input_duration_s.toFixed(2)}s</div>
              </div>
              <div className="bg-indigo-900/20 p-4 rounded-xl border border-indigo-500/20 text-center">
                <div className="text-[10px] text-indigo-400 uppercase tracking-widest mb-1">Output Duration</div>
                <div className="text-lg font-mono text-indigo-300">{result.output_duration_s.toFixed(2)}s</div>
              </div>
              <div className="bg-black/40 p-4 rounded-xl border border-white/5 text-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Output Peak</div>
                <div className="text-lg font-mono text-emerald-400">{result.output_peak.toFixed(3)}</div>
              </div>
              <div className="bg-black/40 p-4 rounded-xl border border-white/5 text-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Output RMS</div>
                <div className="text-lg font-mono text-cyan-400">{result.output_rms.toFixed(3)}</div>
              </div>
            </div>
          </div>
          
          <div className="p-6">
            <SignalComparison
              originalWaveform={state.waveform}
              processedWaveform={procWaveform}
              originalSpectrum={state.spectrum}
              processedSpectrum={procSpectrum}
              originalSpectrogram={state.spectrogram}
              processedSpectrogram={procSpectrogram}
              originalAudioUrl={state.objectUrl}
              processedAudioUrl={procAudioUrl}
              originalDuration={result.input_duration_s}
              processedDuration={result.output_duration_s}
              processedLabel="Processed"
            />
          </div>
        </div>
      )}
    </div>
  );
}
