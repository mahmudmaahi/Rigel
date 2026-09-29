"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Mic, MicOff, Activity, Clock, Zap, FileAudio, Play } from "lucide-react";
import { usePlayground } from "@/contexts/playground-context";
import { AudioPlayer, AudioPlayerRef } from "@/components/audio/audio-player";
import { useVadHistory, VadResult } from "@/hooks/use-vad-history";
import { VadVisualization } from "@/components/vad/vad-visualization";

type SourceMode = "microphone" | "shared";

function encodeToWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  
  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7FFF;
    view.setInt16(offset, s, true);
  }
  
  return new Blob([buffer], { type: 'audio/wav' });
}

export default function SpeechVadPage() {
  const { state: sharedState, uploadFile } = usePlayground();
  
  const [sourceMode, setSourceMode] = useState<SourceMode>("shared");
  const [engineError, setEngineError] = useState<string | null>(null);
  const [vadEngine, setVadEngine] = useState<"silero" | "ten_vad">("silero");
  const [confirmedEngine, setConfirmedEngine] = useState<string | null>(null);
  
  const [isRecording, setIsRecording] = useState(false);
  const [isPlayingShared, setIsPlayingShared] = useState(false);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
  const [sampleRate, setSampleRate] = useState<number | null>(null);

  const diagnosticChunksRef = useRef<Float32Array[]>([]);
  const [diagnosticWavUrl, setDiagnosticWavUrl] = useState<string | null>(null);
  const [micStats, setMicStats] = useState({ chunks: 0, samples: 0 });
  
  const { history, segments, latestResult: vadResult, addResult, clearHistory } = useVadHistory(sourceMode === "microphone" ? 10 : 0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Shared Audio state
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const audioPlayerRef = useRef<AudioPlayerRef>(null);
  const sampleCursorRef = useRef<number>(0);
  const requestRef = useRef<number | undefined>(undefined);
  const baseTimeRef = useRef<number>(0);

  const lastDecodedUrlRef = useRef<string | null>(null);
  
  // Decode audio buffer when shared file changes
  useEffect(() => {
    if (sharedState.status === "success" && sourceMode === "shared") {
      if (lastDecodedUrlRef.current === sharedState.objectUrl) return;
      lastDecodedUrlRef.current = sharedState.objectUrl;
      
      const ctx = new window.AudioContext();
      sharedState.file.arrayBuffer().then(buffer => {
        ctx.decodeAudioData(buffer).then(decoded => {
          setAudioBuffer(decoded);
          
          // Clear VAD state, reset cursor, and stop playback for the new file
          clearHistory();
          sampleCursorRef.current = 0;
          baseTimeRef.current = 0;
          
          setIsPlayingShared(false);
          if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
          }
          
          if (wsRef.current) {
            if (wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ event: "stop" }));
            }
            wsRef.current.close();
            wsRef.current = null;
          }
          setWsStatus("disconnected");
          
          ctx.close();
        }).catch(err => {
          console.error("Failed to decode audio", err);
          ctx.close();
        });
      });
    }
  }, [sharedState, sourceMode]);

  const stopAll = useCallback(() => {
    setIsRecording(false);
    setIsPlayingShared(false);
    
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = undefined;
    }
    
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
    }
    
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ event: "stop" }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    if (diagnosticChunksRef.current.length > 0 && audioContextRef.current) {
      const totalLength = diagnosticChunksRef.current.reduce((acc, val) => acc + val.length, 0);
      const combined = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of diagnosticChunksRef.current) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      const wavBlob = encodeToWav(combined, audioContextRef.current.sampleRate);
      const url = URL.createObjectURL(wavBlob);
      setDiagnosticWavUrl(url);
      setMicStats({
        chunks: diagnosticChunksRef.current.length,
        samples: totalLength
      });
      diagnosticChunksRef.current = [];
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setWsStatus("disconnected");
    setEngineError(null);
    setConfirmedEngine(null);
    setSampleRate(null);
    clearHistory();
    sampleCursorRef.current = 0;
    baseTimeRef.current = 0;
  }, []);

  // Mode Switch
  const handleModeSwitch = (mode: SourceMode) => {
    if (mode === sourceMode) return;
    stopAll();
    setSourceMode(mode);
  };

  const connectWebSocket = (engine: "silero" | "ten_vad", sr: number, isShared: boolean): Promise<void> => {
    return new Promise((resolve, reject) => {
      setWsStatus("connecting");
      setEngineError(null);
      
      const ws = new WebSocket("ws://localhost:8000/api/vad/stream");
      wsRef.current = ws;
      
      ws.onopen = () => {
        setWsStatus("connected");
        ws.send(JSON.stringify({
          event: "start",
          sampleRate: sr,
          channels: 1,
          format: "float32",
          engine: engine
        }));
        resolve();
      };
      
      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          try {
            const data = JSON.parse(event.data);
            if (data.error) {
              console.error("VAD Server Error:", data.error);
              setEngineError(data.error);
            } else if (data.event === "started") {
              setConfirmedEngine(data.engine || engine);
            } else if (data.state) {
              setConfirmedEngine(data.engine); // backup if started was missed
              const res = data as VadResult;
              if (isShared) {
                res.timestamp_seconds += baseTimeRef.current;
              }
              addResult(res);
            }
          } catch (e) {
            console.error("Error parsing JSON:", e);
          }
        }
      };
      
      ws.onerror = (e) => {
          console.error("WS Error:", e);
          reject(e);
      };
      
      ws.onclose = () => {
         if (!isShared && isRecording) {
            stopAll();
         } else {
            setWsStatus("disconnected");
            wsRef.current = null;
         }
      };
    });
  };

  const handleEngineChange = async (newEngine: "silero" | "ten_vad") => {
    if (newEngine === vadEngine) return;
    
    // Save state
    localStorage.setItem("selectedVadEngine", newEngine);
    setVadEngine(newEngine);
    
    // Disconnect old session gracefully
    if (wsRef.current) {
      wsRef.current.onclose = null; // Prevent stopAll
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setEngineError(null);
    setConfirmedEngine(null);
    clearHistory();
    
    // Reconnect new session immediately if active (or if file is loaded for shared audio)
    if (sourceMode === "microphone" && isRecording && sampleRate) {
      await connectWebSocket(newEngine, sampleRate, false);
    } else if (sourceMode === "shared" && sampleRate) {
      await connectWebSocket(newEngine, sampleRate, true);
    }
  };

  // ---------------------------------------------------------------------------
  // MICROPHONE MODE LOGIC
  // ---------------------------------------------------------------------------
  const startRecording = async () => {
    try {
      setDiagnosticWavUrl(null);
      diagnosticChunksRef.current = [];
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        } 
      });
      mediaStreamRef.current = stream;

      const audioCtx = new window.AudioContext();
      audioContextRef.current = audioCtx;
      setSampleRate(audioCtx.sampleRate);

      await audioCtx.audioWorklet.addModule("/worklet-processor.js");

      const source = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, "vad-processor");
      workletNodeRef.current = workletNode;

      source.connect(workletNode);
      workletNode.connect(audioCtx.destination);

      await connectWebSocket(vadEngine, audioCtx.sampleRate, false);
      setIsRecording(true);

      // Receive raw PCM from worklet and forward to backend
      workletNode.port.onmessage = (event) => {
        const buffer = event.data; // Float32Array
        diagnosticChunksRef.current.push(new Float32Array(buffer));
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(buffer);
        }
      };

    } catch (error) {
      console.error("Error starting microphone:", error);
      stopAll();
    }
  };

  // ---------------------------------------------------------------------------
  // SHARED AUDIO LOGIC
  // ---------------------------------------------------------------------------
  const connectSharedWebSocket = (): Promise<void> => {
    if (!audioBuffer) return Promise.reject(new Error("No audio buffer"));
    setSampleRate(audioBuffer.sampleRate);
    return connectWebSocket(vadEngine, audioBuffer.sampleRate, true);
  };

  const sendNextChunk = () => {
    if (!audioBuffer || !audioPlayerRef.current) return;
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const currentTime = audioPlayerRef.current.getCurrentTime();
      const targetCursor = Math.floor(currentTime * audioBuffer.sampleRate);
      const currentCursor = sampleCursorRef.current;
      
      // Always send channelData[sampleCursor:targetCursor] and advance
      if (targetCursor > currentCursor && targetCursor <= audioBuffer.length) {
        const channelData = audioBuffer.getChannelData(0); // Strict mono policy
        const chunk = channelData.subarray(currentCursor, targetCursor);
        wsRef.current.send(chunk);
        sampleCursorRef.current = targetCursor;
      }
    }
    
    // Always request the next frame to keep checking currentTime
    requestRef.current = requestAnimationFrame(sendNextChunk);
  };

  const handleSharedPlay = () => {
    setIsPlayingShared(true);
    
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
       connectSharedWebSocket().then(() => {
           requestRef.current = requestAnimationFrame(sendNextChunk);
       }).catch(() => {
           setIsPlayingShared(false);
       });
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
    // 1. Reset history and time cursor
    clearHistory();
    baseTimeRef.current = time;
    
    // 2. Sync sampleCursor with new position
    if (audioBuffer) {
        sampleCursorRef.current = Math.floor(time * audioBuffer.sampleRate);
    } else {
        sampleCursorRef.current = 0;
    }

    // 3. Reset VAD session cleanly without tearing down the socket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ event: "stop" }));
        
        if (audioBuffer) {
            wsRef.current.send(JSON.stringify({
                event: "start",
                sampleRate: audioBuffer.sampleRate,
                channels: 1,
                format: "float32",
                engine: vadEngine
            }));
        }
    } else if (isPlayingShared) {
        // If it was playing but the socket died, we should reconnect
        connectSharedWebSocket().then(() => {
           if (!requestRef.current) {
               requestRef.current = requestAnimationFrame(sendNextChunk);
           }
        }).catch(err => {
            console.error(err);
            setIsPlayingShared(false);
        });
    }
  };

  const handleSharedEnded = () => {
    handleSharedPause();
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ event: "stop" }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsStatus("disconnected");
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAll();
    };
  }, [stopAll]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col items-center justify-center text-center mt-4 mb-10">
        <h1 className="flex items-center justify-center gap-3 text-3xl font-bold tracking-tight sm:text-4xl text-white">
          <Activity className="h-8 w-8 text-indigo-400" />
          Real-Time Voice Activity Detection
        </h1>
        <p className="mt-4 text-base text-slate-400 max-w-2xl mx-auto">
          Streams uncompressed Float32 PCM directly via Binary WebSockets to the Python DSP backend.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Control Panel */}
        <div className="col-span-1 rounded-2xl border border-white/5 bg-white/5 p-6 backdrop-blur-md flex flex-col gap-6">
          
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
          
          <div>
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
                    onClick={stopAll}
                    className="w-full flex items-center justify-center gap-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 px-4 py-3 text-sm font-medium text-red-400 transition-all"
                  >
                    <MicOff className="h-5 w-5" />
                    Stop Recording
                  </button>
                )}
                
                {diagnosticWavUrl && !isRecording && (
                    <div className="mt-4 p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex flex-col gap-3">
                        <h3 className="text-xs font-bold uppercase tracking-widest text-indigo-400">Raw Capture</h3>
                        <div className="text-xs text-slate-400 flex flex-col gap-1">
                            <div>Sample Rate: {sampleRate} Hz</div>
                            <div>Channels: 1 (Float32 PCM)</div>
                            <div>Chunks Sent: {micStats.chunks}</div>
                            <div>Total Samples: {micStats.samples}</div>
                        </div>
                        <a href={diagnosticWavUrl} download="raw_microphone_capture.wav" className="mt-2 w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 px-3 py-2 text-xs font-medium text-white transition-all">
                            <FileAudio className="h-4 w-4" /> Download WAV
                        </a>
                    </div>
                )}
              </>
            )}

            {sourceMode === "shared" && (
              <div className="space-y-4">
                {sharedState.status === "success" ? (
                  <>
                    <AudioPlayer
                      ref={audioPlayerRef}
                      src={sharedState.objectUrl}
                      duration={sharedState.metadata.duration_seconds}
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

          <div className="space-y-4 pt-4 border-t border-white/5">
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

        {/* Results Panel */}
        <div className="col-span-1 lg:col-span-2 rounded-2xl border border-white/5 bg-white/5 p-6 backdrop-blur-md flex flex-col">
          <h2 className="text-sm font-bold uppercase tracking-widest text-slate-500 mb-6">Live DSP Results</h2>
          
          <div className="flex-1 flex flex-col justify-center items-center gap-8 py-8">
            {/* Active Engine Banner */}
            <div className="absolute top-4 right-4 flex flex-col items-end">
              <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">Detector Status</div>
              <div className="flex items-center gap-2 bg-black/40 border border-white/10 px-3 py-1.5 rounded-full backdrop-blur-sm">
                <div className={`h-2 w-2 rounded-full ${
                  engineError ? 'bg-red-500' :
                  wsStatus === 'disconnected' ? 'bg-slate-500' :
                  wsStatus === 'connecting' ? 'bg-amber-500 animate-pulse' :
                  vadResult?.is_speech ? 'bg-emerald-500 animate-pulse' : 'bg-emerald-500'
                }`}></div>
                <span className="text-xs font-mono font-medium text-white max-w-[200px] truncate">
                  {engineError
                    ? `ERROR: ${engineError}`
                    : wsStatus === "disconnected"
                      ? "DISCONNECTED"
                      : wsStatus === "connecting"
                        ? "CONNECTING..."
                        : confirmedEngine
                          ? "ACTIVE"
                          : "WAITING FOR AUDIO"}
                </span>
              </div>
            </div>

            {/* Main state indicator */}
            <div className="flex flex-col items-center gap-4 mt-8">
              <div className={`h-32 w-32 rounded-full border-4 flex items-center justify-center shadow-[0_0_50px_rgba(0,0,0,0.3)] transition-all duration-300 ${
                vadResult?.is_speech 
                  ? "border-emerald-500/50 bg-emerald-500/10 shadow-emerald-500/20" 
                  : "border-slate-700 bg-slate-800 shadow-transparent"
              }`}>
                {vadResult?.is_speech ? (
                  <Activity className="h-12 w-12 text-emerald-400 animate-pulse" />
                ) : (
                  sourceMode === "microphone" ? <MicOff className="h-12 w-12 text-slate-600" /> : <Play className="h-12 w-12 text-slate-600" />
                )}
              </div>
              <div className="text-center">
                <p className={`text-2xl font-bold tracking-widest transition-colors ${
                  vadResult?.state === "SPEECH" ? "text-emerald-400" :
                  vadResult?.state === "SILENCE" ? "text-slate-500" :
                  "text-slate-500"
                }`}>
                  {vadResult?.state || "WAITING"}
                </p>
                <p className="text-sm text-slate-400 font-mono mt-1">
                  Active: {vadResult?.is_speech ? "YES" : "NO"}
                </p>
              </div>
            </div>

            {/* Visualization */}
            <div className="w-full mt-4">
              <VadVisualization
                history={history}
                segments={segments}
                currentTime={vadResult?.timestamp_seconds ?? 0}
                duration={sourceMode === "shared" && sharedState.status === "success" ? sharedState.metadata.duration_seconds : 0}
                mode={sourceMode}
              />
            </div>

            {/* Metrics Grid */}
            <div className="w-full grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <div className="rounded-xl bg-black/20 p-4 border border-white/5">
                <div className="flex items-center gap-2 text-slate-500 mb-1">
                  <Activity className="h-3 w-3" />
                  <span className="text-xs uppercase tracking-wider font-bold">
                    Probability
                  </span>
                </div>
                <p className="text-lg font-mono text-white">
                  {vadResult ? vadResult.activity_score.toFixed(3) : "0.000"}
                </p>
              </div>
              
              <div className="rounded-xl bg-black/20 p-4 border border-white/5">
                <div className="flex items-center gap-2 text-slate-500 mb-1">
                  <Clock className="h-3 w-3" />
                  <span className="text-xs uppercase tracking-wider font-bold">Time</span>
                </div>
                <p className="text-lg font-mono text-white">
                  {vadResult ? vadResult.timestamp_seconds.toFixed(2) : "0.00"}s
                </p>
              </div>

              <div className="rounded-xl bg-black/20 p-4 border border-white/5">
                <div className="flex items-center gap-2 text-slate-500 mb-1">
                  <Activity className="h-3 w-3" />
                  <span className="text-xs uppercase tracking-wider font-bold">Duration</span>
                </div>
                <p className="text-lg font-mono text-white">
                  {vadResult ? vadResult.speech_duration_frames : 0} f
                </p>
              </div>

              <div className="rounded-xl bg-black/20 p-4 border border-white/5">
                <div className="flex items-center gap-2 text-slate-500 mb-1">
                  <Zap className="h-3 w-3" />
                  <span className="text-xs uppercase tracking-wider font-bold">
                    Inference Time
                  </span>
                </div>
                <p className="text-lg font-mono text-white">
                  {vadResult ? vadResult.compute_ms.toFixed(2) : "0.00"}ms
                </p>
              </div>
              

            </div>
            
            {/* RAW BACKEND RESULT DIAGNOSTIC PANEL */}
            <div className="w-full mt-4 p-4 rounded-xl bg-black/40 border border-white/10 overflow-hidden">
                <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">RAW BACKEND RESULT</h3>
                <pre className="text-xs font-mono text-emerald-400 whitespace-pre-wrap break-words">
                    {vadResult ? JSON.stringify(vadResult, null, 2) : "No frames processed yet."}
                </pre>
            </div>
            
          </div>
        </div>

      </div>
    </div>
  );
}
