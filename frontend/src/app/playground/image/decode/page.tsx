"use client";

import { useState, useRef } from "react";
import { decodeImageToAudio } from "@/lib/api";
import { Loader2, Upload, FileAudio, ShieldCheck, Download, AlertCircle, RefreshCw } from "lucide-react";
import { AudioPlayer } from "@/components/audio/audio-player";
import { usePlayground } from "@/contexts/playground-context";

export default function DecodePage() {
  const { setProcessedAudio, uploadFile } = usePlayground();
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isDecoding, setIsDecoding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{
    protocol: string;
    dim: string;
    payloadSize: string;
    sampleRate: string;
    channels: string;
    sampleCount: string;
  } | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setAudioUrl(null);
      setStats(null);
      setError(null);
    }
  };

  const handleDecode = async () => {
    if (!file) return;
    setIsDecoding(true);
    setError(null);
    try {
      const { blob, headers } = await decodeImageToAudio(file);
      
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(blob));
      
      const recoveredFile = new File([blob], `recovered_audio_${Date.now()}.wav`, { type: 'audio/wav' });
      await uploadFile(recoveredFile);
      setProcessedAudio(blob);
      
      setStats({
        protocol: headers.get("X-Rigel-Protocol") || "Unknown",
        dim: headers.get("X-Rigel-Image-Dim") || "0",
        payloadSize: headers.get("X-Rigel-Payload-Bytes") || "0",
        sampleRate: headers.get("X-Rigel-Sample-Rate") || "0",
        channels: headers.get("X-Rigel-Channels") || "0",
        sampleCount: headers.get("X-Rigel-Sample-Count") || "0",
      });
      
    } catch (err: any) {
      setError(err.message || "Failed to decode image.");
    } finally {
      setIsDecoding(false);
    }
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1fr_350px] gap-8 h-full">
      {/* Main Viewport */}
      <div className="flex flex-col rounded-2xl border border-white/10 bg-[#0c0c16] overflow-hidden shadow-2xl">
        <div className="flex items-center gap-3 border-b border-white/5 bg-white/[0.02] px-6 py-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Decode Audio Image</h2>
            <p className="text-xs text-slate-400">Upload a PNG data container to extract its audio payload</p>
          </div>
        </div>
        
        <div className="flex flex-col p-6 items-center justify-center bg-black/40 min-h-[500px]">
          {isDecoding ? (
            <div className="flex flex-col items-center gap-4 text-indigo-400">
              <Loader2 className="h-12 w-12 animate-spin" />
              <p className="text-lg font-medium animate-pulse">Extracting and Validating Payload...</p>
            </div>
          ) : audioUrl ? (
            <div className="flex flex-col items-center gap-8 w-full max-w-md">
              <div className="flex flex-col items-center gap-2">
                <div className="h-16 w-16 rounded-full bg-indigo-500/20 flex items-center justify-center text-indigo-400">
                  <ShieldCheck className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-bold text-white mt-2">Payload Extracted</h3>
                <p className="text-slate-400 text-sm">CRC32 Checksums validated successfully</p>
              </div>
              
              <div className="w-full mt-6">
                <AudioPlayer 
                  src={audioUrl} 
                  duration={stats ? parseFloat(stats.sampleCount) / parseFloat(stats.sampleRate) : 0} 
                />
              </div>
              
              <button
                onClick={() => {
                  setFile(null);
                  setAudioUrl(null);
                  setStats(null);
                  setError(null);
                }}
                className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 px-6 py-3 font-medium text-white transition-all w-full mt-6"
              >
                <RefreshCw className="h-4 w-4" />
                Decode a Different Image
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-6">
              <div 
                className="w-80 max-w-[90%] aspect-square border-2 border-dashed border-white/20 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all p-4"
                onClick={() => fileInputRef.current?.click()}
              >
                {file ? (
                  <div className="flex flex-col items-center gap-3 text-indigo-400 w-full">
                    <FileAudio className="h-12 w-12" />
                    <div className="w-full px-4 text-center overflow-hidden">
                      <p className="font-medium truncate w-full" title={file.name}>{file.name}</p>
                    </div>
                    <p className="text-xs text-indigo-400/60">{(file.size / 1024).toFixed(1)} KB</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-slate-400">
                    <Upload className="h-10 w-10 mb-2 opacity-50" />
                    <p className="font-medium">Upload PNG Image</p>
                    <p className="text-xs opacity-50">Click to browse</p>
                  </div>
                )}
              </div>
              <input 
                type="file" 
                accept="image/png" 
                className="hidden" 
                ref={fileInputRef}
                onChange={handleFileSelect}
              />
              
              {error && (
                <div className="flex items-center gap-2 text-red-400 bg-red-400/10 px-4 py-3 rounded-lg border border-red-400/20">
                  <AlertCircle className="h-5 w-5 flex-shrink-0" />
                  <p className="text-sm font-medium">{error}</p>
                </div>
              )}
              
              <button
                onClick={handleDecode}
                disabled={!file}
                className="flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed px-8 py-4 font-semibold text-white transition-all shadow-lg"
              >
                <ShieldCheck className="h-5 w-5" />
                Decode Image
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Stats Panel */}
      <div className="flex flex-col gap-6">
        <div className="rounded-2xl border border-white/10 bg-[#0c0c16] overflow-hidden shadow-2xl p-6 flex flex-col gap-6">
          <h3 className="font-semibold text-lg text-white">Recovery Statistics</h3>
          
          <div className="space-y-4 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Protocol:</span>
              <span className="text-indigo-400 font-mono font-bold">{stats ? stats.protocol : "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Image Dimensions:</span>
              <span className="text-white font-mono">{stats ? `${stats.dim} × ${stats.dim}` : "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Payload Size:</span>
              <span className="text-white font-mono">{stats ? (parseInt(stats.payloadSize) / 1024).toFixed(2) + " KB" : "-"}</span>
            </div>
            <div className="h-px bg-white/10 my-2"></div>
            <div className="flex justify-between">
              <span className="text-slate-400">Sample Rate:</span>
              <span className="text-white font-mono">{stats ? `${stats.sampleRate} Hz` : "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Channels:</span>
              <span className="text-white font-mono">{stats ? stats.channels : "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Recovered Samples:</span>
              <span className="text-white font-mono">{stats ? parseInt(stats.sampleCount).toLocaleString() : "-"}</span>
            </div>
          </div>
          
          <div className="h-px w-full bg-white/10 my-2"></div>

          {audioUrl && (
            <a
              href={audioUrl}
              download={`recovered_audio_${Date.now()}.wav`}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-3 text-sm font-semibold text-white transition-all shadow-lg shadow-indigo-900/20"
            >
              <Download className="h-4 w-4" />
              Download WAV
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
