"use client";

import { useState } from "react";
import { usePlayground } from "@/contexts/playground-context";
import { encodeAudioToImage } from "@/lib/api";
import { Loader2, Image as ImageIcon, Download, FileAudio, Lock } from "lucide-react";

export default function EncodePage() {
  const { state } = usePlayground();

  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{
    dim: string;
    payloadSize: string;
    pcmSize: string;
    flacSize: string;
    ratio: string;
  } | null>(null);

  const handleEncode = async () => {
    if (state.status !== "success") return;
    setIsGenerating(true);
    setError(null);
    setStats(null);
    try {
      const fileToProcess = state.processedAudio ? state.processedAudio : state.file;
      const { blob, headers } = await encodeAudioToImage(fileToProcess);
      
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      setBlobUrl(URL.createObjectURL(blob));
      
      setStats({
        dim: headers.get("X-Rigel-Image-Dim") || "0",
        payloadSize: headers.get("X-Rigel-Payload-Bytes") || "0",
        pcmSize: headers.get("X-Rigel-PCM-Bytes") || "0",
        flacSize: headers.get("X-Rigel-FLAC-Bytes") || "0",
        ratio: headers.get("X-Rigel-Compression-Ratio") || "0"
      });
      
    } catch (err: any) {
      setError(err.message || "Failed to encode audio to image.");
    } finally {
      setIsGenerating(false);
    }
  };

  if (state.status !== "success") {
    return (
      <div className="flex h-full flex-col items-center justify-center p-8 text-center bg-black/20 rounded-xl border border-white/5">
        <FileAudio className="mb-4 h-12 w-12 text-slate-500/50" />
        <h2 className="text-xl font-semibold text-white">No Audio Selected</h2>
        <p className="mt-2 text-slate-400 max-w-md">
          Please upload or select an audio file from the sidebar to begin encoding.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1fr_350px] gap-8 h-full">
      {/* Main Viewport */}
      <div className="flex flex-col rounded-2xl border border-white/10 bg-[#0c0c16] overflow-hidden shadow-2xl">
        <div className="flex items-center gap-3 border-b border-white/5 bg-white/[0.02] px-6 py-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400">
            <Lock className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Encoded Data Image</h2>
            <p className="text-xs text-slate-400">RGL1 Protocol • Exact PCM Payload</p>
          </div>
        </div>
        
        <div className="flex flex-col p-6 items-center justify-center bg-black/40 min-h-[500px]">
          {isGenerating ? (
            <div className="flex flex-col items-center gap-4 text-indigo-400">
              <Loader2 className="h-12 w-12 animate-spin" />
              <p className="text-lg font-medium animate-pulse">Encoding Audio Payload...</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center text-red-400">
              <p className="font-medium text-center max-w-md">{error}</p>
            </div>
          ) : blobUrl ? (
            <div className="flex flex-col items-center">
              <img 
                src={blobUrl} 
                alt="Encoded Audio Data" 
                className="w-full max-w-[512px] h-auto rounded-md shadow-[0_0_50px_rgba(99,102,241,0.15)] border border-white/10 pixelated" 
                style={{ imageRendering: 'pixelated' }}
              />
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <button
                onClick={handleEncode}
                className="flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 px-6 py-4 font-semibold text-white transition-all shadow-lg shadow-indigo-900/20"
              >
                <Lock className="h-5 w-5" />
                Encode Audio to Image
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Stats Panel */}
      <div className="flex flex-col gap-6">
        <div className="rounded-2xl border border-white/10 bg-[#0c0c16] overflow-hidden shadow-2xl p-6 flex flex-col gap-6">
          <h3 className="font-semibold text-lg text-white">Encoding Statistics</h3>
          
          <div className="space-y-4 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Original PCM Size:</span>
              <span className="text-white font-mono">{stats ? (parseInt(stats.pcmSize) / 1024).toFixed(2) + " KB" : "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">FLAC Compressed Size:</span>
              <span className="text-white font-mono">{stats ? (parseInt(stats.flacSize) / 1024).toFixed(2) + " KB" : "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Compression Ratio:</span>
              <span className="text-indigo-400 font-mono font-bold">
                {stats ? (parseFloat(stats.ratio) * 100).toFixed(1) + "%" : "-"}
              </span>
            </div>
            <div className="h-px bg-white/10 my-2"></div>
            <div className="flex justify-between">
              <span className="text-slate-400">Image Dimensions:</span>
              <span className="text-white font-mono">{stats ? `${stats.dim} × ${stats.dim}` : "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Total Payload (w/ Header):</span>
              <span className="text-white font-mono">{stats ? (parseInt(stats.payloadSize) / 1024).toFixed(2) + " KB" : "-"}</span>
            </div>
          </div>
          
          <div className="h-px w-full bg-white/10 my-2"></div>
          
          <button
            onClick={handleEncode}
            disabled={isGenerating}
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 px-4 py-3 text-sm font-semibold text-white transition-all disabled:opacity-50"
          >
            Regenerate Image
          </button>

          {blobUrl && (
            <a
              href={blobUrl}
              download="encoded_audio.png"
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-3 text-sm font-semibold text-white transition-all"
            >
              <Download className="h-4 w-4" />
              Download PNG
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
