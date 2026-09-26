import Link from "next/link";
import { Image as ImageIcon } from "lucide-react";

export default function ImageLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-6 p-6 md:p-8 overflow-y-auto w-full max-w-[1600px] mx-auto h-full scrollbar-none">
      
      {/* ── Page header ── */}
      <div className="flex flex-col items-center justify-center text-center mt-4 mb-6">
        <h1 className="flex items-center justify-center gap-3 text-3xl font-bold tracking-tight sm:text-4xl text-white font-display">
          <ImageIcon className="h-8 w-8 text-indigo-400" />
          Audio Image Encoding Protocol
        </h1>
        <p className="mt-4 text-base text-slate-400 max-w-2xl mx-auto">
          Cryptographically encode and decode exact audio payloads directly into image pixels.
        </p>
      </div>

      <div className="flex justify-center mb-4">
        <div className="flex space-x-2 border-b border-white/10 px-8">
          <Link 
            href="/playground/image/encode"
            className="px-6 py-3 text-sm font-bold tracking-widest uppercase text-slate-400 hover:text-white border-b-2 border-transparent hover:border-indigo-400 transition-all"
          >
            Encode
          </Link>
          <Link 
            href="/playground/image/decode"
            className="px-6 py-3 text-sm font-bold tracking-widest uppercase text-slate-400 hover:text-white border-b-2 border-transparent hover:border-indigo-400 transition-all"
          >
            Decode
          </Link>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        {children}
      </div>
    </div>
  );
}
