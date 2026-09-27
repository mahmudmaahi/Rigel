"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Image as ImageIcon, Lock, ShieldCheck, ArrowRight } from "lucide-react";

const TABS = [
  {
    href: "/playground/image/encode",
    label: "Encode",
    direction: "Audio → Data Image",
    icon: Lock,
  },
  {
    href: "/playground/image/decode",
    label: "Decode",
    direction: "Data Image → Audio",
    icon: ShieldCheck,
  },
];

export default function ImageLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex flex-col gap-6 p-6 md:p-8 overflow-y-auto w-full max-w-[1600px] mx-auto h-full scrollbar-none">

      {/* ── Page header ── */}
      <div className="flex flex-col items-center justify-center text-center mt-4 mb-6">
        <h1 className="flex items-center justify-center gap-3 text-3xl font-bold tracking-tight sm:text-4xl text-white font-display">
          <ImageIcon className="h-8 w-8 text-indigo-400" />
          Audio ↔ Image
        </h1>
        <p className="mt-4 text-base text-slate-400 max-w-2xl mx-auto">
          Encode audio losslessly into a data image (RGL1 protocol), and recover it back into audio.
        </p>
      </div>

      <div className="flex justify-center mb-4">
        <div className="flex gap-2 border-b border-white/10 px-4">
          {TABS.map((tab) => {
            const isActive = pathname === tab.href;
            const Icon = tab.icon;
            return (
              <Link
                key={tab.href}
                href={tab.href}
                className={`flex flex-col items-center gap-1 px-6 py-3 border-b-2 transition-all ${
                  isActive
                    ? "border-indigo-400 text-white"
                    : "border-transparent text-slate-400 hover:text-white hover:border-indigo-400/50"
                }`}
              >
                <span className="flex items-center gap-2 text-sm font-bold tracking-widest uppercase">
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </span>
                <span className="flex items-center gap-1 text-[10px] font-mono tracking-wide text-slate-500">
                  {tab.direction.split(" → ")[0]}
                  <ArrowRight className="h-3 w-3" />
                  {tab.direction.split(" → ")[1]}
                </span>
              </Link>
            );
          })}
        </div>
      </div>

      <div className="flex-1 min-h-0">
        {children}
      </div>
    </div>
  );
}
