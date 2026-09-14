import { Github, Linkedin, Instagram, Facebook } from "lucide-react";

export function Footer() {
  const socialLinks = [
    {
      name: "GitHub",
      url: "https://github.com/mahmudmaahi",
      icon: Github,
    },
    {
      name: "LinkedIn",
      url: "https://www.linkedin.com/in/maahimahmud/",
      icon: Linkedin,
    },
    {
      name: "Instagram",
      url: "https://www.instagram.com/maahi_mahmud/",
      icon: Instagram,
    },
    {
      name: "Facebook",
      url: "https://www.facebook.com/mahmudmaahi",
      icon: Facebook,
    },
  ];

  return (
    <footer className="relative mt-auto border-t border-white/5 bg-gradient-to-b from-transparent to-[#080811] backdrop-blur-sm">
      <div className="mx-auto w-full max-w-6xl px-5 py-12 sm:px-8 lg:px-12">
        <div className="flex flex-col gap-10 sm:flex-row sm:items-start sm:justify-between">
          {/* Brand & Description */}
          <div className="max-w-md">
            <div className="flex items-center gap-3">
              <img 
                src="/logo.png" 
                alt="Rigel Logo" 
                className="h-10 w-10 object-contain filter drop-shadow-[0_0_12px_rgba(255,255,255,0.5)]" 
              />
              <h3 
                className="text-4xl font-bold tracking-[0.25em] text-white leading-none mt-2"
                style={{ fontFamily: "'Rivage', serif" }}
              >
                RIGEL
              </h3>
            </div>
            <p className="mt-4 text-sm leading-relaxed text-slate-400">
              Audio processing built from first principles. A digital signal
              processing workbench for exploring, visualizing, and analyzing
              discrete-time audio signals.
            </p>
            <div className="mt-4">
              <span className="rounded-full border border-indigo-500/20 bg-indigo-500/10 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-indigo-300">
                Team Orion
              </span>
            </div>
          </div>

          {/* Social Links */}
          <div>
            <p className="mb-4 font-mono text-xs uppercase tracking-wider text-slate-400">
              Connect
            </p>
            <div className="flex gap-3">
              {socialLinks.map((link) => {
                const Icon = link.icon;
                return (
                  <a
                    key={link.name}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex h-11 w-11 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-slate-400 backdrop-blur-sm transition-all hover:border-indigo-500/30 hover:bg-indigo-500/10 hover:text-indigo-300 hover:scale-105"
                    aria-label={link.name}
                  >
                    <Icon className="h-4 w-4" />
                  </a>
                );
              })}
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-10 flex flex-col items-center justify-between gap-4 border-t border-white/5 pt-6 text-xs text-slate-500 sm:flex-row">
          <p>© {new Date().getFullYear()} Team Orion. Open source under MIT.</p>
          <div className="flex items-center gap-3 font-mono text-[11px] tracking-wider">
            <span>PCM WAV</span>
            <span className="opacity-30">•</span>
            <span>DISCRETE-TIME DSP</span>
            <span className="opacity-30">•</span>
            <span>CANVAS</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
