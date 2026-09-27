import Link from "next/link";
import { notFound } from "next/navigation";
import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";
import { Card } from "@/components/ui/card";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { WORKSPACE_DOCS, WORKSPACE_ORDER } from "@/lib/docs-content";

export function generateStaticParams() {
  return WORKSPACE_ORDER.map((slug) => ({ slug }));
}

export default async function WorkspaceDocPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const doc = WORKSPACE_DOCS[slug];
  if (!doc) notFound();

  const Icon = doc.icon;

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#080811] text-white transition-colors duration-300">
      <Navbar />

      <main className="relative z-10 pt-32 pb-24 px-5 sm:px-8 lg:px-12">
        <div className="max-w-4xl mx-auto">
          <Link
            href="/docs"
            className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors mb-10"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Docs
          </Link>

          {/* Header */}
          <div className="flex items-start gap-4 mb-4">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border border-indigo-500/20 bg-indigo-500/10">
              <Icon className="h-7 w-7 text-indigo-400" />
            </div>
            <div>
              <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white">{doc.name}</h1>
              <p className="mt-2 text-lg text-slate-400">{doc.tagline}</p>
            </div>
          </div>

          <Link
            href={doc.path}
            className="inline-flex items-center gap-2 mt-4 mb-16 text-sm font-semibold text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            Open this workspace
            <ArrowRight className="h-4 w-4" />
          </Link>

          {/* Overview */}
          <section className="mb-16 space-y-4">
            <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-4">Overview</h2>
            {doc.overview.map((p, i) => (
              <p key={i} className="text-base text-white/80 leading-relaxed">{p}</p>
            ))}
          </section>

          {/* Pipeline */}
          <section className="mb-16">
            <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-6">Pipeline</h2>
            <Card className="p-6">
              <ol className="space-y-3">
                {doc.pipeline.map((step, i) => (
                  <li key={i} className="flex gap-4 text-sm text-slate-300">
                    <span className="shrink-0 flex h-6 w-6 items-center justify-center rounded-full bg-indigo-500/15 text-indigo-400 text-xs font-bold font-mono">
                      {i + 1}
                    </span>
                    <span className="pt-0.5 font-mono leading-relaxed">{step}</span>
                  </li>
                ))}
              </ol>
            </Card>
          </section>

          {/* Algorithms & Formulas */}
          <section className="mb-16">
            <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-6">Algorithms &amp; Formulas</h2>
            <div className="space-y-6">
              {doc.algorithms.map((alg) => (
                <Card key={alg.name} className="p-6">
                  <h3 className="text-base font-semibold text-white mb-2">{alg.name}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed mb-4">{alg.detail}</p>
                  {alg.formula && (
                    <pre className="text-xs font-mono text-cyan-300 bg-black/40 border border-white/5 rounded-lg p-4 overflow-x-auto whitespace-pre-wrap">
                      {alg.formula}
                    </pre>
                  )}
                </Card>
              ))}
            </div>
          </section>

          {/* Files */}
          <section className="mb-16">
            <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-6">Key Files</h2>
            <Card className="p-6">
              <ul className="space-y-3">
                {doc.files.map((f) => (
                  <li key={f.path} className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-4 text-sm">
                    <code className="text-indigo-300 font-mono shrink-0">{f.path}</code>
                    <span className="text-slate-500">{f.note}</span>
                  </li>
                ))}
              </ul>
            </Card>
          </section>

          {/* API */}
          <section className="mb-16">
            <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-6">API</h2>
            <Card className="p-6">
              <ul className="space-y-3">
                {doc.api.map((ep) => (
                  <li key={ep.endpoint} className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-4 text-sm">
                    <span className="shrink-0 font-mono">
                      <span className="text-emerald-400 font-bold">{ep.method}</span>{" "}
                      <span className="text-white">{ep.endpoint}</span>
                    </span>
                    <span className="text-slate-500">{ep.detail}</span>
                  </li>
                ))}
              </ul>
            </Card>
          </section>

          {/* Limitations */}
          <section>
            <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-6">Known Limitations</h2>
            <Card className="p-6">
              <ul className="space-y-3 list-disc list-inside">
                {doc.limitations.map((l, i) => (
                  <li key={i} className="text-sm text-slate-400 leading-relaxed">{l}</li>
                ))}
              </ul>
            </Card>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  );
}
