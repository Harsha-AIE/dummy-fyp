import Link from "next/link";
import { ArrowUpRight, FileStack, Sparkles, ShieldCheck, Bot, Search, Brain } from "lucide-react";
import { LandingNav } from "@/components/landing-nav";
import { LandingLiveStats } from "@/components/landing-live-stats";

const STEPS = [
  { icon: FileStack, title: "Ingest", description: "PDF/DOCX documents chunked and OCR'd where needed." },
  { icon: Sparkles, title: "Extract & Link", description: "Entities and relations extracted, resolved against the live graph." },
  { icon: ShieldCheck, title: "Validate & Fuse", description: "Domain/range checked, confidence-fused, weak signal queued for review." },
  { icon: Bot, title: "Agents Reason", description: "Retrieval and intent classification ground themselves in the graph." },
];

const AGENTS = [
  {
    icon: Search,
    name: "Knowledge Manager",
    tagline: "Retrieval-Augmented Generation",
    description:
      "Hybrid dense + BM25 retrieval, fused with Reciprocal Rank Fusion, then re-ranked by how close each result's concepts sit to your question's concepts in the knowledge graph.",
  },
  {
    icon: Brain,
    name: "Pedagogical Guide",
    tagline: "Intent & misconception classification",
    description:
      "A fast keyword-based first pass escalates to an LLM only when it isn't confident — then tags every message with the real concepts it touches, pulled from the live graph.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      <LandingNav />

      {/* Hero — two column, matching the reference's headline-left / proof-right rhythm */}
      <section className="mx-auto max-w-6xl px-6 pt-20 pb-16">
        <div className="grid grid-cols-1 gap-16 lg:grid-cols-2 lg:items-center">
          <div>
            <h1 className="font-heading text-5xl leading-[1.05] font-bold tracking-tight sm:text-6xl">
              Documents
              <br />
              become graphs.
              <br />
              <span className="text-primary">Agents stay grounded.</span>
            </h1>
            <p className="mt-6 max-w-md text-lg text-muted-foreground">
              eduKG ingests real course material, builds a validated Neo4j knowledge graph, and gives two live agents the
              ability to reason over it — not guess.
            </p>
            <Link
              href="/graph"
              className="mt-8 inline-flex items-center gap-1.5 text-sm font-medium text-foreground transition-opacity hover:opacity-70"
            >
              Explore the live graph
              <span className="flex size-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <ArrowUpRight className="size-3.5" />
              </span>
            </Link>
          </div>

          <LandingLiveStats />
        </div>
      </section>

      {/* How it works */}
      <section className="border-t border-border px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <div className="max-w-lg">
            <h2 className="font-heading text-3xl font-bold tracking-tight">How it works</h2>
            <p className="mt-2 text-muted-foreground">A 12-stage pipeline turns raw documents into graph structure before any agent ever sees them.</p>
          </div>

          <div className="mt-14 grid grid-cols-1 divide-y divide-border sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-4">
            {STEPS.map((step, i) => (
              <div key={step.title} className="border-border py-6 sm:border-l sm:py-0 sm:pl-6 lg:py-0">
                <div className="mb-4 flex size-10 items-center justify-center rounded-full border border-primary/30 text-primary">
                  <step.icon className="size-4" />
                </div>
                <p className="mb-1 text-xs font-medium text-muted-foreground">0{i + 1}</p>
                <h3 className="font-heading text-lg font-semibold">{step.title}</h3>
                <p className="mt-1.5 text-sm text-muted-foreground">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Agents */}
      <section className="border-t border-border px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <div className="max-w-lg">
            <h2 className="font-heading text-3xl font-bold tracking-tight">The agents</h2>
            <p className="mt-2 text-muted-foreground">Two of the four agents in the architecture are live today — both reason directly over the graph above.</p>
          </div>

          <div className="mt-12 grid grid-cols-1 gap-px overflow-hidden rounded-2xl border md:grid-cols-2">
            {AGENTS.map((agent) => (
              <div key={agent.name} className="bg-card p-8 transition-colors hover:bg-accent/40">
                <div className="mb-5 flex size-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <agent.icon className="size-5" />
                </div>
                <h3 className="font-heading text-xl font-semibold">{agent.name}</h3>
                <p className="mb-3 text-xs font-medium tracking-wide text-primary uppercase">{agent.tagline}</p>
                <p className="text-sm text-muted-foreground">{agent.description}</p>
                <Link href="/demo" className="mt-5 inline-flex items-center gap-1 text-sm font-medium text-foreground hover:underline">
                  Try it <ArrowUpRight className="size-3.5" />
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-border px-6 py-8 text-center text-xs text-muted-foreground">
        FastAPI · Neo4j · Azure OpenAI · Next.js — built on a real, running system, not a mockup.
      </footer>
    </div>
  );
}
