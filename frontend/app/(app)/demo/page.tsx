"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { Send, Sparkles, Brain, Search, Zap, Gauge, MessageCircle, TriangleAlert } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { useStudentId } from "@/lib/student-id";
import type { AskResponse, MasteryEstimateOut } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const EXAMPLE_QUESTIONS = [
  "What is the role of genome sequencing in discovering new antibiotics?",
  "I'm confused about why microorganisms would produce antibiotics in the first place.",
  "hi",
];

const INTENT_STYLES: Record<string, string> = {
  question: "bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/30",
  misconception: "bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30",
  example_request: "bg-purple-500/15 text-purple-600 dark:text-purple-400 border-purple-500/30",
  off_topic: "bg-zinc-500/15 text-zinc-600 dark:text-zinc-400 border-zinc-500/30",
  unknown: "bg-zinc-500/15 text-zinc-600 dark:text-zinc-400 border-zinc-500/30",
};

export default function DemoPage() {
  const [message, setMessage] = useState("");
  const studentId = useStudentId();
  const mutation = useMutation({
    mutationFn: (msg: string) => api.ask(msg),
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Request failed"),
  });

  const submit = () => {
    if (!message.trim()) return;
    mutation.mutate(message);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Live Agent Demo</h1>
        <p className="text-muted-foreground">
          Ask a question the way a student would. The Pedagogical Guide, Knowledge Manager, and Knowledge Tracing
          agents all run against the real, running system, and the Tutor Response Orchestrator merges their output
          into one reply — expand &quot;How this answer was built&quot; below to see each agent&apos;s individual work.
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Part of the full personalized-learning loop: take a document&apos;s Diagnostic Check on Study → teach here
          → Self-Check on Knowledge Tracing — each step updates your mastery vector.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-3 pt-6">
          <Textarea
            placeholder="e.g. Why do microorganisms produce antibiotics?"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={3}
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => setMessage(q)}
                  className="rounded-full border px-3 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                >
                  {q.length > 40 ? q.slice(0, 40) + "…" : q}
                </button>
              ))}
            </div>
            <Button onClick={submit} disabled={mutation.isPending || !message.trim()}>
              <Send className="size-4" />
              {mutation.isPending ? "Asking…" : "Ask"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {mutation.isPending && <ResultsSkeleton />}
      {mutation.data && <Results data={mutation.data} />}
      {mutation.data && studentId && (
        <p className="text-xs text-muted-foreground">
          Demo student: <span className="font-mono">{studentId.slice(0, 8)}</span> — a local, per-browser identity
          (no login), used only to key Knowledge Tracing&apos;s mastery estimates.
        </p>
      )}
      {!mutation.data && !mutation.isPending && (
        <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed py-16 text-center text-muted-foreground">
          <Sparkles className="size-8" />
          <p>Ask a question above, or pick an example, to see both agents respond.</p>
        </div>
      )}
    </div>
  );
}

function ResultsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Skeleton className="h-48" />
      <Skeleton className="h-48" />
    </div>
  );
}

function Results({ data }: { data: AskResponse }) {
  const { classification, retrieval, tutor_response } = data;
  return (
    <div className="space-y-6">
    <TutorResponseCard response={tutor_response} />

    <details className="group rounded-lg border">
      <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
        How this answer was built — individual agent output
      </summary>
      <div className="space-y-6 border-t p-4">
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="size-4" /> Pedagogical Guide
          </CardTitle>
          <CardDescription>Intent classification, tiered keyword → LLM escalation.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className={cn("capitalize", INTENT_STYLES[classification.intent])}>
              {classification.intent.replace("_", " ")}
            </Badge>
            {classification.escalated_to_llm && (
              <Badge variant="outline" className="gap-1 border-amber-500/30 bg-amber-500/15 text-amber-600 dark:text-amber-400">
                <Zap className="size-3" /> Escalated to LLM
              </Badge>
            )}
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Confidence</span>
              <span>{Math.round(classification.confidence * 100)}%</span>
            </div>
            <Progress value={classification.confidence * 100} />
          </div>

          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">Concepts tagged against the KG</p>
            {classification.detected_concepts.length === 0 ? (
              <p className="text-sm text-muted-foreground italic">No known concepts matched.</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {classification.detected_concepts.map((c) => (
                  <Badge key={c.id} variant="secondary">
                    {c.canonical_name}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="size-4" /> Knowledge Manager
          </CardTitle>
          <CardDescription>Hybrid dense + BM25 retrieval, fused, re-ranked by KG graph-distance.</CardDescription>
        </CardHeader>
        <CardContent>
          {retrieval.results.length === 0 ? (
            <p className="text-sm text-muted-foreground italic">No results.</p>
          ) : (
            <ol className="space-y-3">
              {retrieval.results.map((r, i) => (
                <li key={r.chunk_id} className="rounded-md border p-3">
                  <div className="mb-1.5 flex items-center justify-between text-xs text-muted-foreground">
                    <span>#{i + 1}</span>
                    <span className="flex gap-3">
                      <span>fused {r.fused_score.toFixed(2)}</span>
                      <span>graph {r.graph_score.toFixed(2)}</span>
                      <span className="font-medium text-foreground">score {r.final_score.toFixed(2)}</span>
                    </span>
                  </div>
                  <p className="line-clamp-3 text-sm">{r.text}</p>
                </li>
              ))}
            </ol>
          )}
        </CardContent>
      </Card>
    </div>

    {classification.updated_mastery.length > 0 && <MasteryCard estimates={classification.updated_mastery} />}
      </div>
    </details>
    </div>
  );
}

function TutorResponseCard({ response }: { response: AskResponse["tutor_response"] }) {
  return (
    <Card className="border-primary/30 bg-primary/[0.03]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageCircle className="size-4 text-primary" /> Tutor Response
        </CardTitle>
        <CardDescription>
          The Tutor Response Orchestrator&apos;s synthesis of the Pedagogical Guide&apos;s intent, the Knowledge
          Manager&apos;s retrieved evidence, and the student&apos;s current mastery — one coherent, Socratic-scaffolded reply.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{response.reply}</p>
        {!response.guardrail_passed && response.guardrail_note && (
          <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-2.5 text-xs text-amber-700 dark:text-amber-400">
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
            <span>
              <strong>Guardrail flagged this reply:</strong> {response.guardrail_note}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function MasteryCard({ estimates }: { estimates: MasteryEstimateOut[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Gauge className="size-4" /> Student Mastery
        </CardTitle>
        <CardDescription>
          Knowledge Tracing (BKT + DKT ensemble) — updated live from this turn&apos;s classified intent.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {estimates.map((m) => (
          <div key={m.concept_id} className="space-y-1.5 rounded-md border p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">{m.concept_name}</span>
              <span className="text-xs text-muted-foreground">
                {Math.round(m.ensemble_mastery * 100)}%{m.dkt_mastery === null ? " (BKT only)" : " (BKT+DKT)"}
              </span>
            </div>
            <Progress value={m.ensemble_mastery * 100} />
            <p className="text-xs text-muted-foreground">{m.explanation}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
