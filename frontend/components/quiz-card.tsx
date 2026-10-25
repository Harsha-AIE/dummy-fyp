"use client";

import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { CheckCircle2, XCircle, Loader2, Sparkles, ArrowRight } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import type { QuizQuestionOut } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface QuizCardProps {
  /** Self-Check flow (mastery page): give a concept, QuizCard generates its
   * own question via POST /quiz/question on demand. */
  conceptId?: string;
  conceptName?: string;
  /** Diagnostic flow (Tier 3): the question was already generated as part
   * of a batch (POST /quiz/diagnostic) — pass it directly rather than
   * having QuizCard generate (and pay for) a second one. Takes precedence
   * over conceptId/autoStart when provided. */
  initialQuestion?: QuizQuestionOut;
  /** Self-Check starts idle behind a "Start Self-Check" button, so browsing
   * Recommended Next Topics never fires an LLM call nobody asked for;
   * ignored when initialQuestion is set. */
  autoStart?: boolean;
  /** Fires exactly once, right when grading completes — for a parent
   * tallying a running score across several QuizCards (the diagnostic
   * flow). Not required for a standalone self-check. */
  onGraded?: (correct: boolean) => void;
  /** When provided, a "Continue" button appears after grading — the
   * diagnostic flow uses this to advance to the next concept once the
   * student has read the explanation, rather than auto-advancing. */
  onContinue?: () => void;
  continueLabel?: string;
}

export function QuizCard({
  conceptId,
  conceptName,
  initialQuestion,
  autoStart = false,
  onGraded,
  onContinue,
  continueLabel = "Next question",
}: QuizCardProps) {
  const [question, setQuestion] = useState<QuizQuestionOut | null>(initialQuestion ?? null);
  const [selected, setSelected] = useState<number | null>(null);
  const [result, setResult] = useState<{ correct: boolean; correctIndex: number; explanation: string; masteryPct: number | null } | null>(null);

  const startQuiz = useMutation({
    mutationFn: () => {
      if (!conceptId) throw new Error("QuizCard: conceptId is required when initialQuestion is not provided");
      return api.quizQuestion(conceptId);
    },
    onSuccess: (q) => {
      setQuestion(q);
      setSelected(null);
      setResult(null);
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Couldn't generate a question"),
  });

  const submit = useMutation({
    mutationFn: (selectedIndex: number) => {
      if (!question) throw new Error("no active question");
      return api.quizSubmit(question.question_id, selectedIndex);
    },
    onSuccess: (res) => {
      setResult({
        correct: res.correct,
        correctIndex: res.correct_index,
        explanation: res.explanation,
        masteryPct: res.updated_mastery ? Math.round(res.updated_mastery.ensemble_mastery * 100) : null,
      });
      onGraded?.(res.correct);
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Couldn't submit your answer"),
  });

  useEffect(() => {
    if (!initialQuestion && autoStart) startQuiz.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conceptId, initialQuestion]);

  if (!initialQuestion && !question && !startQuiz.isPending && !autoStart) {
    return (
      <Button size="sm" variant="outline" onClick={() => startQuiz.mutate()}>
        <Sparkles className="size-3.5" /> Self-Check: {conceptName}
      </Button>
    );
  }

  if (!initialQuestion && (startQuiz.isPending || !question)) {
    return (
      <div className="flex items-center gap-2 rounded-md border p-3 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Generating a question grounded in {conceptName}…
      </div>
    );
  }

  if (!question) return null;

  return (
    <div className="space-y-3 rounded-md border p-3">
      <p className="text-sm font-medium">{question.question_text}</p>
      <div className="space-y-1.5">
        {question.options.map((option, index) => {
          const isSelected = selected === index;
          const isCorrectOption = result && index === result.correctIndex;
          const isWrongSelection = result && isSelected && !result.correct;
          return (
            <button
              key={index}
              type="button"
              disabled={!!result}
              onClick={() => setSelected(index)}
              className={cn(
                "flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-sm transition-colors",
                !result && isSelected && "border-primary bg-primary/5",
                !result && !isSelected && "hover:bg-accent",
                isCorrectOption && "border-emerald-500/50 bg-emerald-500/10",
                isWrongSelection && "border-destructive/50 bg-destructive/10",
              )}
            >
              {result && isCorrectOption && <CheckCircle2 className="size-4 shrink-0 text-emerald-500" />}
              {result && isWrongSelection && <XCircle className="size-4 shrink-0 text-destructive" />}
              <span>{option}</span>
            </button>
          );
        })}
      </div>

      {!result && (
        <Button size="sm" disabled={selected === null || submit.isPending} onClick={() => selected !== null && submit.mutate(selected)}>
          {submit.isPending ? <Loader2 className="size-3.5 animate-spin" /> : null}
          Submit Answer
        </Button>
      )}

      {result && (
        <div className="space-y-2 border-t pt-2">
          <p className={cn("flex items-center gap-1.5 text-sm font-medium", result.correct ? "text-emerald-600 dark:text-emerald-400" : "text-destructive")}>
            {result.correct ? <CheckCircle2 className="size-4" /> : <XCircle className="size-4" />}
            {result.correct ? "Correct!" : "Not quite."}
            {result.masteryPct !== null && <span className="font-normal text-muted-foreground">— mastery now {result.masteryPct}%</span>}
          </p>
          <p className="text-sm text-muted-foreground">{result.explanation}</p>
          {onContinue && (
            <Button size="sm" variant="outline" onClick={onContinue}>
              {continueLabel} <ArrowRight className="size-3.5" />
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
