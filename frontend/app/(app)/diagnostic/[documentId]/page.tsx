"use client";

import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Loader2, PartyPopper } from "lucide-react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { ErrorState } from "@/components/error-state";
import { QuizCard } from "@/components/quiz-card";

export default function DiagnosticPage() {
  const params = useParams<{ documentId: string }>();
  const documentId = params.documentId;

  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);

  const diagnostic = useMutation({
    mutationFn: () => api.quizDiagnostic(documentId),
  });

  useEffect(() => {
    diagnostic.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId]);

  const questions = diagnostic.data?.questions ?? [];
  const current = questions[index];
  const finished = diagnostic.isSuccess && questions.length > 0 && index >= questions.length;

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Diagnostic Check</h1>
        <p className="text-muted-foreground">
          A few quick questions on this document&apos;s main concepts before you start studying — this sets your
          starting mastery, so teaching can scaffold to what you already know instead of starting from zero.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {diagnostic.isPending && "Preparing your diagnostic…"}
            {!diagnostic.isPending && !finished && current && `Question ${index + 1} of ${questions.length}`}
            {finished && "Diagnostic complete"}
          </CardTitle>
          {!diagnostic.isPending && !diagnostic.isError && questions.length === 0 && (
            <CardDescription>No concepts were found for this document yet.</CardDescription>
          )}
        </CardHeader>
        <CardContent>
          {diagnostic.isPending && (
            <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" /> Generating grounded questions from this document…
            </div>
          )}
          {diagnostic.isError && <ErrorState error={diagnostic.error} label="Couldn't prepare the diagnostic check." />}
          {!diagnostic.isPending && !diagnostic.isError && questions.length === 0 && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              This document doesn&apos;t have enough linked concepts yet for a diagnostic — try a longer document, or
              head straight to the Live Agent Demo.
            </p>
          )}

          {current && (
            <QuizCard
              key={current.question_id}
              initialQuestion={current}
              onGraded={(correct) => setScore((s) => s + (correct ? 1 : 0))}
              onContinue={() => setIndex((i) => i + 1)}
              continueLabel={index + 1 < questions.length ? "Next question" : "See results"}
            />
          )}

          {finished && (
            <div className="flex flex-col items-center gap-3 py-8 text-center">
              <PartyPopper className="size-8 text-primary" />
              <p className="text-lg font-medium">
                You got {score} of {questions.length}
              </p>
              <p className="text-sm text-muted-foreground">
                That&apos;s your starting point — mastery for these concepts has already been recorded. Head to
                Knowledge Tracing to see your mastery vector and recommended next topics.
              </p>
              <Link href="/mastery" className={buttonVariants({})}>
                Go to Knowledge Tracing
              </Link>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
