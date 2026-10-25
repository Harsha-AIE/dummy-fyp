"use client";

import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { Gauge, Target, GitBranch, Sparkles, Eye } from "lucide-react";
import { api } from "@/lib/api";
import { useStudentId } from "@/lib/student-id";
import { useRole } from "@/lib/role";
import type { KTEventOut, TopicRecommendationOut } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorState } from "@/components/error-state";
import { QuizCard } from "@/components/quiz-card";

export default function MasteryPage() {
  const ownStudentId = useStudentId();
  const role = useRole();
  const searchParams = useSearchParams();
  // The ?student= override is a Teacher-only capability (reached via Class
  // Overview) — gated by role here too, not just by hiding the link,
  // because this is a plain query param anyone could type into the address
  // bar. Without this check, a Student could view any other student's
  // mastery just by guessing/copying an id, which was never an intended
  // capability (unlike the backend's own lack of auth, which is an
  // accepted, explicit project-wide scope decision, this is a feature
  // that was specifically pitched as teacher-only and must behave that way
  // for every role that isn't one).
  const viewedStudentId = role === "teacher" ? searchParams.get("student") : null;
  const viewedName = searchParams.get("name");
  // A teacher viewing another student's page via Class Overview — Self-
  // Check stays hidden in that mode below, since QuizCard always grades
  // against the CURRENT browser's own identity (getStudentId()), not this
  // page's studentId prop; showing it here would misleadingly suggest a
  // teacher's answer updates the student's mastery.
  const isViewingOther = viewedStudentId !== null && viewedStudentId !== ownStudentId;
  const studentId = viewedStudentId ?? ownStudentId;

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["mastery", studentId],
    queryFn: () => api.masteryVector(studentId as string),
    enabled: studentId !== null,
  });

  // Built from this same response, not a second request — every concept
  // with an event also has a mastery estimate (both are written together
  // by KnowledgeTracingAgent.record_dialogue_turn), so there's no case
  // where the event log needs a name this map doesn't already have.
  const conceptNames = new Map(data?.estimates.map((m) => [m.concept_id, m.concept_name]) ?? []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Knowledge Tracing</h1>
        <p className="text-muted-foreground">
          Per-concept mastery for this browser&apos;s demo student, fused from a Bayesian Knowledge Tracing (BKT)
          estimate and a Deep Knowledge Tracing (DKT) neural model. Built from the Live Agent Demo — ask questions
          there to populate this.
        </p>
        {studentId && !isViewingOther && (
          <p className="mt-1 text-xs text-muted-foreground">
            Demo student: <span className="font-mono">{studentId.slice(0, 8)}</span>
          </p>
        )}
        {isViewingOther && (
          <p className="mt-1 flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400">
            <Eye className="size-3.5" /> Viewing {viewedName ?? "this student"}&apos;s mastery as a teacher
          </p>
        )}
      </div>

      {isLoading && <Skeleton className="h-64 w-full" />}
      {isError && <ErrorState error={error} label="Couldn't load mastery data." />}
      {data && data.estimates.length === 0 && (
        <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed py-16 text-center text-muted-foreground">
          <Gauge className="size-8" />
          <p>No mastery data yet — ask a few questions on the Live Agent Demo first.</p>
        </div>
      )}
      {data && data.estimates.length > 0 && studentId && (
        <>
          <ProgressSummary estimates={data.estimates} />
          {!isViewingOther && <RecommendedTopics studentId={studentId} />}
          <Tabs defaultValue="mastery">
          <TabsList>
            <TabsTrigger value="mastery">Mastery Vector</TabsTrigger>
            <TabsTrigger value="events">Event Log</TabsTrigger>
          </TabsList>
          <TabsContent value="mastery" className="mt-4">
            <Card>
              <CardContent className="pt-6">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Concept</TableHead>
                      <TableHead>Mastery</TableHead>
                      <TableHead>BKT</TableHead>
                      <TableHead>DKT</TableHead>
                      <TableHead>Last updated</TableHead>
                      <TableHead>Explanation</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.estimates.map((m) => (
                      <TableRow key={m.concept_id}>
                        <TableCell className="font-medium">{m.concept_name}</TableCell>
                        <TableCell className="w-32">
                          <div className="flex items-center gap-2">
                            <Progress value={m.ensemble_mastery * 100} className="w-16" />
                            <span className="text-xs text-muted-foreground">{Math.round(m.ensemble_mastery * 100)}%</span>
                          </div>
                        </TableCell>
                        <TableCell>{Math.round(m.bkt_mastery * 100)}%</TableCell>
                        <TableCell>{m.dkt_mastery === null ? "not trained yet" : `${Math.round(m.dkt_mastery * 100)}%`}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {new Date(m.updated_at).toLocaleString()}
                        </TableCell>
                        <TableCell className="max-w-xs truncate text-sm text-muted-foreground" title={m.explanation}>
                          {m.explanation}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="events" className="mt-4">
            <EventLog studentId={studentId} conceptNames={conceptNames} />
          </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}

function ProgressSummary({ estimates }: { estimates: { ensemble_mastery: number; concept_name: string }[] }) {
  const MASTERED_THRESHOLD = 0.7;
  const mastered = estimates.filter((e) => e.ensemble_mastery >= MASTERED_THRESHOLD);
  const strongest = estimates.reduce((best, e) => (e.ensemble_mastery > best.ensemble_mastery ? e : best));
  const weakest = estimates.reduce((worst, e) => (e.ensemble_mastery < worst.ensemble_mastery ? e : worst));

  return (
    <div className="flex items-start gap-2 rounded-lg border bg-muted/30 p-3 text-sm">
      <Sparkles className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
      <p>
        You&apos;ve mastered <span className="font-medium">{mastered.length}</span> of{" "}
        <span className="font-medium">{estimates.length}</span> concept{estimates.length === 1 ? "" : "s"} you&apos;ve
        engaged with. Strongest: <span className="font-medium">{strongest.concept_name}</span>. Weakest:{" "}
        <span className="font-medium">{weakest.concept_name}</span>.
      </p>
    </div>
  );
}

function RecommendedTopics({ studentId }: { studentId: string }) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["kt-recommendations", studentId],
    queryFn: () => api.recommendations(studentId),
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="size-4" /> Recommended Next Topics
        </CardTitle>
        <CardDescription>
          Combines your mastery scores with the knowledge graph&apos;s prerequisite structure — when a weak topic
          depends on another weak foundation, that foundation is recommended first.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-24 w-full" />}
        {isError && <ErrorState error={error} label="Couldn't load recommendations." />}
        {data && data.recommendations.length === 0 && (
          <p className="text-sm text-muted-foreground">No gaps found — you&apos;re not behind on any prerequisite right now.</p>
        )}
        {data && data.recommendations.length > 0 && (
          <div className="space-y-2">
            {data.recommendations.map((r: TopicRecommendationOut) => (
              <div key={r.concept_id} className="space-y-2 rounded-md border p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{r.concept_name}</span>
                      {r.reason === "foundational" ? (
                        <Badge
                          variant="outline"
                          className="gap-1 border-amber-500/30 bg-amber-500/15 text-amber-600 dark:text-amber-400"
                        >
                          <GitBranch className="size-3" /> Foundational
                        </Badge>
                      ) : (
                        <Badge
                          variant="outline"
                          className="gap-1 border-emerald-500/30 bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                        >
                          Ready to study
                        </Badge>
                      )}
                    </div>
                    {r.unblocks.length > 0 && (
                      <p className="text-xs text-muted-foreground">Unblocks: {r.unblocks.join(", ")}</p>
                    )}
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">{Math.round(r.mastery * 100)}% mastery</span>
                </div>
                <QuizCard conceptId={r.concept_id} conceptName={r.concept_name} />
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function EventLog({ studentId, conceptNames }: { studentId: string; conceptNames: Map<string, string> }) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["kt-events", studentId],
    queryFn: () => api.ktEvents(studentId),
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError) return <ErrorState error={error} label="Couldn't load the event log." />;
  if (!data || data.length === 0) {
    return (
      <div className="rounded-lg border border-dashed py-16 text-center text-sm text-muted-foreground">
        No events recorded yet.
      </div>
    );
  }

  // Most recent first — the question this tab answers is "why did my
  // mastery just change", so the newest evidence belongs on top.
  const sorted = [...data].sort((a, b) => new Date(b.occurred_at).getTime() - new Date(a.occurred_at).getTime());

  return (
    <Card>
      <CardContent className="pt-6">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Concept</TableHead>
              <TableHead>Source</TableHead>
              <TableHead>Evidence</TableHead>
              <TableHead>Weight</TableHead>
              <TableHead>When</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((e: KTEventOut) => (
              <TableRow key={e.id}>
                <TableCell className="font-medium">{conceptNames.get(e.concept_id) ?? e.concept_id.slice(0, 8)}</TableCell>
                <TableCell className="capitalize text-sm text-muted-foreground">{e.source}</TableCell>
                <TableCell>
                  <Badge variant={e.correct ? "secondary" : "destructive"}>{e.correct ? "Correct" : "Incorrect"}</Badge>
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">{e.weight.toFixed(2)}</TableCell>
                <TableCell className="text-xs text-muted-foreground">{new Date(e.occurred_at).toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
