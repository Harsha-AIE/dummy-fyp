"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { FileText, Play, CheckCircle2, UploadCloud, Loader2, ClipboardCheck, GitFork } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { useRole } from "@/lib/role";
import type { AvailableDocumentOut } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button, buttonVariants } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { PipelineStepper } from "@/components/pipeline-stepper";
import { usePipelineStream } from "@/hooks/use-pipeline-stream";
import { ErrorState } from "@/components/error-state";
import { TeacherOnlyNotice } from "@/components/teacher-only";
import { cn } from "@/lib/utils";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx"];

export default function PipelinePage() {
  const role = useRole();
  const queryClient = useQueryClient();
  const {
    data: documents,
    isLoading,
    isError: documentsError,
    error: documentsErrorDetail,
  } = useQuery({ queryKey: ["pipeline-documents"], queryFn: api.availableDocuments });
  const [selected, setSelected] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const stream = usePipelineStream();

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadDocument(file),
    onSuccess: (uploaded) => {
      toast.success(`Uploaded "${uploaded.name}"`);
      queryClient.invalidateQueries({ queryKey: ["pipeline-documents"] });
      setSelected(uploaded.name);
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Upload failed"),
  });

  const createRun = useMutation({
    mutationFn: (path: string) => api.createPipelineRun(path),
    onSuccess: (created) => stream.start(created.run_id),
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Failed to start pipeline run"),
  });

  const runDisabled = !selected || createRun.isPending || stream.isRunning;

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      toast.error(`Unsupported file type "${ext}" — expected .pdf or .docx`);
      return;
    }
    upload.mutate(file);
  };

  useEffect(() => {
    if (stream.finalStatus?.state === "completed") {
      toast.success("Pipeline run completed — search index refreshed");
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
      queryClient.invalidateQueries({ queryKey: ["graph"] });
      queryClient.invalidateQueries({ queryKey: ["pending-triples"] });
      queryClient.invalidateQueries({ queryKey: ["pending-extensions"] });
      queryClient.invalidateQueries({ queryKey: ["pipeline-documents"] });
    } else if (stream.finalStatus?.state === "failed") {
      toast.error(stream.finalStatus.error ?? "Pipeline run failed");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.finalStatus]);

  if (role !== "teacher") return <TeacherOnlyNotice />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Pipeline Runs</h1>
        <p className="text-muted-foreground">Upload a document, or pick one already on the server, then watch all 12 pipeline stages complete live.</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">1. Upload a document</CardTitle>
              <CardDescription>PDF or DOCX</CardDescription>
            </CardHeader>
            <CardContent>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_EXTENSIONS.join(",")}
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  handleFiles(e.dataTransfer.files);
                }}
                disabled={upload.isPending}
                className={cn(
                  "flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors",
                  isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-accent/40",
                )}
              >
                {upload.isPending ? (
                  <>
                    <Loader2 className="size-6 animate-spin text-primary" />
                    <span className="text-sm text-muted-foreground">Uploading…</span>
                  </>
                ) : (
                  <>
                    <UploadCloud className="size-6 text-muted-foreground" />
                    <span className="text-sm font-medium">Drop a file, or click to browse</span>
                    <span className="text-xs text-muted-foreground">.pdf or .docx</span>
                  </>
                )}
              </button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">2. Choose a document</CardTitle>
              <CardDescription>Uploaded, or already on the server</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {isLoading && <Skeleton className="h-24 w-full" />}
              {documentsError && <ErrorState error={documentsErrorDetail} label="Couldn't load documents." />}
              {documents?.length === 0 && !isLoading && !documentsError && (
                <p className="py-4 text-center text-sm text-muted-foreground">No documents yet — upload one above.</p>
              )}
              {documents?.map((doc: AvailableDocumentOut) => (
                <button
                  key={doc.name}
                  onClick={() => setSelected(doc.name)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-sm transition-colors",
                    selected === doc.name ? "border-primary bg-primary/5" : "hover:bg-accent",
                  )}
                >
                  <FileText className="size-4 shrink-0 text-muted-foreground" />
                  <span className="flex-1 truncate">{doc.name}</span>
                  {doc.already_ingested && (
                    <Badge variant="secondary" className="gap-1 text-xs">
                      <CheckCircle2 className="size-3" /> ingested
                    </Badge>
                  )}
                </button>
              ))}

              <Button className="mt-2 w-full" disabled={runDisabled} onClick={() => selected && createRun.mutate(selected)}>
                <Play className="size-4" />
                {stream.isRunning ? "Running…" : "Run Pipeline"}
              </Button>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">3. Live stage progress</CardTitle>
          </CardHeader>
          <CardContent>
            {!stream.runId ? (
              <p className="py-8 text-center text-sm text-muted-foreground">Pick a document and run the pipeline to see live progress here.</p>
            ) : (
              <>
                <PipelineStepper statuses={stream.stageStatuses} />
                {stream.streamError && (
                  <p className="mt-4 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm text-amber-600 dark:text-amber-400">
                    {stream.streamError}
                  </p>
                )}
                {stream.finalStatus && stream.finalStatus.state !== "running" && <RunResult status={stream.finalStatus} />}
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function RunResult({ status }: { status: NonNullable<ReturnType<typeof usePipelineStream>["finalStatus"]> }) {
  if (status.state === "failed") {
    return <p className="mt-4 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">{status.error}</p>;
  }
  if (!status.summary) return null;
  const s = status.summary;
  return (
    <div className="mt-4 space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {(
          [
            ["Chunks", s.chunk_count],
            ["Mentions", s.mention_count],
            ["Entities", s.entity_count],
            ["Relations", s.relation_count],
            ["Triples", s.triple_count],
          ] as const
        ).map(([label, value]) => (
          <div key={label} className="rounded-md border p-2 text-center">
            <p className="text-lg font-semibold">{value}</p>
            <p className="text-xs text-muted-foreground">{label}</p>
          </div>
        ))}
      </div>
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <CheckCircle2 className="size-3.5 text-emerald-500" />
        Search index refreshed — this document is now searchable from the Live Agent Demo.
      </p>
      <div className="flex flex-wrap gap-2 pt-1">
        <Link href={`/diagnostic/${s.document_id}`} className={buttonVariants({ size: "sm" })}>
          <ClipboardCheck className="size-4" /> Start Diagnostic Check
        </Link>
        <Link href={`/mindmap/${s.document_id}`} className={buttonVariants({ size: "sm", variant: "outline" })}>
          <GitFork className="size-4" /> View Mind Map
        </Link>
      </div>
    </div>
  );
}
