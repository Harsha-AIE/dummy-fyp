"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { FileText, ClipboardCheck, GitFork, BookOpen } from "lucide-react";
import { api } from "@/lib/api";
import type { AvailableDocumentOut } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorState } from "@/components/error-state";

export default function StudyPage() {
  const { data, isLoading, isError, error } = useQuery({ queryKey: ["pipeline-documents"], queryFn: api.availableDocuments });

  const ready = (data ?? []).filter((d): d is AvailableDocumentOut & { document_id: string } => d.document_id !== null);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Study</h1>
        <p className="text-muted-foreground">
          Pick a document to take its Diagnostic Check or explore its Mind Map — your answers update your mastery on
          the Knowledge Tracing page.
        </p>
      </div>

      {isLoading && <Skeleton className="h-64 w-full" />}
      {isError && <ErrorState error={error} label="Couldn't load documents." />}
      {data && ready.length === 0 && !isLoading && !isError && (
        <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed py-16 text-center text-muted-foreground">
          <BookOpen className="size-8" />
          <p>No documents are ready yet — ask your teacher to upload and run one on Pipeline Runs.</p>
        </div>
      )}

      {ready.length > 0 && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {ready.map((doc) => (
            <Card key={doc.document_id} className="flex flex-col justify-between">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <FileText className="size-4 shrink-0 text-muted-foreground" />
                  {doc.name}
                </CardTitle>
                <CardDescription>Ready to study.</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-wrap gap-2">
                <Link href={`/diagnostic/${doc.document_id}`} className={buttonVariants({ size: "sm" })}>
                  <ClipboardCheck className="size-4" /> Diagnostic Check
                </Link>
                <Link href={`/mindmap/${doc.document_id}`} className={buttonVariants({ size: "sm", variant: "outline" })}>
                  <GitFork className="size-4" /> Mind Map
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
