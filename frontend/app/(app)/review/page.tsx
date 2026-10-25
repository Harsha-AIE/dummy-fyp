"use client";

import { useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, CheckCheck, ChevronLeft, ChevronRight, X, XCircle } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { useRole } from "@/lib/role";
import type { PendingExtensionOut, PendingTripleOut } from "@/lib/types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { ErrorState } from "@/components/error-state";
import { TeacherOnlyNotice } from "@/components/teacher-only";

const TRIPLES_PAGE_SIZE = 50;

export default function ReviewPage() {
  const role = useRole();
  if (role !== "teacher") return <TeacherOnlyNotice />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Review Console</h1>
        <p className="text-muted-foreground">Human-in-the-loop approval for what the pipeline proposed but wasn&apos;t confident enough to auto-accept.</p>
      </div>

      <Tabs defaultValue="triples">
        <TabsList>
          <TabsTrigger value="triples">Pending Triples</TabsTrigger>
          <TabsTrigger value="extensions">Ontology Extensions</TabsTrigger>
        </TabsList>
        <TabsContent value="triples" className="mt-4">
          <TriplesTable />
        </TabsContent>
        <TabsContent value="extensions" className="mt-4">
          <ExtensionsTable />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function TriplesTable() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [threshold, setThreshold] = useState(70);
  const { data, isLoading, isError, error, isPlaceholderData } = useQuery({
    queryKey: ["pending-triples", page],
    queryFn: () => api.pendingTriples(page, TRIPLES_PAGE_SIZE),
    placeholderData: keepPreviousData,
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["pending-triples"] });
    queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    queryClient.invalidateQueries({ queryKey: ["graph"] });
  };

  const approve = useMutation({
    mutationFn: (id: string) => api.approveTriple(id),
    onSuccess: () => {
      toast.success("Triple approved — written to the graph");
      invalidate();
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Approve failed"),
  });
  const reject = useMutation({
    mutationFn: (id: string) => api.rejectTriple(id),
    onSuccess: () => {
      toast("Triple rejected");
      invalidate();
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Reject failed"),
  });
  const bulkApprove = useMutation({
    mutationFn: (minConfidence: number) => api.bulkApproveTriples(minConfidence),
    onSuccess: (result) => {
      toast.success(`${result.count.toLocaleString()} triple(s) approved — written to the graph`);
      setPage(1);
      invalidate();
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Bulk approve failed"),
  });
  const bulkReject = useMutation({
    mutationFn: (maxConfidence: number) => api.bulkRejectTriples(maxConfidence),
    onSuccess: (result) => {
      toast(`${result.count.toLocaleString()} triple(s) rejected`);
      setPage(1);
      invalidate();
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Bulk reject failed"),
  });
  const bulkPending = bulkApprove.isPending || bulkReject.isPending;

  const bulkControls = (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-muted/30 p-3 text-sm">
      <span className="text-muted-foreground">Bulk action — confidence threshold</span>
      <Input
        type="number"
        min={0}
        max={100}
        value={threshold}
        onChange={(e) => setThreshold(Math.min(100, Math.max(0, Number(e.target.value))))}
        className="h-8 w-20"
      />
      <span className="text-muted-foreground">%</span>
      <Button
        size="sm"
        variant="outline"
        disabled={bulkPending}
        onClick={() => {
          if (
            window.confirm(
              `Approve every pending triple at or above ${threshold}% confidence? This affects the entire queue, not just this page, and cannot be undone.`,
            )
          ) {
            bulkApprove.mutate(threshold / 100);
          }
        }}
      >
        <CheckCheck className="size-3.5" /> Bulk Approve ≥ {threshold}%
      </Button>
      <Button
        size="sm"
        variant="outline"
        disabled={bulkPending}
        onClick={() => {
          if (
            window.confirm(
              `Reject every pending triple at or below ${threshold}% confidence? This affects the entire queue, not just this page, and cannot be undone.`,
            )
          ) {
            bulkReject.mutate(threshold / 100);
          }
        }}
      >
        <XCircle className="size-3.5" /> Bulk Reject ≤ {threshold}%
      </Button>
    </div>
  );

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError) return <ErrorState error={error} label="Couldn't load pending triples." />;
  if (!data || data.total === 0) return <EmptyState label="No triples pending review." />;

  const lastPage = Math.max(1, Math.ceil(data.total / data.page_size));

  return (
    <div className="space-y-3">
      {bulkControls}

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          {data.total.toLocaleString()} pending triples, sorted by confidence — showing page {data.page} of {lastPage}
        </span>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1 || isPlaceholderData}
          >
            <ChevronLeft className="size-3.5" /> Prev
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setPage((p) => Math.min(lastPage, p + 1))}
            disabled={page >= lastPage || isPlaceholderData}
          >
            Next <ChevronRight className="size-3.5" />
          </Button>
        </div>
      </div>

      <Table className={isPlaceholderData ? "opacity-60 transition-opacity" : "transition-opacity"}>
        <TableHeader>
          <TableRow>
            <TableHead>Subject</TableHead>
            <TableHead>Predicate</TableHead>
            <TableHead>Object</TableHead>
            <TableHead>Confidence</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.items.map((t: PendingTripleOut) => (
            <TableRow key={t.id}>
              <TableCell className="font-medium">{t.subject_name}</TableCell>
              <TableCell>
                <Badge variant="outline">{t.predicate}</Badge>
              </TableCell>
              <TableCell>{t.object_name}</TableCell>
              <TableCell>{Math.round(t.confidence * 100)}%</TableCell>
              <TableCell>
                <Badge variant={t.validation_status === "invalid" ? "destructive" : "secondary"} className="capitalize">
                  {t.validation_status}
                </Badge>
              </TableCell>
              <TableCell className="text-right">
                <RowActions
                  onApprove={() => approve.mutate(t.id)}
                  onReject={() => reject.mutate(t.id)}
                  pending={approve.isPending || reject.isPending}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function ExtensionsTable() {
  const queryClient = useQueryClient();
  const { data, isLoading, isError, error } = useQuery({ queryKey: ["pending-extensions"], queryFn: api.pendingExtensions });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["pending-extensions"] });
    queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
  };

  const approve = useMutation({
    mutationFn: (id: string) => api.approveExtension(id),
    onSuccess: () => {
      toast.success("Extension approved — merged into the ontology");
      invalidate();
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Approve failed"),
  });
  const reject = useMutation({
    mutationFn: (id: string) => api.rejectExtension(id),
    onSuccess: () => {
      toast("Extension rejected");
      invalidate();
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Reject failed"),
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError) return <ErrorState error={error} label="Couldn't load pending extensions." />;
  if (!data || data.length === 0) return <EmptyState label="No ontology extensions pending review." />;

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Proposed name</TableHead>
          <TableHead>Kind</TableHead>
          <TableHead>Subject</TableHead>
          <TableHead>Justification</TableHead>
          <TableHead>Confidence</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.map((e: PendingExtensionOut) => (
          <TableRow key={e.id}>
            <TableCell className="font-medium">{e.proposed_name}</TableCell>
            <TableCell>
              <Badge variant="outline" className="capitalize">
                {e.kind}
              </Badge>
            </TableCell>
            <TableCell className="capitalize">{e.subject.replace("_", " ")}</TableCell>
            <TableCell className="max-w-xs truncate text-sm text-muted-foreground" title={e.justification}>
              {e.justification}
            </TableCell>
            <TableCell>{Math.round(e.confidence * 100)}%</TableCell>
            <TableCell className="text-right">
              <RowActions
                onApprove={() => approve.mutate(e.id)}
                onReject={() => reject.mutate(e.id)}
                pending={approve.isPending || reject.isPending}
              />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function RowActions({ onApprove, onReject, pending }: { onApprove: () => void; onReject: () => void; pending: boolean }) {
  return (
    <div className="flex justify-end gap-2">
      <Button size="sm" variant="outline" onClick={onApprove} disabled={pending}>
        <Check className="size-3.5" /> Approve
      </Button>
      <Button size="sm" variant="ghost" onClick={onReject} disabled={pending}>
        <X className="size-3.5" /> Reject
      </Button>
    </div>
  );
}

function EmptyState({ label }: { label: string }) {
  return <div className="rounded-lg border border-dashed py-16 text-center text-muted-foreground">{label}</div>;
}
