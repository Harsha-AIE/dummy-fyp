"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { useState } from "react";
import { Search, GitFork } from "lucide-react";
import { api } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorState } from "@/components/error-state";
import { GraphView, type GraphNode } from "@/components/graph-view";

export default function MindMapPage() {
  const params = useParams<{ documentId: string }>();
  const documentId = params.documentId;

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["document-graph", documentId],
    queryFn: () => api.documentGraph(documentId),
    refetchInterval: false,
  });
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<GraphNode | null>(null);

  const documentName = data?.nodes.find((n) => n.type === "document")?.name ?? "this document";

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-3xl font-semibold tracking-tight">Mind Map</h1>
          <p className="text-muted-foreground">
            {data ? `${documentName} — ${data.nodes.length - 1} key concepts` : "Loading this document's concept map…"}
          </p>
        </div>
        <div className="relative w-72">
          <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
          <Input placeholder="Search concepts…" className="pl-8" value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
        <Card className="min-h-0 overflow-hidden">
          <CardContent className="h-full p-0">
            {isLoading && <Skeleton className="h-full w-full" />}
            {isError && (
              <div className="flex h-full items-center justify-center p-6">
                <ErrorState error={error} label="Couldn't load the mind map." />
              </div>
            )}
            {data && data.nodes.length <= 1 && (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
                <GitFork className="size-8" />
                <p>No concepts were found for this document yet.</p>
              </div>
            )}
            {data && data.nodes.length > 1 && <GraphView data={data} highlightQuery={query} onSelectNode={setSelected} layout="radial" />}
          </CardContent>
        </Card>

        <Card className="overflow-y-auto">
          <CardHeader>
            <CardTitle className="text-base">{selected ? "Concept" : "Select a node"}</CardTitle>
          </CardHeader>
          <CardContent>
            {selected ? (
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-muted-foreground">Name</p>
                  <p className="font-medium">{selected.name}</p>
                </div>
                <div>
                  <p className="mb-1 text-xs text-muted-foreground">Type</p>
                  <Badge variant="secondary" className="capitalize">
                    {selected.type}
                  </Badge>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Click a node in the mind map to inspect it — the dark center node is {documentName}.
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
