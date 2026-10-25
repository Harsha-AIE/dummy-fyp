"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Search, Network } from "lucide-react";
import { api } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorState } from "@/components/error-state";
import { GraphView, type GraphNode } from "@/components/graph-view";

export default function GraphPage() {
  // Longer interval than the app default (8s): every refetch feeds a new
  // object into the force-directed layout, which restarts the simulation —
  // fine occasionally, distracting if it happens every few seconds while
  // someone's trying to read the graph.
  const { data, isLoading, isError, error } = useQuery({ queryKey: ["graph"], queryFn: api.graph, refetchInterval: 25_000 });
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<GraphNode | null>(null);

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-3xl font-semibold tracking-tight">Knowledge Graph</h1>
          <p className="text-muted-foreground">
            {data ? `${data.nodes.length} entities, ${data.edges.length} relationships` : "Loading the live graph…"}
            {data && data.edges.some((e) => e.inferred) && " — dashed edges are transitively inferred, not from a document"}
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
                <ErrorState error={error} label="Couldn't load the graph." />
              </div>
            )}
            {data && data.nodes.length === 0 && (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
                <Network className="size-8" />
                <p>No entities yet — run the pipeline first.</p>
              </div>
            )}
            {data && data.nodes.length > 0 && <GraphView data={data} highlightQuery={query} onSelectNode={setSelected} />}
          </CardContent>
        </Card>

        <Card className="overflow-y-auto">
          <CardHeader>
            <CardTitle className="text-base">{selected ? "Entity" : "Select a node"}</CardTitle>
          </CardHeader>
          <CardContent>
            {selected ? (
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-muted-foreground">Canonical name</p>
                  <p className="font-medium">{selected.name}</p>
                </div>
                <div>
                  <p className="mb-1 text-xs text-muted-foreground">Type</p>
                  <Badge variant="secondary" className="capitalize">
                    {selected.type}
                  </Badge>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Entity ID</p>
                  <p className="font-mono text-xs break-all">{selected.id}</p>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Click a node in the graph to inspect it here.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
