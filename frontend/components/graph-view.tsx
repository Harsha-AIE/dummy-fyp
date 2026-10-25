"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { GraphResponse } from "@/lib/types";

// react-force-graph-2d touches `window`/canvas at import time — must never
// run during Next.js server rendering, hence ssr:false plus this component
// itself only being used from a "use client" page.
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

// Saturated 500/600-weight hues — chosen for contrast against the light
// cream background, not the darker 400-weight tints a dark-mode graph would use.
const TYPE_COLORS: Record<string, string> = {
  concept: "#2563eb",
  person: "#db2777",
  organization: "#ea580c",
  formula: "#059669",
  algorithm: "#7c3aed",
  theorem: "#b45309",
  definition: "#0891b2",
  other: "#475569",
  // Mind Map's synthetic root node (routers/graph.py's get_document_mind_map)
  // — never present on the full /graph page, so this is additive, not a
  // visual change to the existing graph view.
  document: "#111827",
};
const HIGHLIGHT_COLOR = "#dc2626";
const ROOT_NODE_VAL = 12;
const DEFAULT_NODE_VAL = 1;

export interface GraphNode {
  id: string;
  name: string;
  type: string;
}

interface ForceLink {
  source: string;
  target: string;
  predicate: string;
  confidence: number;
  inferred: boolean;
}

export function GraphView({
  data,
  highlightQuery,
  onSelectNode,
  layout = "force",
}: {
  data: GraphResponse;
  highlightQuery: string;
  onSelectNode: (node: GraphNode | null) => void;
  /** "radial" is the Mind Map's layout (Tier 4) — react-force-graph-2d's
   * native dagMode="radialout", radiating out from whichever node(s) have
   * no incoming links (the synthetic document root). "force" (default) is
   * the plain force-directed layout the full Knowledge Graph page uses. */
  layout?: "force" | "radial";
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const graphData = useMemo(
    () => ({
      nodes: data.nodes.map((n) => ({ ...n })),
      links: data.edges.map((e) => ({ ...e })) as ForceLink[],
    }),
    [data],
  );

  const matchesQuery = (name: string) =>
    highlightQuery.trim().length > 0 && name.toLowerCase().includes(highlightQuery.trim().toLowerCase());

  return (
    <div ref={containerRef} className="h-full w-full">
      <ForceGraph2D
        graphData={graphData}
        width={size.width}
        height={size.height}
        dagMode={layout === "radial" ? "radialout" : undefined}
        dagLevelDistance={layout === "radial" ? 80 : undefined}
        nodeLabel={(node) => `${(node as GraphNode).name} (${(node as GraphNode).type})`}
        nodeColor={(node) => {
          const n = node as GraphNode;
          if (matchesQuery(n.name)) return HIGHLIGHT_COLOR;
          return TYPE_COLORS[n.type] ?? TYPE_COLORS.other;
        }}
        nodeVal={(node) => ((node as GraphNode).type === "document" ? ROOT_NODE_VAL : DEFAULT_NODE_VAL)}
        nodeRelSize={5}
        linkLabel={(link) => {
          const l = link as unknown as ForceLink;
          return l.inferred ? `${l.predicate} (inferred — transitive closure, not from a document)` : l.predicate;
        }}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkColor={() => "rgba(87, 66, 53, 0.35)"}
        linkWidth={(link) => 0.5 + (link as unknown as ForceLink).confidence * 2}
        linkLineDash={(link) => ((link as unknown as ForceLink).inferred ? [2, 2] : null)}
        onNodeClick={(node) => onSelectNode(node as GraphNode)}
        backgroundColor="transparent"
        cooldownTicks={100}
      />
    </div>
  );
}
