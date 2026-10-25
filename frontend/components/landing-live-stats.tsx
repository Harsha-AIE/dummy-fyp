"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function LandingLiveStats() {
  const { data, isLoading, isError } = useQuery({ queryKey: ["dashboard-summary"], queryFn: api.dashboardSummary });
  // A marketing stat row isn't the place for a full error banner — but
  // isError must still be checked explicitly (not just `data?.x ?? "–"`),
  // otherwise a failed fetch and "still loading" are indistinguishable and
  // the row hangs on "–" forever with no way to tell the two apart in code.
  const unavailable = isLoading || isError;

  const stats = [
    { value: unavailable ? "–" : (data?.entity_count ?? "–"), label: "Entities extracted and linked into the live graph." },
    { value: unavailable ? "–" : (data?.accepted_triple_count ?? "–"), label: "Relationships validated and written to Neo4j." },
    { value: "3", label: "Agents reasoning live over the graph right now." },
  ];

  return (
    <div className="divide-y divide-border">
      {stats.map((stat) => (
        <div key={stat.label} className="grid grid-cols-[auto_1fr] items-center gap-6 py-6">
          <span className="font-heading text-5xl font-bold tracking-tight sm:text-6xl">{stat.value}</span>
          <span className="text-sm text-muted-foreground sm:text-base">{stat.label}</span>
        </div>
      ))}
    </div>
  );
}
