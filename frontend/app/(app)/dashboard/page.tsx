"use client";

import { useQuery } from "@tanstack/react-query";
import { Boxes, GitBranch, Hourglass, FileQuestion, ArrowRight } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { useRole } from "@/lib/role";
import { StatCard } from "@/components/stat-card";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { ErrorState } from "@/components/error-state";
import { TeacherOnlyNotice } from "@/components/teacher-only";

const SHORTCUTS = [
  {
    href: "/demo",
    title: "Live Agent Demo",
    description: "Ask a question — see the Pedagogical Guide's classification and the Knowledge Manager's KG-grounded retrieval, live.",
  },
  {
    href: "/graph",
    title: "Knowledge Graph",
    description: "Explore the real Neo4j graph — entities and relationships extracted from ingested course material.",
  },
  {
    href: "/review",
    title: "Review Console",
    description: "Approve or reject pending ontology extensions and low-confidence triples before they join the graph.",
  },
  {
    href: "/pipeline",
    title: "Pipeline Runs",
    description: "Ingest a new document and watch all 12 pipeline stages complete live.",
  },
];

export default function DashboardPage() {
  const role = useRole();
  const { data, isLoading, isError, error } = useQuery({ queryKey: ["dashboard-summary"], queryFn: api.dashboardSummary });

  if (role !== "teacher") return <TeacherOnlyNotice />;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">Live state of the eduKG knowledge graph and review queues.</p>
      </div>

      {isError ? (
        <ErrorState error={error} label="Couldn't load dashboard data." />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard title="Entities in graph" value={data?.entity_count ?? 0} icon={Boxes} loading={isLoading} accent="emerald" />
          <StatCard
            title="Accepted relationships"
            value={data?.accepted_triple_count ?? 0}
            icon={GitBranch}
            loading={isLoading}
            accent="emerald"
          />
          <StatCard
            title="Triples pending review"
            value={data?.pending_triple_count ?? 0}
            icon={Hourglass}
            loading={isLoading}
            accent="amber"
          />
          <StatCard
            title="Ontology extensions pending"
            value={data?.pending_extension_count ?? 0}
            icon={FileQuestion}
            loading={isLoading}
            accent="amber"
          />
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {SHORTCUTS.map((s) => (
          <Card key={s.href} className="card-hover flex flex-col justify-between">
            <CardHeader>
              <CardTitle>{s.title}</CardTitle>
              <CardDescription>{s.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <Link href={s.href} className={buttonVariants({ variant: "secondary", size: "sm" })}>
                Open <ArrowRight className="ml-1 size-3.5" />
              </Link>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
