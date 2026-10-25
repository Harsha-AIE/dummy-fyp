import { CheckCircle2, CircleDashed, LoaderCircle, XCircle } from "lucide-react";
import { PIPELINE_STAGES } from "@/lib/types";
import type { StageStatus } from "@/hooks/use-pipeline-stream";
import { cn } from "@/lib/utils";

const STAGE_LABELS: Record<string, string> = {
  ingestion: "Ingestion",
  subject_detection: "Subject Detection",
  ontology_load: "Ontology Load",
  ontology_extension: "Ontology Extension",
  ontology_validation: "Ontology Validation",
  entity_extraction: "Entity Extraction",
  entity_linking: "Entity Linking",
  relation_extraction: "Relation Extraction",
  triple_generation: "Triple Generation",
  triple_validation: "Triple Validation",
  knowledge_fusion: "Knowledge Fusion",
  graph_storage: "Graph Storage",
};

export function PipelineStepper({ statuses }: { statuses: Partial<Record<string, StageStatus>> }) {
  return (
    <ol className="space-y-1">
      {PIPELINE_STAGES.map((stage, index) => {
        const status = statuses[stage] ?? "pending";
        return (
          <li
            key={stage}
            className={cn(
              "flex items-center gap-3 rounded-md border px-3 py-2 text-sm transition-colors",
              status === "started" && "border-blue-500/40 bg-blue-500/5",
              status === "completed" && "border-emerald-500/40 bg-emerald-500/5",
              status === "failed" && "border-destructive/40 bg-destructive/5",
              status === "pending" && "border-transparent",
            )}
          >
            <span className="w-5 text-xs text-muted-foreground">{index + 1}</span>
            <StatusIcon status={status} />
            <span className={cn("flex-1", status === "pending" && "text-muted-foreground")}>{STAGE_LABELS[stage]}</span>
            <span className="text-xs capitalize text-muted-foreground">{status}</span>
          </li>
        );
      })}
    </ol>
  );
}

function StatusIcon({ status }: { status: StageStatus }) {
  switch (status) {
    case "completed":
    case "skipped":
      return <CheckCircle2 className="size-4 text-emerald-500" />;
    case "started":
      return <LoaderCircle className="size-4 animate-spin text-blue-500" />;
    case "failed":
      return <XCircle className="size-4 text-destructive" />;
    default:
      return <CircleDashed className="size-4 text-muted-foreground" />;
  }
}
