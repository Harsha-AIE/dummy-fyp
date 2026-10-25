// Hand-mirrored from src/edukg/api/schemas.py — keep in sync when the backend changes.

export interface ConceptOut {
  id: string;
  canonical_name: string;
}

export interface MasteryEstimateOut {
  concept_id: string;
  concept_name: string;
  bkt_mastery: number;
  dkt_mastery: number | null;
  ensemble_mastery: number;
  explanation: string;
  updated_at: string;
}

export interface MasteryVectorResponse {
  student_id: string;
  estimates: MasteryEstimateOut[];
}

export interface TopicRecommendationOut {
  concept_id: string;
  concept_name: string;
  mastery: number;
  reason: "foundational" | "ready";
  unblocks: string[];
}

export interface RecommendationsResponse {
  student_id: string;
  recommendations: TopicRecommendationOut[];
}

export interface KTEventOut {
  id: string;
  concept_id: string;
  source: "dialogue" | "worksheet" | "quiz";
  correct: boolean;
  weight: number;
  occurred_at: string;
}

export interface QuizQuestionOut {
  question_id: string;
  concept_id: string;
  concept_name: string;
  question_text: string;
  options: string[];
}

export interface QuizSubmitResponse {
  correct: boolean;
  correct_index: number;
  explanation: string;
  updated_mastery: MasteryEstimateOut | null;
}

export interface QuizDiagnosticResponse {
  document_id: string;
  questions: QuizQuestionOut[];
}

export interface ClassifyResponse {
  intent: string;
  confidence: number;
  escalated_to_llm: boolean;
  detected_concepts: ConceptOut[];
  updated_mastery: MasteryEstimateOut[];
}

export interface RetrievedChunkOut {
  chunk_id: string;
  text: string;
  page_number: number | null;
  fused_score: number;
  graph_score: number;
  final_score: number;
}

export interface RetrieveResponse {
  results: RetrievedChunkOut[];
}

export interface TutorResponseOut {
  reply: string;
  guardrail_passed: boolean;
  guardrail_note: string | null;
}

export interface AskResponse {
  classification: ClassifyResponse;
  retrieval: RetrieveResponse;
  tutor_response: TutorResponseOut;
}

export interface GraphNodeOut {
  id: string;
  name: string;
  type: string;
}

export interface GraphEdgeOut {
  source: string;
  target: string;
  predicate: string;
  confidence: number;
  evidence_count: number;
  inferred: boolean;
}

export interface GraphResponse {
  nodes: GraphNodeOut[];
  edges: GraphEdgeOut[];
}

export interface PendingExtensionOut {
  id: string;
  kind: string;
  subject: string;
  proposed_name: string;
  proposed_description: string | null;
  justification: string;
  confidence: number;
}

export interface PendingTripleOut {
  id: string;
  subject_entity_id: string;
  subject_name: string;
  predicate: string;
  object_entity_id: string;
  object_name: string;
  confidence: number;
  validation_status: string;
}

export interface PaginatedTriplesResponse {
  items: PendingTripleOut[];
  total: number;
  page: number;
  page_size: number;
}

export interface ReviewActionResponse {
  id: string;
  status: "approved" | "rejected";
}

export interface BulkActionResponse {
  count: number;
}

export interface DashboardSummaryResponse {
  entity_count: number;
  accepted_triple_count: number;
  pending_triple_count: number;
  pending_extension_count: number;
}

export interface AvailableDocumentOut {
  name: string;
  already_ingested: boolean;
  document_id: string | null;
}

export interface UploadedDocumentOut {
  name: string;
}

export interface PipelineRunCreated {
  run_id: string;
}

export interface PipelineStageEventOut {
  stage: string;
  status: "started" | "completed" | "failed" | "skipped";
  timestamp: string;
}

export interface PipelineRunSummary {
  document_id: string;
  chunk_count: number;
  mention_count: number;
  entity_count: number;
  relation_count: number;
  triple_count: number;
}

export interface PipelineRunStatus {
  run_id: string;
  state: "running" | "completed" | "failed";
  events: PipelineStageEventOut[];
  error: string | null;
  summary: PipelineRunSummary | null;
}

// Matches PipelineStage enum values in core/models/enums.py — order matters,
// this drives the pipeline dashboard's stepper UI.
export const PIPELINE_STAGES = [
  "ingestion",
  "subject_detection",
  "ontology_load",
  "ontology_extension",
  "ontology_validation",
  "entity_extraction",
  "entity_linking",
  "relation_extraction",
  "triple_generation",
  "triple_validation",
  "knowledge_fusion",
  "graph_storage",
] as const;

export type PipelineStageName = (typeof PIPELINE_STAGES)[number];

export interface ProfileOut {
  student_id: string;
  display_name: string;
  role: "teacher" | "student";
}

export interface RosterEntryOut {
  student_id: string;
  display_name: string;
  concept_count: number;
  average_mastery: number | null;
  last_active: string | null;
}

export interface RosterResponse {
  students: RosterEntryOut[];
}
