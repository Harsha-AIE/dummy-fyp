import type {
  AskResponse,
  AvailableDocumentOut,
  BulkActionResponse,
  ClassifyResponse,
  DashboardSummaryResponse,
  GraphResponse,
  KTEventOut,
  MasteryVectorResponse,
  PaginatedTriplesResponse,
  PendingExtensionOut,
  PipelineRunCreated,
  PipelineRunStatus,
  ProfileOut,
  QuizDiagnosticResponse,
  QuizQuestionOut,
  QuizSubmitResponse,
  RecommendationsResponse,
  RetrieveResponse,
  ReviewActionResponse,
  RosterResponse,
  UploadedDocumentOut,
} from "@/lib/types";
import { getStudentId } from "@/lib/student-id";
import type { Role } from "@/lib/role";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, detail || response.statusText);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),

  dashboardSummary: () => request<DashboardSummaryResponse>("/api/dashboard/summary"),

  ask: (message: string, topK = 5) =>
    request<AskResponse>("/api/agents/ask", {
      method: "POST",
      body: JSON.stringify({ message, top_k: topK, student_id: getStudentId() }),
    }),
  classify: (message: string) =>
    request<ClassifyResponse>("/api/agents/classify", {
      method: "POST",
      body: JSON.stringify({ message, student_id: getStudentId() }),
    }),
  retrieve: (query: string, topK = 5) =>
    request<RetrieveResponse>("/api/agents/retrieve", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK, student_id: getStudentId() }),
    }),

  graph: () => request<GraphResponse>("/api/graph"),
  documentGraph: (documentId: string) => request<GraphResponse>(`/api/graph/document/${documentId}`),

  masteryVector: (studentId: string) => request<MasteryVectorResponse>(`/api/kt/mastery/${studentId}`),
  ktEvents: (studentId: string) => request<KTEventOut[]>(`/api/kt/events/${studentId}`),
  recommendations: (studentId: string) => request<RecommendationsResponse>(`/api/kt/recommendations/${studentId}`),

  pendingExtensions: () => request<PendingExtensionOut[]>("/api/review/ontology-extensions"),
  approveExtension: (id: string) =>
    request<ReviewActionResponse>(`/api/review/ontology-extensions/${id}/approve`, { method: "POST" }),
  rejectExtension: (id: string) =>
    request<ReviewActionResponse>(`/api/review/ontology-extensions/${id}/reject`, { method: "POST" }),
  pendingTriples: (page = 1, pageSize = 50) =>
    request<PaginatedTriplesResponse>(`/api/review/triples?page=${page}&page_size=${pageSize}`),
  approveTriple: (id: string) => request<ReviewActionResponse>(`/api/review/triples/${id}/approve`, { method: "POST" }),
  rejectTriple: (id: string) => request<ReviewActionResponse>(`/api/review/triples/${id}/reject`, { method: "POST" }),
  bulkApproveTriples: (minConfidence: number) =>
    request<BulkActionResponse>("/api/review/triples/bulk-approve", {
      method: "POST",
      body: JSON.stringify({ min_confidence: minConfidence }),
    }),
  bulkRejectTriples: (maxConfidence: number) =>
    request<BulkActionResponse>("/api/review/triples/bulk-reject", {
      method: "POST",
      body: JSON.stringify({ max_confidence: maxConfidence }),
    }),

  availableDocuments: () => request<AvailableDocumentOut[]>("/api/pipeline/documents"),
  uploadDocument: async (file: File): Promise<UploadedDocumentOut> => {
    // Deliberately not going through request(): that helper always sets
    // Content-Type: application/json, which would break multipart uploads —
    // the browser must set its own Content-Type (with the boundary) when
    // the body is a FormData, so no Content-Type header is set here at all.
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${BASE_URL}/api/pipeline/documents`, { method: "POST", body: formData });
    if (!response.ok) {
      const detail = await response.text().catch(() => response.statusText);
      throw new ApiError(response.status, detail || response.statusText);
    }
    return response.json() as Promise<UploadedDocumentOut>;
  },
  createPipelineRun: (path: string) =>
    request<PipelineRunCreated>("/api/pipeline/runs", { method: "POST", body: JSON.stringify({ path }) }),
  getPipelineRun: (runId: string) => request<PipelineRunStatus>(`/api/pipeline/runs/${runId}`),
  pipelineRunStreamUrl: (runId: string) => `${BASE_URL}/api/pipeline/runs/${runId}/stream`,

  quizQuestion: (conceptId: string) =>
    request<QuizQuestionOut>("/api/quiz/question", {
      method: "POST",
      body: JSON.stringify({ concept_id: conceptId, student_id: getStudentId() }),
    }),
  quizSubmit: (questionId: string, selectedIndex: number) =>
    request<QuizSubmitResponse>("/api/quiz/submit", {
      method: "POST",
      body: JSON.stringify({ question_id: questionId, selected_index: selectedIndex, student_id: getStudentId() }),
    }),
  quizDiagnostic: (documentId: string, limit = 5) =>
    request<QuizDiagnosticResponse>("/api/quiz/diagnostic", {
      method: "POST",
      body: JSON.stringify({ document_id: documentId, student_id: getStudentId(), limit }),
    }),

  registerProfile: (studentId: string, role: Role, displayName: string) =>
    request<ProfileOut>("/api/roster/register", {
      method: "POST",
      body: JSON.stringify({ student_id: studentId, role, display_name: displayName }),
    }),
  roster: () => request<RosterResponse>("/api/roster/students"),
};
