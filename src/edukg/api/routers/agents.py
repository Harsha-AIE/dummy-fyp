from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends

from edukg.agents.knowledge_manager.knowledge_manager_agent import RetrievedChunk
from edukg.api.schemas import (
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    ConceptOut,
    MasteryEstimateOut,
    RetrieveRequest,
    RetrievedChunkOut,
    RetrieveResponse,
    TutorResponseOut,
)
from edukg.api.state import AppResources, get_resources
from edukg.core.logging_setup import get_logger
from edukg.core.models.knowledge_tracing import MasteryEstimate

logger = get_logger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


def _classify(
    message: str, student_id: str | None, resources: AppResources
) -> tuple[ClassifyResponse, list[MasteryEstimate]]:
    turn = resources.pedagogical_guide_agent.classify(message, student_id=student_id)
    concepts: list[ConceptOut] = []
    for concept_id in turn.detected_concepts:
        entity = resources.entity_repo.get(concept_id)
        if entity is not None:
            concepts.append(ConceptOut(id=entity.id, canonical_name=entity.canonical_name))

    # Best-effort: a Knowledge Tracing failure (e.g. Postgres briefly
    # unreachable) must not break the classify response it's piggybacking
    # on — intent classification/concept tagging already succeeded above.
    estimates: list[MasteryEstimate] = []
    updated_mastery: list[MasteryEstimateOut] = []
    try:
        estimates = resources.knowledge_tracing_agent.record_dialogue_turn(turn)
        updated_mastery = [_mastery_out(resources, estimate) for estimate in estimates]
    except Exception as exc:  # noqa: BLE001
        logger.error("kt_record_dialogue_turn_failed", error=str(exc))

    response = ClassifyResponse(
        intent=turn.intent.value,
        confidence=turn.confidence,
        escalated_to_llm=turn.escalated_to_llm,
        detected_concepts=concepts,
        updated_mastery=updated_mastery,
    )
    return response, estimates


def _mastery_out(resources: AppResources, estimate) -> MasteryEstimateOut:
    entity = resources.entity_repo.get(estimate.concept_id)
    return MasteryEstimateOut(
        concept_id=estimate.concept_id,
        concept_name=entity.canonical_name if entity is not None else "this concept",
        bkt_mastery=estimate.bkt_mastery,
        dkt_mastery=estimate.dkt_mastery,
        ensemble_mastery=estimate.ensemble_mastery,
        explanation=estimate.explanation,
        updated_at=estimate.updated_at,
    )


def _retrieve(
    query: str, top_k: int, student_id: str | None, resources: AppResources
) -> tuple[RetrieveResponse, list[RetrievedChunk]]:
    results = resources.knowledge_manager_agent.retrieve(query, top_k=top_k, student_id=student_id)
    response = RetrieveResponse(
        results=[
            RetrievedChunkOut(
                chunk_id=r.chunk.id,
                text=r.chunk.text,
                page_number=r.chunk.page_number,
                fused_score=r.fused_score,
                graph_score=r.graph_score,
                final_score=r.final_score,
            )
            for r in results
        ]
    )
    return response, results


@router.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest, resources: AppResources = Depends(get_resources)) -> ClassifyResponse:
    response, _estimates = _classify(payload.message, payload.student_id, resources)
    return response


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest, resources: AppResources = Depends(get_resources)) -> RetrieveResponse:
    response, _results = _retrieve(payload.query, payload.top_k, payload.student_id, resources)
    return response


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, resources: AppResources = Depends(get_resources)) -> AskResponse:
    # classification and retrieval are fully independent (neither needs the
    # other's output), but classification can take several extra seconds
    # whenever it escalates to the tier-2 LLM — running them sequentially
    # meant every /ask request paid both costs back to back. A per-request
    # 2-worker pool (not a shared module-level one, which would bottleneck
    # concurrent requests against each other) runs them at the same time
    # instead, so retrieval's work happens "for free" during an LLM
    # escalation rather than strictly after it.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ask-agents") as executor:
        classification_future = executor.submit(_classify, payload.message, payload.student_id, resources)
        retrieval_future = executor.submit(_retrieve, payload.message, payload.top_k, payload.student_id, resources)
        classification, mastery_estimates = classification_future.result()
        retrieval, retrieved_chunks = retrieval_future.result()

    # The Tutor Response Orchestrator needs both prior results, so it can
    # only run after the concurrent pair above completes — this is the final
    # synthesis layer the original architecture called for (see
    # METHODOLOGY.md §5/§8): merges intent + evidence + mastery into one
    # reply instead of /ask returning three independent agent outputs.
    mastery_summary: list[tuple[str, float]] = []
    for estimate in mastery_estimates:
        entity = resources.entity_repo.get(estimate.concept_id)
        if entity is not None:
            mastery_summary.append((entity.canonical_name, estimate.ensemble_mastery))

    tutor_response = resources.tutor_response_agent.generate(
        message=payload.message,
        intent=classification.intent,
        concept_names=[c.canonical_name for c in classification.detected_concepts],
        evidence_chunks=[chunk.chunk.text for chunk in retrieved_chunks],
        mastery_summary=mastery_summary,
    )

    return AskResponse(
        classification=classification,
        retrieval=retrieval,
        tutor_response=TutorResponseOut(
            reply=tutor_response.reply,
            guardrail_passed=tutor_response.guardrail_passed,
            guardrail_note=tutor_response.guardrail_note,
        ),
    )
