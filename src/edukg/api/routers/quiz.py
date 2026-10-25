from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from edukg.agents.quiz.quiz_agent import QuizAgent
from edukg.agents.quiz.quiz_registry import QuizRegistry
from edukg.api.schemas import (
    MasteryEstimateOut,
    QuizDiagnosticRequest,
    QuizDiagnosticResponse,
    QuizQuestionOut,
    QuizQuestionRequest,
    QuizSubmitRequest,
    QuizSubmitResponse,
)
from edukg.api.state import AppResources, get_resources
from edukg.modules.content.document_concepts import get_document_chunk_texts_for_concept, get_document_top_concepts

router = APIRouter(prefix="/api/quiz", tags=["quiz"])

# Module-level, same deliberate in-memory/process-lifetime scope as
# PipelineRunRegistry (routers/pipeline.py's `_registry`) — nothing here
# needs to survive a restart.
_registry = QuizRegistry()


def _build_quiz_agent(resources: AppResources) -> QuizAgent:
    """Constructed fresh per request rather than cached on AppResources:
    resources.knowledge_manager_agent is swapped out by
    AppResources.refresh_content_index() after every pipeline run (see
    state.py), so caching a QuizAgent built on today's instance would keep
    generating questions against a stale, pre-refresh index until the next
    process restart — the exact staleness bug _populate_entity_linking_index
    was written to avoid elsewhere. QuizAgent's __init__ is two attribute
    assignments, so this costs nothing."""
    return QuizAgent(resources.llm_provider, resources.knowledge_manager_agent)


@router.post("/question", response_model=QuizQuestionOut)
def generate_question(
    payload: QuizQuestionRequest, resources: AppResources = Depends(get_resources)
) -> QuizQuestionOut:
    entity = resources.entity_repo.get(payload.concept_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"No concept {payload.concept_id}")

    question = _build_quiz_agent(resources).generate_question(payload.concept_id, entity.canonical_name)
    _registry.put(question)
    return QuizQuestionOut(
        question_id=question.id,
        concept_id=question.concept_id,
        concept_name=question.concept_name,
        question_text=question.question_text,
        options=question.options,
    )


@router.post("/submit", response_model=QuizSubmitResponse)
def submit_answer(payload: QuizSubmitRequest, resources: AppResources = Depends(get_resources)) -> QuizSubmitResponse:
    question = _registry.pop(payload.question_id)
    if question is None:
        raise HTTPException(status_code=404, detail=f"No pending quiz question {payload.question_id}")

    correct = payload.selected_index == question.correct_index
    estimate = resources.knowledge_tracing_agent.record_quiz_result(
        payload.student_id, question.concept_id, correct, weight=1.0
    )
    updated_mastery = (
        MasteryEstimateOut(
            concept_id=estimate.concept_id,
            concept_name=question.concept_name,
            bkt_mastery=estimate.bkt_mastery,
            dkt_mastery=estimate.dkt_mastery,
            ensemble_mastery=estimate.ensemble_mastery,
            explanation=estimate.explanation,
            updated_at=estimate.updated_at,
        )
        if estimate is not None
        else None
    )
    return QuizSubmitResponse(
        correct=correct,
        correct_index=question.correct_index,
        explanation=question.explanation,
        updated_mastery=updated_mastery,
    )


@router.post("/diagnostic", response_model=QuizDiagnosticResponse)
def generate_diagnostic(
    payload: QuizDiagnosticRequest, resources: AppResources = Depends(get_resources)
) -> QuizDiagnosticResponse:
    top_concepts = get_document_top_concepts(
        resources.content_index, resources.entity_repo, payload.document_id, limit=payload.limit
    )
    quiz_agent = _build_quiz_agent(resources)

    questions: list[QuizQuestionOut] = []
    for concept_id, concept_name in top_concepts:
        # Document-scoped evidence, not a corpus-wide retrieve() — see
        # QuizAgent.generate_question's docstring for why this matters.
        evidence = get_document_chunk_texts_for_concept(resources.content_index, payload.document_id, concept_id)
        question = quiz_agent.generate_question(concept_id, concept_name, evidence=evidence)
        _registry.put(question)
        questions.append(
            QuizQuestionOut(
                question_id=question.id,
                concept_id=question.concept_id,
                concept_name=question.concept_name,
                question_text=question.question_text,
                options=question.options,
            )
        )
    return QuizDiagnosticResponse(document_id=payload.document_id, questions=questions)
