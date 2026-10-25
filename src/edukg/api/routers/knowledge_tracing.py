from __future__ import annotations

from fastapi import APIRouter, Depends

from edukg.api.schemas import (
    KTEventOut,
    MasteryEstimateOut,
    MasteryVectorResponse,
    RecommendationsResponse,
    TopicRecommendationOut,
)
from edukg.api.state import AppResources, get_resources

router = APIRouter(prefix="/api/kt", tags=["knowledge-tracing"])


@router.get("/mastery/{student_id}", response_model=MasteryVectorResponse)
def mastery(student_id: str, resources: AppResources = Depends(get_resources)) -> MasteryVectorResponse:
    estimates = resources.knowledge_tracing_agent.get_mastery_vector(student_id)
    out: list[MasteryEstimateOut] = []
    for estimate in estimates:
        entity = resources.entity_repo.get(estimate.concept_id)
        out.append(
            MasteryEstimateOut(
                concept_id=estimate.concept_id,
                concept_name=entity.canonical_name if entity is not None else "this concept",
                bkt_mastery=estimate.bkt_mastery,
                dkt_mastery=estimate.dkt_mastery,
                ensemble_mastery=estimate.ensemble_mastery,
                explanation=estimate.explanation,
                updated_at=estimate.updated_at,
            )
        )
    return MasteryVectorResponse(student_id=student_id, estimates=out)


@router.get("/recommendations/{student_id}", response_model=RecommendationsResponse)
def recommendations(student_id: str, resources: AppResources = Depends(get_resources)) -> RecommendationsResponse:
    recs = resources.topic_recommender.recommend(student_id)
    return RecommendationsResponse(
        student_id=student_id,
        recommendations=[
            TopicRecommendationOut(
                concept_id=r.concept_id,
                concept_name=r.concept_name,
                mastery=r.mastery,
                reason=r.reason,
                unblocks=r.unblocks,
            )
            for r in recs
        ],
    )


@router.get("/events/{student_id}", response_model=list[KTEventOut])
def events(student_id: str, resources: AppResources = Depends(get_resources)) -> list[KTEventOut]:
    return [
        KTEventOut(
            id=e.id,
            concept_id=e.concept_id,
            source=e.source.value,
            correct=e.correct,
            weight=e.weight,
            occurred_at=e.created_at,
        )
        for e in resources.kt_store.list_events(student_id)
    ]
