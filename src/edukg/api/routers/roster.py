from __future__ import annotations

from fastapi import APIRouter, Depends

from edukg.api.schemas import ProfileOut, ProfileRegisterRequest, RosterEntryOut, RosterResponse
from edukg.api.state import AppResources, get_resources
from edukg.core.models.enums import ProfileRole
from edukg.core.models.knowledge_tracing import MasteryEstimate

router = APIRouter(prefix="/api/roster", tags=["roster"])


@router.post("/register", response_model=ProfileOut)
def register(body: ProfileRegisterRequest, resources: AppResources = Depends(get_resources)) -> ProfileOut:
    profile = resources.profile_store.upsert_profile(body.student_id, body.display_name, ProfileRole(body.role))
    return ProfileOut(student_id=profile.student_id, display_name=profile.display_name, role=profile.role.value)


@router.get("/students", response_model=RosterResponse)
def students(resources: AppResources = Depends(get_resources)) -> RosterResponse:
    profiles = resources.profile_store.list_by_role(ProfileRole.STUDENT)

    # Students-first, mastery second: a student who hasn't engaged with
    # anything yet still belongs on the roster (with no mastery data),
    # which an inner join starting from mastery_estimates would silently
    # drop.
    estimates_by_student: dict[str, list[MasteryEstimate]] = {}
    for estimate in resources.kt_store.list_all_mastery():
        estimates_by_student.setdefault(estimate.student_id, []).append(estimate)

    entries = []
    for profile in profiles:
        rows = estimates_by_student.get(profile.student_id, [])
        entries.append(
            RosterEntryOut(
                student_id=profile.student_id,
                display_name=profile.display_name,
                concept_count=len(rows),
                average_mastery=(sum(r.ensemble_mastery for r in rows) / len(rows)) if rows else None,
                last_active=max((r.updated_at for r in rows), default=None),
            )
        )
    return RosterResponse(students=entries)
