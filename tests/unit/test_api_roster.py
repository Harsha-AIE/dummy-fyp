"""Router functions called directly, same convention as test_api_kt.py /
test_api_quiz.py — no FastAPI TestClient, a lightweight fake resources
object standing in for AppResources."""

from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from pydantic import ValidationError

from edukg.api.routers import roster
from edukg.api.schemas import ProfileRegisterRequest
from edukg.core.models.enums import ProfileRole
from edukg.core.models.knowledge_tracing import MasteryEstimate
from edukg.core.models.profile import Profile


class _FakeProfileStore:
    def __init__(self, profiles: list[Profile] | None = None) -> None:
        self._profiles: dict[str, Profile] = {p.student_id: p for p in (profiles or [])}

    def upsert_profile(self, student_id: str, display_name: str, role: ProfileRole) -> Profile:
        profile = Profile(student_id=student_id, display_name=display_name, role=role)
        self._profiles[student_id] = profile
        return profile

    def list_by_role(self, role: ProfileRole) -> list[Profile]:
        return [p for p in self._profiles.values() if p.role == role]


class _FakeKTStore:
    def __init__(self, estimates: list[MasteryEstimate] | None = None) -> None:
        self._estimates = estimates or []

    def list_all_mastery(self) -> list[MasteryEstimate]:
        return self._estimates


@dataclass
class _FakeResources:
    profile_store: _FakeProfileStore = field(default_factory=_FakeProfileStore)
    kt_store: _FakeKTStore = field(default_factory=_FakeKTStore)


def _estimate(student_id: str, ensemble_mastery: float) -> MasteryEstimate:
    return MasteryEstimate(
        student_id=student_id,
        concept_id=uuid4(),
        bkt_mastery=ensemble_mastery,
        ensemble_mastery=ensemble_mastery,
        explanation="x",
    )


def test_register_request_rejects_an_invalid_role() -> None:
    # Literal["teacher", "student"], not a bare str — a bad value must be
    # rejected by Pydantic at request-parsing time (a clean 422), not by an
    # unguarded ProfileRole(body.role) raising a bare ValueError inside the
    # handler, which would bypass the app's EduKGError handler entirely and
    # surface as an unhandled 500.
    with pytest.raises(ValidationError):
        ProfileRegisterRequest(student_id="s1", display_name="x", role="admin")


def test_register_creates_a_new_profile() -> None:
    resources = _FakeResources()

    result = roster.register(
        ProfileRegisterRequest(student_id="s1", display_name="Harsha", role="student"), resources
    )

    assert result.student_id == "s1"
    assert result.display_name == "Harsha"
    assert result.role == "student"


def test_register_upserts_rather_than_duplicates() -> None:
    resources = _FakeResources()
    roster.register(ProfileRegisterRequest(student_id="s1", display_name="Old Name", role="student"), resources)

    result = roster.register(
        ProfileRegisterRequest(student_id="s1", display_name="New Name", role="teacher"), resources
    )

    assert result.display_name == "New Name"
    assert result.role == "teacher"
    assert len(resources.profile_store._profiles) == 1


def test_students_endpoint_includes_zero_mastery_student() -> None:
    profile = Profile(student_id="s1", display_name="Harsha", role=ProfileRole.STUDENT)
    resources = _FakeResources(profile_store=_FakeProfileStore([profile]))

    result = roster.students(resources)

    assert len(result.students) == 1
    assert result.students[0].display_name == "Harsha"
    assert result.students[0].concept_count == 0
    assert result.students[0].average_mastery is None
    assert result.students[0].last_active is None


def test_students_endpoint_aggregates_mastery_correctly() -> None:
    profile = Profile(student_id="s1", display_name="Harsha", role=ProfileRole.STUDENT)
    resources = _FakeResources(
        profile_store=_FakeProfileStore([profile]),
        kt_store=_FakeKTStore([_estimate("s1", 0.4), _estimate("s1", 0.8)]),
    )

    result = roster.students(resources)

    entry = result.students[0]
    assert entry.concept_count == 2
    assert entry.average_mastery == pytest.approx(0.6)
    assert entry.last_active is not None


def test_students_endpoint_excludes_teachers() -> None:
    teacher = Profile(student_id="t1", display_name="Dhanush", role=ProfileRole.TEACHER)
    student = Profile(student_id="s1", display_name="Harsha", role=ProfileRole.STUDENT)
    resources = _FakeResources(profile_store=_FakeProfileStore([teacher, student]))

    result = roster.students(resources)

    assert len(result.students) == 1
    assert result.students[0].student_id == "s1"


def test_students_endpoint_only_aggregates_the_matching_student() -> None:
    profile = Profile(student_id="s1", display_name="Harsha", role=ProfileRole.STUDENT)
    resources = _FakeResources(
        profile_store=_FakeProfileStore([profile]),
        kt_store=_FakeKTStore([_estimate("s1", 0.5), _estimate("someone-else", 0.9)]),
    )

    result = roster.students(resources)

    assert result.students[0].concept_count == 1
    assert result.students[0].average_mastery == 0.5
