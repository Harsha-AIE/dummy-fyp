import random

from edukg.agents.knowledge_tracing.bkt import BKTParams
from edukg.agents.knowledge_tracing.synthetic import generate_synthetic_sequences


def test_generates_requested_student_count() -> None:
    random.seed(0)
    sequences = generate_synthetic_sequences([0, 1, 2, 3], student_count=10, interactions_range=(5, 5))
    assert len(sequences) == 10
    assert all(len(seq) == 5 for seq in sequences)


def test_interactions_only_use_provided_concept_indices() -> None:
    random.seed(1)
    concept_indices = [7, 8, 9]
    sequences = generate_synthetic_sequences(concept_indices, student_count=5, interactions_range=(10, 10))
    used = {i.concept_idx for seq in sequences for i in seq}
    assert used <= set(concept_indices)


def test_high_p_guess_and_p_init_produce_mostly_correct_answers() -> None:
    random.seed(2)
    params = BKTParams(p_init=0.95, p_transit=0.5, p_slip=0.0, p_guess=0.95)
    sequences = generate_synthetic_sequences(
        [0, 1], student_count=20, interactions_range=(30, 30), params=params
    )
    all_interactions = [i for seq in sequences for i in seq]
    correct_rate = sum(1 for i in all_interactions if i.correct) / len(all_interactions)
    assert correct_rate > 0.85
