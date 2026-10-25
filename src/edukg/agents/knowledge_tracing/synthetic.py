from __future__ import annotations

import random
from dataclasses import dataclass

from edukg.agents.knowledge_tracing.bkt import BKTParams, DEFAULT_BKT_PARAMS


@dataclass(slots=True)
class Interaction:
    concept_idx: int
    correct: bool


def generate_synthetic_sequences(
    concept_indices: list[int],
    *,
    student_count: int = 60,
    interactions_range: tuple[int, int] = (20, 60),
    concepts_per_student_range: tuple[int, int] = (2, 6),
    params: BKTParams = DEFAULT_BKT_PARAMS,
) -> list[list[Interaction]]:
    """Cold-start bootstrap data for DKT (scripts/train_dkt.py) and a fresh,
    never-trained-on held-out set for evaluating it (scripts/metrics_report.py)
    — same generator, different calls/seeds, so "held-out" is genuine.

    One simulated student = a random subset of concepts, each following its
    own independent BKT generative trajectory (the same p_init/p_transit/
    p_slip/p_guess model bkt.py itself uses), interleaved into a single
    chronological sequence the way a real multi-topic dialogue would be.
    """
    sequences: list[list[Interaction]] = []

    for _ in range(student_count):
        n_interactions = random.randint(*interactions_range)
        k = min(len(concept_indices), random.randint(*concepts_per_student_range))
        touched_concepts = random.sample(concept_indices, k=k)
        mastered = {c: random.random() < params.p_init for c in touched_concepts}

        seq: list[Interaction] = []
        for _ in range(n_interactions):
            concept_idx = random.choice(touched_concepts)
            if mastered[concept_idx]:
                correct = random.random() > params.p_slip
            else:
                correct = random.random() < params.p_guess
            seq.append(Interaction(concept_idx, correct))
            if not mastered[concept_idx]:
                mastered[concept_idx] = random.random() < params.p_transit
        sequences.append(seq)

    return sequences
