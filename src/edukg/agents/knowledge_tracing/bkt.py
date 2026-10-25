from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BKTParams:
    """Standard Bayesian Knowledge Tracing parameters (Corbett & Anderson,
    1994). Defaults are the commonly-cited literature values, not yet fit
    against real EduKG usage data — swappable per-concept once enough real
    observations exist to estimate them properly."""

    p_init: float = 0.3
    """P(L0): prior probability the student already knows the concept."""
    p_transit: float = 0.1
    """P(T): probability of learning after one opportunity."""
    p_slip: float = 0.1
    """P(S): probability of an incorrect response despite knowing it."""
    p_guess: float = 0.2
    """P(G): probability of a correct response despite not knowing it."""


DEFAULT_BKT_PARAMS = BKTParams()


def update_mastery(prior: float, correct: bool, weight: float = 1.0, params: BKTParams = DEFAULT_BKT_PARAMS) -> float:
    """One Bayesian Knowledge Tracing update step.

    Bayes' rule turns the observation into a posterior P(mastered |
    observation), then the fixed learning-transition probability P(T) is
    applied so mastery can only move up between one observed opportunity and
    the next.

    `weight` (0..1) dampens partial-confidence observations: engagement
    signals derived from unchecked dialogue (e.g. "asked a question about
    the concept") are real but weaker evidence than a directly detected
    misconception, so weight<1.0 blends the Bayes-updated posterior back
    toward the prior instead of fully trusting the observation. weight=1.0
    is a standard, undamped BKT step.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0.0 and 1.0")
    if not 0.0 <= prior <= 1.0:
        raise ValueError("prior must be between 0.0 and 1.0")

    if correct:
        p_obs_given_mastered = 1.0 - params.p_slip
        p_obs_given_not_mastered = params.p_guess
    else:
        p_obs_given_mastered = params.p_slip
        p_obs_given_not_mastered = 1.0 - params.p_guess

    numerator = prior * p_obs_given_mastered
    denominator = numerator + (1.0 - prior) * p_obs_given_not_mastered
    posterior = numerator / denominator if denominator > 0 else prior

    posterior = weight * posterior + (1.0 - weight) * prior

    return posterior + (1.0 - posterior) * params.p_transit


def compute_mastery(observations: list[tuple[bool, float]], params: BKTParams = DEFAULT_BKT_PARAMS) -> float:
    """Replays a chronological (correct, weight) observation sequence
    starting from the P(L0) prior. An empty history returns the prior
    itself — "no evidence yet," not a fabricated estimate."""
    mastery = params.p_init
    for correct, weight in observations:
        mastery = update_mastery(mastery, correct, weight, params)
    return mastery
