from edukg.agents.knowledge_tracing.bkt import BKTParams, compute_mastery, update_mastery


def test_empty_history_returns_prior() -> None:
    params = BKTParams(p_init=0.3)
    assert compute_mastery([], params) == params.p_init


def test_repeated_correct_answers_raise_mastery() -> None:
    mastery = compute_mastery([(True, 1.0)] * 5)
    assert mastery > BKTParams().p_init


def test_repeated_incorrect_answers_lower_mastery_relative_to_correct() -> None:
    correct_mastery = compute_mastery([(True, 1.0)] * 5)
    incorrect_mastery = compute_mastery([(False, 1.0)] * 5)
    assert incorrect_mastery < correct_mastery


def test_weight_dampens_the_update_toward_the_prior() -> None:
    params = BKTParams()
    full_weight = update_mastery(params.p_init, correct=True, weight=1.0, params=params)
    low_weight = update_mastery(params.p_init, correct=True, weight=0.2, params=params)
    # A low-confidence positive observation should move mastery up by less
    # than a fully-trusted one — closer to the untouched prior.
    assert params.p_init < low_weight < full_weight


def test_zero_weight_leaves_mastery_at_the_transition_of_the_untouched_prior() -> None:
    params = BKTParams()
    updated = update_mastery(params.p_init, correct=False, weight=0.0, params=params)
    # weight=0 blends fully back to the prior before the transition step, so
    # this should equal running the transition step on the prior alone.
    expected = params.p_init + (1.0 - params.p_init) * params.p_transit
    assert abs(updated - expected) < 1e-9


def test_mastery_stays_within_bounds_over_a_long_sequence() -> None:
    observations = [(True, 1.0), (False, 1.0)] * 20
    mastery = compute_mastery(observations)
    assert 0.0 <= mastery <= 1.0
