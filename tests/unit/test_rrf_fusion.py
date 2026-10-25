from uuid import uuid4

from edukg.agents.knowledge_manager.rrf_fusion import reciprocal_rank_fusion


def test_top_ranked_in_both_lists_wins() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    fused = reciprocal_rank_fusion([[a, b, c], [a, c, b]])

    assert fused[0][0] == a  # rank 0 in both lists


def test_item_only_in_one_list_still_scores() -> None:
    a, b = uuid4(), uuid4()
    fused = reciprocal_rank_fusion([[a], []])

    ids = [item_id for item_id, _ in fused]
    assert a in ids
    assert b not in ids


def test_empty_rankings_produce_empty_result() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_agreement_across_lists_outranks_single_list_top_pick() -> None:
    a, b = uuid4(), uuid4()
    # a is rank-0 in list1 only. b is rank-1 in list1 AND rank-0 in list2 —
    # appearing (even lower) in both lists should beat a single top pick.
    fused = reciprocal_rank_fusion([[a, b], [b]])
    scores = dict(fused)

    assert scores[b] > scores[a]
