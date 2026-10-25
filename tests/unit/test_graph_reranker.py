from uuid import uuid4

from edukg.agents.knowledge_manager.graph_reranker import GraphDistanceReranker


class _FakeGraphRepository:
    def __init__(self, distances: dict[tuple, int | None]) -> None:
        self._distances = distances
        self.bulk_calls: list[tuple] = []

    def shortest_path_length(self, from_entity_id, to_entity_id, max_hops):
        return self._distances.get((from_entity_id, to_entity_id))

    def shortest_path_lengths_bulk(self, from_ids, to_ids, max_hops):
        self.bulk_calls.append((tuple(from_ids), tuple(to_ids), max_hops))
        result = {}
        for f in from_ids:
            for t in to_ids:
                d = self._distances.get((f, t))
                if d is not None:
                    result[(f, t)] = d
        return result


def test_no_concepts_on_either_side_scores_zero() -> None:
    reranker = GraphDistanceReranker(_FakeGraphRepository({}))
    assert reranker.score([], [uuid4()]) == 0.0
    assert reranker.score([uuid4()], []) == 0.0


def test_identical_concept_scores_one() -> None:
    shared = uuid4()
    reranker = GraphDistanceReranker(_FakeGraphRepository({(shared, shared): 0}))
    assert reranker.score([shared], [shared]) == 1.0


def test_closest_pair_wins_across_multiple_candidates() -> None:
    q1, q2, c1, c2 = uuid4(), uuid4(), uuid4(), uuid4()
    graph = _FakeGraphRepository({(q1, c1): 3, (q1, c2): 1, (q2, c1): None, (q2, c2): 2})
    reranker = GraphDistanceReranker(graph)

    score = reranker.score([q1, q2], [c1, c2])

    assert score == 1.0 / (1 + 1)  # best distance found is 1, via (q1, c2)


def test_no_path_within_bound_scores_zero() -> None:
    q, c = uuid4(), uuid4()
    reranker = GraphDistanceReranker(_FakeGraphRepository({(q, c): None}))
    assert reranker.score([q], [c]) == 0.0


def test_score_batch_matches_score_for_each_candidate_individually() -> None:
    q1, q2, c1, c2, c3 = uuid4(), uuid4(), uuid4(), uuid4(), uuid4()
    graph = _FakeGraphRepository({(q1, c1): 3, (q1, c2): 1, (q2, c1): None, (q2, c2): 2, (q1, c3): 0})
    reranker = GraphDistanceReranker(graph)

    candidates = [[c1], [c1, c2], [c3], []]
    batch_scores = reranker.score_batch([q1, q2], candidates)

    assert batch_scores == [reranker.score([q1, q2], c) for c in candidates]


def test_score_batch_issues_a_single_bulk_lookup_not_one_per_candidate() -> None:
    """The whole point of score_batch: one Neo4j round trip for the entire
    candidate set, not one per candidate — this is what turns O(candidates)
    sequential graph queries per RAG request into O(1)."""
    q, c1, c2 = uuid4(), uuid4(), uuid4()
    graph = _FakeGraphRepository({(q, c1): 1, (q, c2): 2})
    reranker = GraphDistanceReranker(graph)

    reranker.score_batch([q], [[c1] for _ in range(20)] + [[c2] for _ in range(20)])

    assert len(graph.bulk_calls) == 1


def test_score_batch_empty_query_concepts_scores_all_zero() -> None:
    reranker = GraphDistanceReranker(_FakeGraphRepository({}))
    assert reranker.score_batch([], [[uuid4()], [uuid4()]]) == [0.0, 0.0]
