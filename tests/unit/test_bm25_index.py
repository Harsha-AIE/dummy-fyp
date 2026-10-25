from pathlib import Path
from uuid import uuid4

from edukg.agents.knowledge_manager.bm25_index import BM25Index


def test_search_ranks_lexically_matching_chunk_first() -> None:
    id_a, id_b = uuid4(), uuid4()
    index = BM25Index()
    index.build([id_a, id_b], ["Gradient descent minimizes the loss function.", "Photosynthesis occurs in chloroplasts."])

    results = index.search("loss function gradient descent", top_k=2)

    assert results[0][0] == id_a


def test_search_on_empty_index_returns_empty() -> None:
    index = BM25Index()
    assert index.search("anything", top_k=5) == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    id_a = uuid4()
    index = BM25Index()
    index.build([id_a], ["Backpropagation computes gradients across layers."])
    path = tmp_path / "bm25.json"

    index.save(path)

    reloaded = BM25Index()
    reloaded.load(path)
    results = reloaded.search("backpropagation gradients", top_k=1)

    assert results and results[0][0] == id_a
