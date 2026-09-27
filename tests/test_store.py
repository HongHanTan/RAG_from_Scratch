import numpy as np
import pytest

from rag.chunking import Chunk
from rag.store import VectorStore


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"d:{i}",
        doc_id="d",
        index=i,
        text=f"chunk {i}",
        token_start=i * 10,
        token_end=i * 10 + 10,
        char_start=i * 50,
        char_end=i * 50 + 50,
    )


def _store(n: int = 3) -> VectorStore:
    vectors = np.eye(n, dtype=np.float32)
    return VectorStore(vectors=vectors, chunks=[_chunk(i) for i in range(n)])


def test_length_is_the_chunk_count():
    assert len(_store(3)) == 3


def test_dim_reports_the_vector_width():
    assert _store(3).dim == 3


def test_vector_and_chunk_counts_must_agree():
    with pytest.raises(ValueError, match="3 vectors"):
        VectorStore(vectors=np.eye(3, dtype=np.float32), chunks=[_chunk(0)])


def test_search_returns_the_nearest_chunk_first():
    store = _store(3)
    query = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    results = store.search(query, k=1)
    assert len(results) == 1
    item = results[0][0]
    assert item.chunk.chunk_id == "d:1"
    assert item.score == pytest.approx(1.0, abs=1e-6)


def test_search_returns_one_result_list_per_query():
    store = _store(3)
    queries = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    results = store.search(queries, k=2)
    assert [r[0].chunk.chunk_id for r in results] == ["d:0", "d:2"]


def test_search_results_are_ordered_by_descending_score():
    store = _store(3)
    query = np.array([[0.9, 0.4, 0.1]], dtype=np.float32)
    scores = [item.score for item in store.search(query, k=3)[0]]
    assert scores == sorted(scores, reverse=True)


def test_search_assigns_one_based_ranks_in_score_order():
    store = _store(3)
    query = np.array([[0.9, 0.4, 0.1]], dtype=np.float32)
    results = store.search(query, k=3)[0]
    assert [item.rank for item in results] == [1, 2, 3]


def test_search_k_larger_than_the_store_returns_everything():
    assert len(_store(2).search(np.array([[1.0, 0.0]], dtype=np.float32), k=10)[0]) == 2


def test_search_on_an_empty_store_returns_empty_results():
    store = VectorStore(vectors=np.zeros((0, 4), dtype=np.float32), chunks=[])
    assert store.search(np.ones((1, 4), dtype=np.float32), k=5) == [[]]


def test_scores_are_plain_floats_not_numpy_scalars():
    # The trace is serialised to JSON for the dashboard; np.float32 is not
    # JSON-serialisable and the failure would surface far from here.
    item = _store(2).search(np.array([[1.0, 0.0]], dtype=np.float32), k=1)[0][0]
    assert type(item.score) is float


def test_save_and_load_round_trip(tmp_path):
    store = _store(3)
    path = tmp_path / "index.npz"
    store.save(path)
    loaded = VectorStore.load(path)
    np.testing.assert_allclose(loaded.vectors, store.vectors)
    assert loaded.chunks == store.chunks


def test_loaded_vectors_are_float32(tmp_path):
    path = tmp_path / "index.npz"
    _store(3).save(path)
    assert VectorStore.load(path).vectors.dtype == np.float32


def test_save_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "index.npz"
    _store(2).save(path)
    assert path.is_file()


def test_loading_a_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope.npz")


# --- normalisation at the store boundary (search stops re-normalising) ------


def test_vectors_are_unit_length_after_construction():
    # Rows of very different magnitude, none already unit length.
    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    store = VectorStore(vectors=vectors, chunks=[_chunk(0), _chunk(1)])
    norms = np.linalg.norm(store.vectors, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_vectors_are_unit_length_after_load(tmp_path):
    vectors = np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    store = VectorStore(vectors=vectors, chunks=[_chunk(0), _chunk(1)])
    path = tmp_path / "index.npz"
    store.save(path)
    loaded = VectorStore.load(path)
    norms = np.linalg.norm(loaded.vectors, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
