import json
from dataclasses import asdict

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


# --- provenance, atomic writes, suffix handling ------------------------------


def _meta():
    return {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_tokens": 200,
        "chunk_overlap": 50,
        "dim": 3,
    }


def test_save_records_provenance_and_load_returns_it(tmp_path):
    path = tmp_path / "index.npz"
    _store(3).save(path, meta=_meta())
    assert VectorStore.load(path).meta == _meta()


def test_load_rejects_an_index_built_with_a_different_config(tmp_path):
    path = tmp_path / "index.npz"
    _store(3).save(path, meta=_meta())
    expected = dict(_meta(), chunk_tokens=400)
    with pytest.raises(ValueError, match="chunk_tokens"):
        VectorStore.load(path, expect_meta=expected)


def test_load_accepts_a_matching_config(tmp_path):
    path = tmp_path / "index.npz"
    _store(3).save(path, meta=_meta())
    assert len(VectorStore.load(path, expect_meta=_meta())) == 3


def test_load_tolerates_an_index_saved_without_meta(tmp_path):
    # Indexes written before provenance existed must still load.
    path = tmp_path / "index.npz"
    store = _store(3)
    np.savez_compressed(
        path,
        vectors=store.vectors,
        chunks=np.array(json.dumps([asdict(c) for c in store.chunks])),
    )
    loaded = VectorStore.load(path, expect_meta=_meta())
    assert len(loaded) == 3
    assert loaded.meta == {}


def test_save_and_load_agree_when_the_suffix_is_omitted(tmp_path):
    # save(Path("index")) writes index.npz; load(Path("index")) must find it
    # rather than telling the user to run the command they just ran.
    base = tmp_path / "index"
    _store(2).save(base)
    assert len(VectorStore.load(base)) == 2


def test_save_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "index.npz"
    _store(2).save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["index.npz"]


def test_a_failed_save_does_not_destroy_the_existing_index(tmp_path):
    path = tmp_path / "index.npz"
    _store(3).save(path)
    good = path.read_bytes()

    class Unserialisable:
        pass

    store = _store(2)
    store.chunks = [Unserialisable()]  # json.dumps will raise mid-save
    with pytest.raises(Exception):
        store.save(path)
    assert path.read_bytes() == good


def _doc_meta():
    return {"d": {"title": "T", "source": "arxiv", "topic": "foundations",
                  "publish_date": "2020-01-01", "author": "A", "url": None}}


def test_doc_meta_round_trips(tmp_path):
    path = tmp_path / "index.npz"
    store = _store(3)
    store.doc_meta = _doc_meta()
    store.save(path, meta=None)
    assert VectorStore.load(path).doc_meta == _doc_meta()


def test_doc_meta_defaults_to_empty(tmp_path):
    path = tmp_path / "index.npz"
    _store(2).save(path)
    assert VectorStore.load(path).doc_meta == {}


def test_an_index_written_without_doc_meta_still_loads(tmp_path):
    # Indexes built before this field existed must not become unreadable.
    import json
    from dataclasses import asdict

    path = tmp_path / "index.npz"
    store = _store(2)
    np.savez_compressed(
        path,
        vectors=store.vectors,
        chunks=np.array(json.dumps([asdict(c) for c in store.chunks])),
    )
    loaded = VectorStore.load(path)
    assert len(loaded) == 2
    assert loaded.doc_meta == {}
