import numpy as np
import pytest

from rag.chunking import Chunk
from rag.config import Config
from rag.indexing.raptor import build_raptor
from rag.trace import Trace
from tests.conftest import FakeEmbedder


class ScriptedLLM:
    def __init__(self, reply="a cluster summary"):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return f"{self.reply} {self.calls}"


def _chunks(n=40):
    return [Chunk(f"d:{i}", f"d{i % 4}", i, f"chunk text {i}", 0, 10, i * 50, i * 50 + 40)
            for i in range(n)]


def _vectors(n=40, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _config(**kw):
    # The defaults are spelled out so these tests do not silently change
    # meaning if Config's defaults ever move, but `kw` must be able to
    # override them rather than collide with them.
    return Config(**{"raptor_max_depth": 3, "raptor_cluster_size": 8, **kw})


def test_returns_the_original_chunks_plus_summaries():
    chunks, vectors = _chunks(), _vectors()
    out_chunks, out_vectors = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert len(out_chunks) > len(chunks)
    assert out_chunks[: len(chunks)] == chunks


def test_original_chunks_come_first_and_keep_their_order():
    # evaluation/spans.py resolves gold spans positionally against
    # store.chunks; appending summaries rather than interleaving them keeps
    # Phase 3's numbers comparable.
    chunks, vectors = _chunks(), _vectors()
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert [c.chunk_id for c in out_chunks[: len(chunks)]] == [c.chunk_id for c in chunks]


def test_vectors_and_chunks_stay_aligned():
    chunks, vectors = _chunks(), _vectors()
    out_chunks, out_vectors = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert out_vectors.shape[0] == len(out_chunks)


def test_summaries_are_synthetic_and_levelled():
    chunks, vectors = _chunks(), _vectors()
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    synthetic = [c for c in out_chunks if c.is_synthetic]
    assert synthetic
    assert all(c.level >= 1 for c in synthetic)


def test_levels_increase_as_the_tree_is_built():
    chunks, vectors = _chunks(80), _vectors(80)
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    levels = sorted({c.level for c in out_chunks})
    assert levels[0] == 0
    assert len(levels) >= 2


def test_recursion_stops_at_the_depth_cap():
    chunks, vectors = _chunks(200), _vectors(200)
    out_chunks, _ = build_raptor(
        chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config(raptor_max_depth=2)
    )
    assert max(c.level for c in out_chunks) <= 2


def test_recursion_stops_when_a_level_collapses_to_one_node():
    # Without this, a level of one would cluster into one cluster forever.
    chunks, vectors = _chunks(10), _vectors(10)
    out_chunks, _ = build_raptor(
        chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config(raptor_max_depth=9, raptor_cluster_size=8)
    )
    assert max(c.level for c in out_chunks) < 9


def test_summary_ids_are_unique():
    chunks, vectors = _chunks(80), _vectors(80)
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    ids = [c.chunk_id for c in out_chunks]
    assert len(ids) == len(set(ids))


def test_a_cluster_summary_has_a_synthetic_doc_id():
    # It spans several documents, so it belongs to none of them. That is what
    # makes compile_mask exclude it from filtered search.
    chunks, vectors = _chunks(), _vectors()
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    summary = next(c for c in out_chunks if c.is_synthetic)
    assert summary.doc_id not in {c.doc_id for c in chunks}


def test_building_is_deterministic():
    chunks, vectors = _chunks(), _vectors()
    a, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    b, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_a_failed_cluster_summary_is_skipped_and_noted():
    class OneFails:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                from rag.llm import LLMError

                raise LLMError("rate limited")
            return f"summary {self.calls}"

    trace = Trace(question="build")
    chunks, vectors = _chunks(), _vectors()
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), OneFails(), _config(), trace)
    assert any("degraded" in n for n in trace.notes)
    assert len(out_chunks) > len(chunks)


def test_too_few_chunks_to_cluster_returns_them_unchanged():
    chunks, vectors = _chunks(1), _vectors(1)
    out_chunks, out_vectors = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert out_chunks == chunks
    assert out_vectors.shape[0] == 1
