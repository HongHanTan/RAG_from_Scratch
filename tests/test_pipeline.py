import pytest

from rag.config import Config
from rag.pipeline import ask, build_index, load_index
from rag.store import VectorStore
from tests.conftest import FakeEmbedder, FakeLLM


# --- indexing ---------------------------------------------------------------

def test_build_index_covers_every_document(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    assert {c.doc_id for c in store.chunks} == {"alpha", "beta"}


def test_build_index_produces_one_vector_per_chunk(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    assert store.vectors.shape[0] == len(store.chunks)
    assert store.vectors.shape[1] == FakeEmbedder.dim


def test_build_index_writes_the_index_file(tiny_corpus: Config):
    build_index(tiny_corpus, FakeEmbedder())
    assert tiny_corpus.index_path.is_file()


def test_build_index_is_reproducible(tiny_corpus: Config):
    first = build_index(tiny_corpus, FakeEmbedder())
    second = build_index(tiny_corpus, FakeEmbedder())
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]


def test_load_index_round_trips(tiny_corpus: Config):
    built = build_index(tiny_corpus, FakeEmbedder())
    loaded = load_index(tiny_corpus)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in built.chunks]


def test_load_index_without_building_first_raises(tiny_corpus: Config):
    with pytest.raises(FileNotFoundError, match="rag index"):
        load_index(tiny_corpus)


# --- asking -----------------------------------------------------------------

def test_ask_returns_a_trace_with_the_question(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("what is cosine?", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert trace.question == "what is cosine?"


def test_ask_records_the_original_question_as_the_only_query(tiny_corpus: Config):
    # Phase 2 adds rewritten queries here; in Phase 1 there is exactly one.
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert trace.queries == ["q"]


def test_ask_retrieves_top_k_chunks(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, k=2)
    assert len(trace.retrieved) == 2


def test_ask_ranks_retrieved_chunks_from_one(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, k=3)
    assert [r.rank for r in trace.retrieved] == [1, 2, 3]


def test_ask_orders_retrieved_chunks_by_descending_score(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, k=3)
    scores = [r.score for r in trace.retrieved]
    assert scores == sorted(scores, reverse=True)


def test_ask_defaults_k_to_the_config(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert len(trace.retrieved) == tiny_corpus.top_k


def test_ask_records_the_answer(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM("an answer"), tiny_corpus)
    assert trace.answer == "an answer"


def test_ask_times_every_stage(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert [t.name for t in trace.timings] == ["embed", "search", "generate"]


def test_ask_without_an_llm_retrieves_but_does_not_generate(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), None, tiny_corpus)
    assert trace.retrieved
    assert trace.answer is None
    assert any("retrieval only" in n for n in trace.notes)
    assert [t.name for t in trace.timings] == ["embed", "search"]


def test_ask_gives_the_llm_the_retrieved_context(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM()
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, k=1)
    assert trace.retrieved[0].chunk.text in llm.prompts[0]


def test_build_index_records_the_config_it_was_built_with(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    assert store.meta["chunk_tokens"] == tiny_corpus.chunk_tokens
    assert store.meta["chunk_overlap"] == tiny_corpus.chunk_overlap
    assert store.meta["embedding_model"] == tiny_corpus.embedding_model


def test_load_index_refuses_an_index_built_with_different_chunking(
    tiny_corpus: Config,
):
    from dataclasses import replace

    build_index(tiny_corpus, FakeEmbedder())
    changed = replace(tiny_corpus, chunk_tokens=4, chunk_overlap=1)
    with pytest.raises(ValueError, match="chunk_tokens"):
        load_index(changed)


# --- strategy dispatch --------------------------------------------------------

def test_ask_defaults_to_the_direct_strategy(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert trace.strategy == "direct"


def test_ask_records_the_strategy_it_used(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, strategy="direct")
    assert trace.strategy == "direct"


def test_ask_rejects_an_unknown_strategy(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    with pytest.raises(ValueError, match="nope"):
        ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, strategy="nope")
