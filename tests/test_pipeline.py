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


def test_ask_can_skip_generation(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM()
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, generate=False)
    assert trace.retrieved
    assert trace.answer is None
    assert llm.prompts == []
    assert [t.name for t in trace.timings] == ["embed", "search"]


def test_ask_without_generation_still_uses_the_llm_for_translation(
    tiny_corpus: Config,
):
    # generate=False must skip the *answer*, not the rewrite -- otherwise the
    # benchmark would measure every strategy as plain retrieval.
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM("1. first rewrite\n2. second rewrite")
    trace = ask(
        "q", store, FakeEmbedder(), llm, tiny_corpus,
        strategy="multi-query", generate=False,
    )
    assert len(llm.prompts) == 1
    assert trace.answer is None
    assert len(trace.queries) == 3


def test_build_index_records_document_metadata(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    assert set(store.doc_meta) == {"alpha", "beta"}
    assert store.doc_meta["alpha"]["title"] == "Cosine Similarity"


def test_load_index_restores_document_metadata(tiny_corpus: Config):
    build_index(tiny_corpus, FakeEmbedder())
    assert load_index(tiny_corpus).doc_meta["beta"]["publish_date"] == "2024-02-11"


# --- routing and query construction wiring -----------------------------------

def test_ask_without_routing_or_construction_is_unchanged(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert trace.retrieved
    assert not any(s.kind in ("route", "filter") for s in trace.translation)


def test_ask_with_construction_records_a_filter(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "2024-01-01"}')
    trace = ask("before 2024?", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert any(s.kind == "filter" for s in trace.translation)


def test_construction_restricts_what_is_retrieved(tiny_corpus: Config):
    # tiny_corpus: alpha is 2023-05-01, beta is 2024-02-11.
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "2024-01-01"}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert trace.retrieved
    assert all(r.chunk.doc_id == "alpha" for r in trace.retrieved)


def test_a_filter_matching_nothing_leaves_a_note_and_no_results(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "1900-01-01"}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert trace.retrieved == []
    assert any("filter" in n.lower() for n in trace.notes)


def test_filter_exclusion_note_reports_the_count(tiny_corpus: Config):
    # tiny_corpus: alpha is 2023-05-01 (kept), beta is 2024-02-11 (excluded).
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "2024-01-01"}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert any("excluded 1 of 2 documents" in n for n in trace.notes)


def test_filter_exclusion_note_names_the_excluded_document_when_few(
    tiny_corpus: Config,
):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "2024-01-01"}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert any("beta" in n and "excluded" in n for n in trace.notes)


def test_filter_exclusion_note_is_reported_even_when_nothing_is_excluded(
    tiny_corpus: Config,
):
    # Unconditional means unconditional: an active filter that happens to
    # exclude nothing still gets the note, with a zero count.
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "2100-01-01"}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert any("excluded 0 of 2 documents" in n for n in trace.notes)


def test_filter_exclusion_note_omits_names_when_there_are_many(tiny_corpus: Config):
    from rag.chunking import Chunk
    from rag.store import VectorStore

    embedder = FakeEmbedder()
    chunks = [Chunk(f"d{i}:0", f"d{i}", 0, "text", 0, 4, 0, 4) for i in range(8)]
    store = VectorStore(
        vectors=embedder.encode([c.text for c in chunks]), chunks=chunks
    )
    store.doc_meta = {
        f"d{i}": {"topic": None, "author": None, "publish_date": "2020-01-01"}
        for i in range(8)
    }
    llm = FakeLLM('{"published_before": "2019-01-01"}')
    trace = ask("q", store, embedder, llm, tiny_corpus, construct=True)
    exclusion_note = next(n for n in trace.notes if "excluded" in n)
    assert "excluded 8 of 8 documents" in exclusion_note
    assert ":" not in exclusion_note.split("documents", 1)[1]


def test_ask_with_routing_records_a_route(tiny_corpus: Config):
    # The tiny fixture corpus has no topics, so give it two: routing is
    # skipped entirely when there is nothing to choose between.
    store = build_index(tiny_corpus, FakeEmbedder())
    store.doc_meta["alpha"]["topic"] = "similarity"
    store.doc_meta["beta"]["topic"] = "fusion"
    llm = FakeLLM('{"topics": ["similarity"]}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, route=True)
    assert any(s.kind == "route" for s in trace.translation)


def test_routing_is_skipped_when_the_corpus_has_no_topics(tiny_corpus: Config):
    # Nothing to route between is not a failure, and must not cost an LLM call.
    # generate=False isolates that claim to routing itself -- answer
    # generation is a separate, unconditional LLM call the moment an LLM is
    # given at all, and asserting it away here would conflate the two.
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"topics": ["anything"]}')
    trace = ask(
        "q", store, FakeEmbedder(), llm, tiny_corpus, route=True, generate=False
    )
    assert not any(s.kind == "route" for s in trace.translation)
    assert llm.prompts == []


def test_semantic_prompt_selection_records_its_choice(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask(
        "q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, semantic_prompt=True
    )
    assert any(s.kind == "route" and "prompt" in s.text for s in trace.translation)


def test_semantic_prompt_is_skipped_without_an_llm(tiny_corpus: Config):
    # With no LLM there is no generation at all, so a prompt choice would
    # never be used -- tracing one anyway would show a routing decision that
    # had no effect.
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), None, tiny_corpus, semantic_prompt=True)
    assert not any(s.kind == "route" for s in trace.translation)


def test_semantic_router_is_built_once_per_embedder_across_calls(
    tiny_corpus: Config,
):
    # SemanticRouter's own docstring promises exemplars are embedded once at
    # construction; building a fresh router inside every ask() call would
    # re-embed them on every question instead, silently breaking that
    # promise even though nothing at the class level would ever show it.
    from rag.prompts import PROMPT_EXEMPLARS

    store = build_index(tiny_corpus, FakeEmbedder())
    embedder = FakeEmbedder()
    calls = []
    original = embedder.encode

    def counting(texts, batch_size=32):
        calls.append(len(texts))
        return original(texts, batch_size=batch_size)

    embedder.encode = counting

    ask("q1", store, embedder, FakeLLM(), tiny_corpus, semantic_prompt=True)
    ask("q2", store, embedder, FakeLLM(), tiny_corpus, semantic_prompt=True)

    # One batch of exemplars embedded once, plus one query embedding per
    # ask() call for retrieval and one for the router -- never a second
    # exemplar batch.
    exemplar_batches = [c for c in calls if c == len(PROMPT_EXEMPLARS)]
    assert len(exemplar_batches) == 1


# --- index modes ------------------------------------------------------------

def test_index_path_differs_per_mode(tiny_corpus: Config):
    from rag.pipeline import index_path_for

    flat = index_path_for(tiny_corpus, "flat")
    raptor = index_path_for(tiny_corpus, "raptor")
    assert flat != raptor
    assert flat == tiny_corpus.index_path
    assert "raptor" in raptor.name


def test_an_unknown_mode_is_rejected(tiny_corpus: Config):
    from rag.pipeline import index_path_for

    with pytest.raises(ValueError, match="nope"):
        index_path_for(tiny_corpus, "nope")


def test_index_meta_records_the_mode(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    assert store.meta["index_mode"] == "flat"


def test_loading_a_flat_index_as_raptor_is_refused(tiny_corpus: Config):
    from rag.pipeline import index_path_for

    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    # Point the raptor path at the flat file to simulate a stale build.
    index_path_for(tiny_corpus, "raptor").write_bytes(
        index_path_for(tiny_corpus, "flat").read_bytes()
    )
    with pytest.raises(ValueError, match="index_mode"):
        load_index(tiny_corpus, mode="raptor")


def test_a_legacy_index_without_a_mode_still_loads_as_flat(tiny_corpus: Config):
    # VectorStore.load tolerates a key that is absent from the stored meta
    # ("if key in meta and meta[key] != value"), which is what lets an index
    # written before index_mode existed still load. Do not tighten that: the
    # non-flat modes write their own files, so a file at the flat path with
    # no index_mode can only be a legacy flat index, and refusing it would
    # force a needless rebuild.
    import json

    import numpy as np

    from rag.pipeline import index_path_for

    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    path = index_path_for(tiny_corpus, "flat")
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    meta = json.loads(str(payload["meta"]))
    del meta["index_mode"]
    payload["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, **payload)

    store = load_index(tiny_corpus, mode="flat")
    assert len(store) > 0


def test_building_multirep_needs_an_llm(tiny_corpus: Config):
    with pytest.raises(ValueError, match="llm"):
        build_index(tiny_corpus, FakeEmbedder(), llm=None, mode="multirep")


def test_building_raptor_needs_an_llm(tiny_corpus: Config):
    with pytest.raises(ValueError, match="llm"):
        build_index(tiny_corpus, FakeEmbedder(), llm=None, mode="raptor")


def test_multirep_index_has_one_node_per_document(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), llm=FakeLLM("a summary"), mode="multirep")
    assert len(store) == 2
    assert store.docstore


def test_raptor_index_is_larger_than_flat(tiny_corpus: Config):
    flat = build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    raptor = build_index(
        tiny_corpus, FakeEmbedder(), llm=FakeLLM("a summary"), mode="raptor"
    )
    assert len(raptor) >= len(flat)


def test_a_multirep_hit_is_expanded_to_the_document(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), llm=FakeLLM("a summary"), mode="multirep")
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, generate=False)
    texts = {r.chunk.text for r in trace.retrieved}
    assert "a summary" not in texts
    assert any("Cosine similarity" in t or "Reciprocal rank" in t for t in texts)


def test_expansion_applies_to_every_strategy_not_just_direct(tiny_corpus: Config):
    # The expansion happens after the strategy returns, so a translated
    # strategy gets documents too. Retrieving summaries through hyde or
    # multi-query and handing them to the generator unexpanded would make
    # the technique silently strategy-dependent.
    store = build_index(tiny_corpus, FakeEmbedder(), llm=FakeLLM("a summary"), mode="multirep")
    for strategy, response in (
        ("multi-query", "1. first rewrite\n2. second rewrite"),
        ("hyde", "a hypothetical passage about vectors"),
    ):
        trace = ask(
            "q", store, FakeEmbedder(), FakeLLM(response), tiny_corpus,
            strategy=strategy, generate=False,
        )
        texts = {r.chunk.text for r in trace.retrieved}
        assert "a summary" not in texts, strategy
        assert any(
            "Cosine similarity" in t or "Reciprocal rank" in t for t in texts
        ), strategy


def test_expansion_does_not_happen_without_a_docstore(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, generate=False)
    assert trace.retrieved


def test_the_trace_records_which_index_was_used(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), llm=FakeLLM("s"), mode="multirep")
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, generate=False)
    assert any("multirep" in n for n in trace.notes)


def test_the_trace_records_a_raptor_index(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), llm=FakeLLM("s"), mode="raptor")
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, generate=False)
    assert any("raptor" in n for n in trace.notes)
