import pytest

from rag.chunking import chunk_documents
from rag.config import Config
from rag.llm import LLMError
from rag.loader import load_documents
from rag.store import VectorStore
from rag.strategies import STRATEGY_NAMES, get_strategy
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct
from rag.trace import Trace
from tests.conftest import FakeEmbedder, FakeLLM


def build_context(tiny_corpus: Config, llm=None) -> StrategyContext:
    embedder = FakeEmbedder()
    documents = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    chunks = chunk_documents(
        documents, embedder.tokenizer, tiny_corpus.chunk_tokens, tiny_corpus.chunk_overlap
    )
    store = VectorStore(
        vectors=embedder.encode([c.text for c in chunks]), chunks=chunks
    )
    return StrategyContext(
        store=store,
        embedder=embedder,
        llm=llm,
        config=tiny_corpus,
        trace=Trace(question="q"),
    )


class FailingLLM:
    def generate(self, prompt: str) -> str:
        raise LLMError("rate limited")


# --- context -----------------------------------------------------------------

def test_context_search_returns_one_list_per_query(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    results = ctx.search(["alpha", "beta"], k=2)
    assert len(results) == 2
    assert all(len(r) == 2 for r in results)


def test_context_search_times_embed_and_search(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    ctx.search(["alpha"], k=1)
    assert [t.name for t in ctx.trace.timings] == ["embed", "search"]


def test_context_search_of_no_queries_returns_nothing(tiny_corpus: Config):
    assert build_context(tiny_corpus).search([], k=3) == []


# --- direct strategy ---------------------------------------------------------

def test_direct_retrieves_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    result = get_strategy("direct").run("what is cosine?", ctx)
    assert len(result.retrieved) == tiny_corpus.top_k
    assert result.extra_context is None


def test_direct_records_the_question_as_the_only_query(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    get_strategy("direct").run("q", ctx)
    assert ctx.trace.queries == ["q"]


def test_direct_needs_no_llm(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=None)
    assert get_strategy("direct").run("q", ctx).retrieved


# --- degradation helper ------------------------------------------------------

def test_degrade_falls_back_to_direct_retrieval(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    result = degrade_to_direct("q", ctx, "multi-query rewrite failed")
    assert len(result.retrieved) == tiny_corpus.top_k


def test_degrade_records_the_reason_on_the_trace(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    degrade_to_direct("q", ctx, "multi-query rewrite failed")
    assert any("multi-query rewrite failed" in n for n in ctx.trace.notes)
    assert any("direct retrieval" in n for n in ctx.trace.notes)


# --- registry ----------------------------------------------------------------

def test_registry_exposes_every_strategy_name():
    assert "direct" in STRATEGY_NAMES


def test_unknown_strategy_is_rejected_by_name():
    with pytest.raises(ValueError, match="nope"):
        get_strategy("nope")


# --- multi-query ---------------------------------------------------------------

MULTI_QUERY_REPLY = """1. How does cosine similarity work?
2. What does the angle between vectors measure?
3. Why is magnitude ignored in cosine similarity?"""


def test_multi_query_sends_the_question_to_the_llm(tiny_corpus: Config):
    llm = FakeLLM(MULTI_QUERY_REPLY)
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("multi-query").run("what is cosine?", ctx)
    assert "what is cosine?" in llm.prompts[0]


def test_multi_query_records_the_original_plus_the_rewrites(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    get_strategy("multi-query").run("what is cosine?", ctx)
    assert ctx.trace.queries[0] == "what is cosine?"
    assert "How does cosine similarity work?" in ctx.trace.queries
    assert len(ctx.trace.queries) == 4


def test_multi_query_records_each_rewrite_as_a_translation_step(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    get_strategy("multi-query").run("q", ctx)
    kinds = [s.kind for s in ctx.trace.translation]
    assert kinds == ["query", "query", "query"]


def test_multi_query_returns_at_most_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("multi-query").run("q", ctx)
    assert len(result.retrieved) <= tiny_corpus.top_k


def test_multi_query_deduplicates_chunks_found_by_several_queries(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("multi-query").run("q", ctx)
    ids = [r.chunk.chunk_id for r in result.retrieved]
    assert len(ids) == len(set(ids))


def test_multi_query_results_are_ranked_from_one(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("multi-query").run("q", ctx)
    assert [r.rank for r in result.retrieved] == list(range(1, len(result.retrieved) + 1))


def test_multi_query_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("multi-query").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_multi_query_degrades_when_there_is_no_llm(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=None)
    result = get_strategy("multi-query").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_multi_query_degrades_when_the_llm_returns_nothing_usable(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("   "))
    result = get_strategy("multi-query").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_multi_query_drops_rewrites_that_are_near_duplicates_of_the_question(
    tiny_corpus: Config,
):
    reply = "1. What is cosine similarity?  \n2. What is cosine similarity?\n3. How does cosine similarity work?"
    ctx = build_context(tiny_corpus, llm=FakeLLM(reply))
    get_strategy("multi-query").run("what is cosine similarity?", ctx)
    assert ctx.trace.queries == [
        "what is cosine similarity?",
        "How does cosine similarity work?",
    ]


def test_multi_query_truncates_rewrites_to_n(tiny_corpus: Config):
    reply = "\n".join(f"{i}. rewrite {i}" for i in range(1, 9))
    ctx = build_context(tiny_corpus, llm=FakeLLM(reply))
    get_strategy("multi-query", n=3).run("q", ctx)
    # question + at most n rewrites
    assert len(ctx.trace.queries) == 4
    assert len(ctx.trace.translation) == 3


def test_rag_fusion_scores_are_labelled_rrf(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("rag-fusion").run("q", ctx)
    assert all(r.score_kind == "rrf" for r in result.retrieved)


def test_rag_fusion_scores_differ_from_multi_query_scores(tiny_corpus: Config):
    # Same rewrites, same corpus: if fusion returned cosine scores it would be
    # multi-query wearing a different name.
    fusion_ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    fusion = get_strategy("rag-fusion").run("q", fusion_ctx)
    merge_ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    merged = get_strategy("multi-query").run("q", merge_ctx)
    assert fusion.retrieved[0].score != merged.retrieved[0].score


def test_rag_fusion_returns_at_most_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    assert len(get_strategy("rag-fusion").run("q", ctx).retrieved) <= tiny_corpus.top_k


def test_rag_fusion_reranks_from_one(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("rag-fusion").run("q", ctx)
    assert [r.rank for r in result.retrieved] == list(range(1, len(result.retrieved) + 1))


def test_rag_fusion_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("rag-fusion").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_rag_fusion_degrades_when_there_is_no_llm(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=None)
    result = get_strategy("rag-fusion").run("q", ctx)
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_rag_fusion_drops_rewrites_that_are_near_duplicates_of_the_question(
    tiny_corpus: Config,
):
    reply = "1. What is cosine similarity?  \n2. What is cosine similarity?\n3. How does cosine similarity work?"
    ctx = build_context(tiny_corpus, llm=FakeLLM(reply))
    get_strategy("rag-fusion").run("what is cosine similarity?", ctx)
    assert ctx.trace.queries == [
        "what is cosine similarity?",
        "How does cosine similarity work?",
    ]


def test_rag_fusion_truncates_rewrites_to_n(tiny_corpus: Config):
    reply = "\n".join(f"{i}. rewrite {i}" for i in range(1, 9))
    ctx = build_context(tiny_corpus, llm=FakeLLM(reply))
    get_strategy("rag-fusion", n=3).run("q", ctx)
    assert len(ctx.trace.queries) == 4
    assert len(ctx.trace.translation) == 3


# --- step-back -----------------------------------------------------------------

def test_step_back_asks_for_a_more_general_question(tiny_corpus: Config):
    llm = FakeLLM("What is vector similarity?")
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("step-back").run("how does cosine handle magnitude?", ctx)
    assert "how does cosine handle magnitude?" in llm.prompts[0]


def test_step_back_searches_both_questions(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("What is vector similarity?"))
    get_strategy("step-back").run("specific question", ctx)
    assert ctx.trace.queries == ["specific question", "What is vector similarity?"]


def test_step_back_records_the_general_question_as_a_translation_step(
    tiny_corpus: Config,
):
    ctx = build_context(tiny_corpus, llm=FakeLLM("What is vector similarity?"))
    get_strategy("step-back").run("q", ctx)
    assert [(s.kind, s.text) for s in ctx.trace.translation] == [
        ("step_back", "What is vector similarity?")
    ]


def test_step_back_returns_at_most_top_k_deduplicated(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("general question"))
    result = get_strategy("step-back").run("q", ctx)
    ids = [r.chunk.chunk_id for r in result.retrieved]
    assert len(ids) == len(set(ids))
    assert len(ids) <= tiny_corpus.top_k


def test_step_back_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("step-back").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_step_back_degrades_when_the_general_question_is_blank(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("   "))
    result = get_strategy("step-back").run("q", ctx)
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


# --- hyde ------------------------------------------------------------------

HYDE_REPLY = (
    "Cosine similarity measures the angle between two vectors in an inner "
    "product space. Because it normalises for magnitude, two documents on the "
    "same topic score highly even when one is far longer than the other."
)


def test_hyde_generates_a_hypothetical_document(tiny_corpus: Config):
    llm = FakeLLM(HYDE_REPLY)
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("hyde").run("what is cosine similarity?", ctx)
    assert "what is cosine similarity?" in llm.prompts[0]


def test_hyde_records_the_hypothetical_document(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    get_strategy("hyde").run("q", ctx)
    assert [(s.kind, s.text) for s in ctx.trace.translation] == [
        ("hypothetical", HYDE_REPLY)
    ]


def test_hyde_searches_on_the_hypothetical_document(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    get_strategy("hyde").run("q", ctx)
    assert HYDE_REPLY in ctx.trace.queries


def test_hyde_also_searches_the_original_question_by_default(tiny_corpus: Config):
    # The hypothetical document can be wrong. Keeping the original question in
    # the mix means a bad hallucination degrades the result rather than
    # replacing it.
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    get_strategy("hyde").run("q", ctx)
    assert "q" in ctx.trace.queries


def test_hyde_can_search_the_hypothetical_document_alone(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    get_strategy("hyde", include_question=False).run("q", ctx)
    assert ctx.trace.queries == [HYDE_REPLY]


def test_hyde_returns_at_most_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    assert len(get_strategy("hyde").run("q", ctx).retrieved) <= tiny_corpus.top_k


def test_hyde_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("hyde").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_hyde_degrades_when_the_document_is_blank(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("  \n "))
    result = get_strategy("hyde").run("q", ctx)
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)
