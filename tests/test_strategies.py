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
