"""Wiring. This module composes the others and owns no algorithm of its own.

Later phases insert routing, query construction and translation between the
question arriving and the search running; the Trace shape does not change.
"""

from __future__ import annotations

from dataclasses import replace

from rag.chunking import chunk_documents
from rag.config import Config
from rag.generation import generate_answer
from rag.loader import load_documents
from rag.prompts import PROMPT_EXEMPLARS
from rag.query_construction import MetadataFilter, build_filter, compile_mask
from rag.routing import SemanticRouter, logical_route
from rag.store import VectorStore
from rag.strategies import get_strategy
from rag.strategies.base import StrategyContext
from rag.trace import Trace


def build_index(config: Config, embedder) -> VectorStore:
    """Load the corpus, chunk it, embed it, and persist the result."""
    documents = load_documents(config.corpus_dir, config.metadata_path)
    chunks = chunk_documents(
        documents, embedder.tokenizer, config.chunk_tokens, config.chunk_overlap
    )
    vectors = embedder.encode([chunk.text for chunk in chunks])
    store = VectorStore(vectors=vectors, chunks=chunks)
    store.meta = _index_meta(config, store.dim)
    store.doc_meta = {
        doc.doc_id: {
            "title": doc.title,
            "source": doc.source,
            "topic": doc.topic,
            "publish_date": doc.publish_date,
            "author": doc.author,
            "url": doc.url,
        }
        for doc in documents
    }
    store.save(config.index_path, meta=store.meta)
    return store


def _index_meta(config: Config, dim: int) -> dict:
    """What an index must agree with to be safely reused.

    Chunk size and overlap change which text each vector represents, and the
    embedding model changes what the vectors mean. Reusing an index across any
    of those returns plausible nonsense rather than an error, because the
    dimensions still line up.
    """
    return {
        "embedding_model": config.embedding_model,
        "chunk_tokens": config.chunk_tokens,
        "chunk_overlap": config.chunk_overlap,
        "dim": dim,
    }


def load_index(config: Config) -> VectorStore:
    # dim is unknown until the file is read, and an index whose dim differs
    # already fails cleanly at search time, so it is left out of the check.
    expected = _index_meta(config, dim=None)
    expected.pop("dim")
    return VectorStore.load(config.index_path, expect_meta=expected)


def ask(
    question: str,
    store: VectorStore,
    embedder,
    llm,
    config: Config,
    k: int | None = None,
    strategy: str = "direct",
    strategy_options: dict | None = None,
    generate: bool = True,
    route: bool = False,
    construct: bool = False,
    semantic_prompt: bool = False,
) -> Trace:
    """Answer one question. Pass llm=None to retrieve without generating.

    The data flow is route, then construct, then translate, then search: a
    logical route narrows the topic before query construction adds its own
    constraints, and both run before the strategy translates the question and
    searches. The named strategy decides what to retrieve; everything after
    that is the same for all of them.
    """
    trace = Trace(question=question)
    trace.strategy = strategy
    effective_k = config.top_k if k is None else k

    topics = tuple(
        sorted({r.get("topic") for r in store.doc_meta.values() if r.get("topic")})
    )

    active = MetadataFilter()
    if route and topics:
        # Nothing to route between is not a failure and must not cost an LLM
        # call, so routing is skipped entirely when the corpus has no topics.
        chosen = logical_route(question, llm, topics, trace)
        if chosen:
            active = MetadataFilter(topics=chosen)
    if construct:
        inferred = build_filter(question, llm, topics, trace)
        if not inferred.is_empty():
            # Routing narrows by topic; construction adds its own constraints.
            # Keep the routed topics unless construction named its own.
            active = MetadataFilter(
                topics=inferred.topics or active.topics,
                authors=inferred.authors,
                published_before=inferred.published_before,
                published_after=inferred.published_after,
            )

    mask = None
    if not active.is_empty():
        mask = compile_mask(active, store.chunks, store.doc_meta)
        if not mask.any():
            # A filter matching nothing must return no results rather than
            # silently falling back to unfiltered search, which would answer
            # a filtered question from the whole corpus.
            trace.note(
                f"filter matched no documents ({active.describe()}); "
                "returning no results"
            )

    prompt_name = None
    if semantic_prompt:
        prompt_name = SemanticRouter(embedder, PROMPT_EXEMPLARS).route(question, trace)

    chosen_strategy = get_strategy(strategy, **(strategy_options or {}))
    ctx = StrategyContext(
        store=store,
        embedder=embedder,
        llm=llm,
        config=replace(
            config,
            top_k=effective_k,
            retrieval_depth=max(config.retrieval_depth, effective_k),
        ),
        trace=trace,
        mask=mask,
    )
    result = chosen_strategy.run(question, ctx)
    trace.retrieved = result.retrieved

    if llm is None:
        trace.note("retrieval only: no LLM configured")
        return trace

    if not generate:
        # The strategy still used the LLM to translate the query; only the
        # final answer is skipped. The benchmark measures retrieval, and
        # generating an answer it never reads would cost a call per question.
        return trace

    generate_answer(
        llm,
        question,
        result.retrieved,
        trace,
        extra_context=result.extra_context,
        prompt_name=prompt_name,
    )
    return trace
