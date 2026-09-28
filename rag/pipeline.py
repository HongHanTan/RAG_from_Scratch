"""Wiring. This module composes the others and owns no algorithm of its own.

Later phases insert routing, query construction and translation between the
question arriving and the search running; the Trace shape does not change.
"""

from __future__ import annotations

import weakref
from dataclasses import replace
from pathlib import Path

from rag.chunking import chunk_documents
from rag.config import Config
from rag.generation import generate_answer
from rag.indexing.multi_representation import build_multi_representation
from rag.indexing.raptor import build_raptor
from rag.loader import load_documents
from rag.prompts import PROMPT_EXEMPLARS
from rag.query_construction import MetadataFilter, build_filter, compile_mask
from rag.routing import SemanticRouter, logical_route
from rag.store import VectorStore
from rag.strategies import get_strategy
from rag.strategies.base import StrategyContext
from rag.trace import Trace

_MAX_NAMED_EXCLUSIONS = 5
"""Above this many excluded documents, the exclusion note reports only the
count. Naming every one of a large exclusion set would make the trace less
readable than the count alone; naming a handful is exactly what lets a reader
see, at a glance, that a document the question named is among them."""

_semantic_routers: "weakref.WeakKeyDictionary[object, SemanticRouter]" = (
    weakref.WeakKeyDictionary()
)
"""One SemanticRouter per embedder, built the first time it is needed.

`SemanticRouter.__init__` embeds every exemplar question -- its own docstring
promises that happens "once at construction". Building a fresh router inside
every `ask()` call would re-embed the same handful of exemplars on every
question, silently breaking that promise. Keyed by the embedder object
(weakly, so a discarded embedder does not pin a router forever) rather than
by config, because the router's vectors are only valid for the embedder that
produced them.
"""


def _get_semantic_router(embedder) -> SemanticRouter:
    router = _semantic_routers.get(embedder)
    if router is None:
        router = SemanticRouter(embedder, PROMPT_EXEMPLARS)
        _semantic_routers[embedder] = router
    return router


INDEX_MODES = ("flat", "multirep", "raptor")
"""The index layouts this project can build.

Each gets its own file. Both new techniques change what goes *into* the
index rather than how it is searched, so mixing them into the flat index
would move every benchmark number for reasons unrelated to the strategies
being measured.
"""


def index_path_for(config: Config, mode: str) -> Path:
    """Where the index for this mode lives.

    Each mode gets its own file so the flat index -- which every benchmark
    number in the README rests on -- is never overwritten by a RAPTOR build.
    """
    if mode not in INDEX_MODES:
        raise ValueError(
            f"unknown index mode: {mode} (choose one of {', '.join(INDEX_MODES)})"
        )
    if mode == "flat":
        return config.index_path
    return config.index_path.with_name(
        f"{config.index_path.stem}-{mode}{config.index_path.suffix}"
    )


def build_index(config: Config, embedder, llm=None, mode: str = "flat") -> VectorStore:
    """Build and persist the index for `mode`.

    `multirep` and `raptor` both summarise with the LLM, so they need one;
    asking for them without it is a configuration error, not something to
    degrade around -- an index silently built without its summaries would be
    a flat index wearing the wrong name.
    """
    path = index_path_for(config, mode)
    documents = load_documents(config.corpus_dir, config.metadata_path)
    if mode != "flat" and llm is None:
        raise ValueError(f"index mode {mode} needs an llm to summarise with")

    if mode == "multirep":
        store = build_multi_representation(documents, embedder, llm)
    else:
        chunks = chunk_documents(
            documents, embedder.tokenizer, config.chunk_tokens, config.chunk_overlap
        )
        vectors = embedder.encode([chunk.text for chunk in chunks])
        if mode == "raptor":
            chunks, vectors = build_raptor(chunks, vectors, embedder, llm, config)
        store = VectorStore(vectors=vectors, chunks=chunks)
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

    store.meta = _index_meta(config, store.dim, mode)
    store.save(path, meta=store.meta)
    return store


def _index_meta(config: Config, dim: int | None, mode: str = "flat") -> dict:
    """What an index must agree with to be safely reused.

    Chunk size and overlap change which text each vector represents, and the
    embedding model changes what the vectors mean. Reusing an index across any
    of those returns plausible nonsense rather than an error, because the
    dimensions still line up. `index_mode` is the same argument one level up:
    a flat index loads cleanly against a config expecting RAPTOR, and the only
    symptom is worse answers.
    """
    meta = {
        "embedding_model": config.embedding_model,
        "chunk_tokens": config.chunk_tokens,
        "chunk_overlap": config.chunk_overlap,
        "dim": dim,
        "index_mode": mode,
    }
    if mode == "raptor":
        meta["raptor_max_depth"] = config.raptor_max_depth
        meta["raptor_cluster_size"] = config.raptor_cluster_size
    return meta


def load_index(config: Config, mode: str = "flat") -> VectorStore:
    # dim is unknown until the file is read, and an index whose dim differs
    # already fails cleanly at search time, so it is left out of the check.
    expected = _index_meta(config, dim=None, mode=mode)
    expected.pop("dim")
    return VectorStore.load(index_path_for(config, mode), expect_meta=expected)


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

        # Report what the filter excluded, unconditionally and structurally --
        # not by detecting that the question *names* an excluded document
        # (that needs entity matching against titles, the prose-heuristic
        # path Phase 4 deleted eight tests to escape). Counting is
        # deterministic, needs no LLM, and still puts the excluded document's
        # id on screen next to a question that names it, in the common case
        # where there are few enough to list.
        excluded_doc_ids = sorted(
            doc_id
            for doc_id, record in store.doc_meta.items()
            if not active.matches(record)
        )
        detail = (
            f": {', '.join(excluded_doc_ids)}"
            if 0 < len(excluded_doc_ids) <= _MAX_NAMED_EXCLUSIONS
            else ""
        )
        trace.note(
            f"filter ({active.describe()}) excluded {len(excluded_doc_ids)} of "
            f"{len(store.doc_meta)} documents{detail}"
        )

        if not mask.any():
            # A filter matching nothing must return no results rather than
            # silently falling back to unfiltered search, which would answer
            # a filtered question from the whole corpus.
            trace.note(
                f"filter matched no documents ({active.describe()}); "
                "returning no results"
            )

    prompt_name = None
    if semantic_prompt and llm is not None:
        # With no LLM there is no generation at all (see the `llm is None`
        # return below), so a prompt choice here would never be used --
        # tracing one anyway would show a routing decision that had no
        # effect. Skipping is the same choice already made for logical
        # routing when there is nothing to route between.
        prompt_name = _get_semantic_router(embedder).route(question, trace)

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
