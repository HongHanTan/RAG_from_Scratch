"""Wiring. This module composes the others and owns no algorithm of its own.

Later phases insert routing, query construction and translation between the
question arriving and the search running; the Trace shape does not change.
"""

from __future__ import annotations

from rag.chunking import chunk_documents
from rag.config import Config
from rag.generation import generate_answer
from rag.loader import load_documents
from rag.store import VectorStore
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
) -> Trace:
    """Answer one question. Pass llm=None to retrieve without generating."""
    trace = Trace(question=question)
    trace.queries = [question]
    k = config.top_k if k is None else k

    with trace.stage("embed"):
        query_vectors = embedder.encode([question])

    with trace.stage("search"):
        results = store.search(query_vectors, k)[0]

    trace.retrieved = results

    if llm is None:
        trace.note("retrieval only: no LLM configured")
        return trace

    generate_answer(llm, question, results, trace)
    return trace
