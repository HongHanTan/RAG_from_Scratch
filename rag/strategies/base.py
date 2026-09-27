"""The Strategy protocol and the context strategies work through.

A strategy turns one question into retrieved chunks. It never touches the
store or the embedder directly — it calls `ctx.search()`, which owns the embed
and search timing, so every strategy's trace is shaped the same way and the
CLI and benchmark can read them uniformly.

Strategies that need the LLM must degrade rather than fail: retrieval usually
still works without the rewrite, and an answer from plain retrieval beats no
answer. `degrade_to_direct` records why, because a silent downgrade would make
a strategy look like it ran when it did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag.chunking import RetrievedChunk
from rag.config import Config
from rag.store import VectorStore
from rag.trace import Trace


@dataclass
class StrategyContext:
    store: VectorStore
    embedder: object
    llm: object | None
    config: Config
    trace: Trace

    def search(self, queries: list[str], k: int) -> list[list[RetrievedChunk]]:
        """Embed queries and retrieve k chunks for each, timing both stages."""
        if not queries:
            return []
        with self.trace.stage("embed"):
            vectors = self.embedder.encode(queries)
        with self.trace.stage("search"):
            return self.store.search(vectors, k)


@dataclass
class StrategyResult:
    retrieved: list[RetrievedChunk]
    extra_context: str | None = None
    """Text the strategy produced that belongs in the answer prompt.

    Only decomposition uses it, to carry sub-question/sub-answer pairs. It is
    kept separate from `retrieved` because it is generated text, not a
    retrieved excerpt, and must not be presented to the model as a source.
    """


class Strategy(Protocol):
    name: str

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult: ...


def degrade_to_direct(
    question: str, ctx: StrategyContext, reason: str
) -> StrategyResult:
    """Fall back to plain retrieval, recording why on the trace."""
    ctx.trace.note(f"{reason}; degraded to direct retrieval")
    ctx.trace.queries = [question]
    return StrategyResult(retrieved=ctx.search([question], ctx.config.top_k)[0])
