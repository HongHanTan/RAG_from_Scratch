"""The Trace: one record of everything a single question caused.

Three consumers read this object and nothing else — the CLI's --trace output,
the Phase 6 dashboard, and the Phase 3 benchmark. Keeping them behind one
representation is what stops the dashboard growing logic of its own.

Later phases add fields (rewritten queries, the inferred metadata filter, the
routing decision) without changing this shape.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Iterator

from rag.chunking import Chunk, RetrievedChunk

__all__ = ["StageTiming", "RetrievedChunk", "Trace"]


@dataclass
class StageTiming:
    name: str
    ms: float


@dataclass
class Trace:
    question: str
    queries: list[str] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    prompt: str | None = None
    answer: str | None = None
    timings: list[StageTiming] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and record it, whether or not the block raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.timings.append(StageTiming(name=name, ms=elapsed_ms))

    def note(self, message: str) -> None:
        """Record something the reader needs to know, such as a degradation."""
        self.notes.append(message)

    @property
    def total_ms(self) -> float:
        return sum(t.ms for t in self.timings)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "queries": list(self.queries),
            "retrieved": [
                {
                    "chunk": asdict(r.chunk),
                    "score": float(r.score),
                    "rank": int(r.rank),
                }
                for r in self.retrieved
            ],
            "prompt": self.prompt,
            "answer": self.answer,
            "timings": [{"name": t.name, "ms": t.ms} for t in self.timings],
            "notes": list(self.notes),
            "total_ms": self.total_ms,
        }
