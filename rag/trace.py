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

__all__ = ["DEGRADED", "StageTiming", "RetrievedChunk", "Trace", "TranslationStep"]

DEGRADED = "degraded"
"""The sentinel `evaluation/benchmark.py` hard-fails on.

Every silent fallback in `rag/` must go through `Trace.degraded`, which
writes this word, rather than a bare `Trace.note` call an author might
forget to word correctly. `tests/test_degradation_notes.py` enforces that
mechanically by walking the source for `.note(` calls on a trace and
failing unless the call site is on an explicit, justified allowlist."""


@dataclass
class StageTiming:
    name: str
    ms: float
    depth: int = 0


@dataclass(frozen=True)
class TranslationStep:
    """One artifact a translation strategy produced, for display.

    `kind` is one of "query", "step_back", "hypothetical", "sub_question",
    "sub_answer". Keeping this generic means a new strategy does not need a
    new Trace field.
    """

    kind: str
    text: str


@dataclass
class Trace:
    question: str
    queries: list[str] = field(default_factory=list)
    strategy: str = "direct"
    translation: list[TranslationStep] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    prompt: str | None = None
    answer: str | None = None
    timings: list[StageTiming] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _depth: int = field(default=0, repr=False, compare=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and record it, whether or not the block raises.

        Nested stages record their depth so total_ms can count only the
        outermost ones — an outer stage already contains its children's time.
        """
        started = time.perf_counter()
        depth = self._depth
        self._depth = depth + 1
        try:
            yield
        finally:
            self._depth = depth
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.timings.append(StageTiming(name=name, ms=elapsed_ms, depth=depth))

    def note(self, message: str) -> None:
        """Record something the reader needs to know.

        For a fact about the run (retrieval-only, cache hit, nothing
        matched). For a fallback, use `degraded` instead: the benchmark
        hard-fails on that sentinel, and this method does not write it.
        """
        self.notes.append(message)

    def degraded(self, what: str, fallback: str) -> None:
        """Record that something failed and the pipeline fell back.

        Distinct from `note`, which records a fact for the reader. This
        writes the sentinel the benchmark hard-fails on, so a silent
        fallback cannot be measured as "the technique does not help" — a
        failure this project hit for real in Phase 2. The word cannot be
        forgotten because the method writes it.
        """
        self.note(f"{what}; {DEGRADED} to {fallback}")

    def add_translation(self, kind: str, text: str) -> None:
        """Record something a translation strategy produced."""
        self.translation.append(TranslationStep(kind=kind, text=text))

    @property
    def total_ms(self) -> float:
        """Wall time, counting each top-level stage once.

        Nested stages are excluded because their time is already inside the
        stage that contains them.
        """
        return sum(t.ms for t in self.timings if t.depth == 0)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "queries": list(self.queries),
            "strategy": self.strategy,
            "translation": [
                {"kind": s.kind, "text": s.text} for s in self.translation
            ],
            "retrieved": [
                {
                    "chunk": asdict(r.chunk),
                    "score": float(r.score),
                    "rank": int(r.rank),
                    "score_kind": r.score_kind,
                }
                for r in self.retrieved
            ],
            "prompt": self.prompt,
            "answer": self.answer,
            "timings": [
                {"name": t.name, "ms": t.ms, "depth": t.depth} for t in self.timings
            ],
            "notes": list(self.notes),
            "total_ms": self.total_ms,
        }
