"""PDF Stage 3: turn a natural-language constraint into a metadata filter.

"anything published before 2024" becomes `published_before="2024-01-01"`,
which becomes a NumPy boolean mask over chunks, applied *before* top-k so
that k results come back — rather than filtering the top k afterwards and
returning however many survived.

Dates are ISO `YYYY-MM-DD` strings throughout, so they compare correctly as
strings and nothing needs parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rag.chunking import Chunk
from rag.llm import LLMError
from rag.prompts import FILTER_SCHEMA, FILTER_TEMPLATE

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MIN_AUTHOR_LENGTH = 3
"""Shortest author constraint `build_filter` will accept.

`MetadataFilter.matches` does case-insensitive substring matching on authors,
and this is the only producer of author constraints, so a one- or
two-character name would match almost any author field in the corpus."""


@dataclass(frozen=True)
class MetadataFilter:
    topics: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    published_before: str | None = None
    published_after: str | None = None

    def is_empty(self) -> bool:
        return not (
            self.topics or self.authors or self.published_before or self.published_after
        )

    def describe(self) -> str:
        """A one-line description, for the trace and the CLI."""
        parts: list[str] = []
        if self.topics:
            parts.append(f"topic in {', '.join(self.topics)}")
        if self.authors:
            parts.append(f"author matches {', '.join(self.authors)}")
        if self.published_before:
            parts.append(f"published before {self.published_before}")
        if self.published_after:
            parts.append(f"published after {self.published_after}")
        return "; ".join(parts) if parts else "no filter"

    def matches(self, record: dict) -> bool:
        """Whether one document's metadata satisfies every constraint.

        A document missing a constrained field does not match. Keeping it
        would make the filter quietly weaker than it says it is.
        """
        if self.topics and record.get("topic") not in self.topics:
            return False
        if self.authors:
            author = (record.get("author") or "").lower()
            if not any(wanted.lower() in author for wanted in self.authors):
                return False
        if self.published_before:
            date = record.get("publish_date")
            if not date or date >= self.published_before:
                return False
        if self.published_after:
            date = record.get("publish_date")
            if not date or date <= self.published_after:
                return False
        return True


def compile_mask(
    filter_: MetadataFilter, chunks: list[Chunk], doc_meta: dict
) -> np.ndarray:
    """A boolean mask over chunks, True where the chunk's document matches.

    Summary nodes follow from the existing rule rather than a special case. A
    multi-representation summary carries its document's real `doc_id`, so it
    inherits that document's filters. A RAPTOR cluster summary spans several
    documents, has a synthetic `doc_id` absent from `doc_meta`, and is
    therefore excluded by any active filter — which means a filtered search
    against a RAPTOR index loses its abstraction levels and falls back to
    raw chunks. That is the conservative choice: a summary of documents that
    mostly fail the filter should not survive it.
    """
    if filter_.is_empty():
        return np.ones(len(chunks), dtype=bool)
    allowed = {
        doc_id for doc_id, record in doc_meta.items() if filter_.matches(record)
    }
    return np.fromiter(
        (chunk.doc_id in allowed for chunk in chunks), dtype=bool, count=len(chunks)
    )


def build_filter(question: str, llm, topics: tuple[str, ...], trace) -> MetadataFilter:
    """Infer a metadata filter from the question, or an empty one.

    Every failure path returns an empty filter and records a note containing
    "degraded": a filter is an optimisation, and losing it costs precision,
    while wrongly applying one can mask the answer out entirely.
    """
    if llm is None:
        trace.degraded("query construction needs an LLM", "no filter")
        return MetadataFilter()

    prompt = FILTER_TEMPLATE.format(question=question, topics=", ".join(topics))
    try:
        with trace.stage("construct"):
            parsed = llm.structured(prompt, FILTER_SCHEMA)
    except LLMError as exc:
        trace.degraded(f"query construction failed: {exc}", "no filter")
        return MetadataFilter()

    # A hallucinated topic would mask out the whole corpus, so drop unknowns.
    chosen = tuple(t for t in parsed.get("topics", []) if t in topics)
    dropped = [t for t in parsed.get("topics", []) if t not in topics]
    if dropped and not chosen:
        # Every topic the model proposed was unknown: the topic constraint it
        # meant to add has silently vanished rather than merely shrunk.
        trace.degraded(
            f"query construction proposed unknown topics: {', '.join(dropped)}",
            "no topic filter",
        )
    elif dropped:
        # Some proposed topics were valid and are still applied here; this is
        # visibility into what was dropped, not a fallback to a safe default.
        trace.note(f"query construction proposed unknown topics: {', '.join(dropped)}")

    def _date(key: str) -> str | None:
        value = parsed.get(key)
        if value is None:
            return None
        if isinstance(value, str) and ISO_DATE.match(value):
            return value
        # The model tried to constrain on this field and produced something
        # that cannot be compared as an ISO date; the constraint disappears
        # silently unless this is recorded.
        trace.degraded(
            f"query construction produced a malformed {key}: {value!r}",
            f"no {key} constraint",
        )
        return None

    filter_ = MetadataFilter(
        topics=chosen,
        authors=tuple(
            a
            for a in parsed.get("authors", [])
            if isinstance(a, str) and len(a) >= MIN_AUTHOR_LENGTH
        ),
        published_before=_date("published_before"),
        published_after=_date("published_after"),
    )
    trace.add_translation("filter", filter_.describe())
    return filter_
