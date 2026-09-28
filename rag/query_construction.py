"""PDF Stage 3: turn a natural-language constraint into a metadata filter.

"anything published before 2024" becomes `published_before="2024-01-01"`,
which becomes a NumPy boolean mask over chunks, applied *before* top-k so
that k results come back — rather than filtering the top k afterwards and
returning however many survived.

Dates are ISO `YYYY-MM-DD` strings throughout, so they compare correctly as
strings and nothing needs parsing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.chunking import Chunk


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
    """A boolean mask over chunks, True where the chunk's document matches."""
    if filter_.is_empty():
        return np.ones(len(chunks), dtype=bool)
    allowed = {
        doc_id for doc_id, record in doc_meta.items() if filter_.matches(record)
    }
    return np.fromiter(
        (chunk.doc_id in allowed for chunk in chunks), dtype=bool, count=len(chunks)
    )
