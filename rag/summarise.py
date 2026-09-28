"""Summaries for the two indexing techniques.

Both go through the LLM's on-disk cache, so rebuilding an index after the
first run costs nothing — which matters when a full RAPTOR tree over 5,116
chunks is several hundred calls.
"""

from __future__ import annotations

from rag.llm import LLMError


class SummaryError(Exception):
    """Raised when a summary could not be produced."""


DOC_SUMMARY_TEMPLATE = """Summarise the document below in three or four sentences.

Say what it proposes or reports, the problem it addresses, and the terms a
reader would search for to find it. Write plainly, as documentation rather
than as a review. Do not begin with "This document" or "This paper".

Document:
{text}

Summary:"""


CLUSTER_SUMMARY_TEMPLATE = """Summarise what the passages below have in common.

They were grouped because their embeddings are close, so they share a subject
even though they come from different places. Describe the shared theme and the
specific ideas that recur, in three or four sentences. Do not list the
passages separately or number them — a summary that just concatenates them is
useless as an abstraction.

Passages:
{text}

Shared summary:"""


def summarise(text: str, llm, template: str, max_chars: int = 12000) -> str:
    """Summarise `text` with `llm`, truncating over-long input.

    Truncation keeps the beginning: a paper states its contribution in its
    opening, so the front is the most summarisable part, and a 200,000
    character document would otherwise exceed the context window.
    """
    excerpt = text[:max_chars]
    try:
        summary = llm.generate(template.format(text=excerpt)).strip()
    except LLMError as exc:
        raise SummaryError(f"summarisation failed: {exc}") from exc
    if not summary:
        raise SummaryError("model returned an empty summary")
    return summary
