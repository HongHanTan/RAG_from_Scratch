"""The gold set: questions paired with the text that answers them.

Each question names a document and quotes the passage that answers it,
verbatim. The quote is resolved to a character span when the gold set loads,
and spans are resolved to chunk ids at scoring time.

Storing a quote rather than raw offsets is what makes the file writable by
hand and checkable by eye — and loading validates that the quote occurs
exactly once, so a typo or an ambiguous phrase fails loudly instead of
silently scoring against the wrong passage.

The span survives re-chunking, which is the property that matters: chunk ids
shift whenever chunk size or overlap changes, so a chunk-id-based gold set
would break the first time retrieval is tuned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.loader import Document

REQUIRED_FIELDS = ("id", "question", "doc_id", "quotes", "why")


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    doc_id: str
    quotes: tuple[str, ...]
    why: str
    spans: tuple[tuple[int, int], ...]
    """Every passage that answers this question, as half-open char ranges.

    A question usually has more than one passage that answers it: a paper
    states its contribution in the abstract and then explains it properly in
    the body. Accepting only one makes a strategy that finds the better
    explanation score zero, which measures the gold set rather than the
    strategy.
    """


def load_gold(path: Path, documents: list[Document]) -> list[GoldQuestion]:
    """Load the gold set, resolving each quote to a character span."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw["questions"]
    by_id = {doc.doc_id: doc for doc in documents}

    gold: list[GoldQuestion] = []
    seen: set[str] = set()
    for entry in entries:
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(
                f"gold question {entry.get('id', '<no id>')} in {path} is "
                f"missing required field(s): {', '.join(missing)}"
            )
        question_id = entry["id"]
        if question_id in seen:
            raise ValueError(f"duplicate gold question id in {path}: {question_id}")
        seen.add(question_id)

        doc = by_id.get(entry["doc_id"])
        if doc is None:
            raise ValueError(
                f"gold question {question_id} names unknown doc_id: {entry['doc_id']}"
            )

        quotes = entry["quotes"]
        if isinstance(quotes, str) or not quotes:
            raise ValueError(
                f"gold question {question_id}: 'quotes' must be a non-empty "
                "list of strings"
            )
        spans = []
        for quote in quotes:
            if not isinstance(quote, str):
                raise ValueError(
                    f"gold question {question_id}: 'quotes' must be a list of "
                    f"strings, got {quote!r} ({type(quote).__name__})"
                )
            occurrences = doc.text.count(quote)
            if occurrences == 0:
                raise ValueError(
                    f"gold question {question_id}: quote not found in "
                    f"{doc.doc_id}: {quote[:60]!r}. It must match the document "
                    "text exactly, including whitespace."
                )
            if occurrences > 1:
                raise ValueError(
                    f"gold question {question_id}: quote is ambiguous, it "
                    f"appears {occurrences} times in {doc.doc_id}: "
                    f"{quote[:60]!r}. Extend it until unique."
                )
            start = doc.text.index(quote)
            spans.append((start, start + len(quote)))

        gold.append(
            GoldQuestion(
                id=question_id,
                question=entry["question"],
                doc_id=entry["doc_id"],
                quotes=tuple(quotes),
                why=entry["why"],
                spans=tuple(spans),
            )
        )
    return gold
