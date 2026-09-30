"""The gold set: questions paired with the text that answers them.

Each question names one or more documents and quotes the passages that
answer it, verbatim. Every quote is resolved to a character span when the
gold set loads, and spans are resolved to chunk ids at scoring time.

Storing a quote rather than raw offsets is what makes the file writable by
hand and checkable by eye — and loading validates that the quote occurs
exactly once, so a typo or an ambiguous phrase fails loudly instead of
silently scoring against the wrong passage.

The span survives re-chunking, which is the property that matters: chunk ids
shift whenever chunk size or overlap changes, so a chunk-id-based gold set
would break the first time retrieval is tuned.

Two spellings of a question are accepted. The original names one document:

    {"id": ..., "question": ..., "doc_id": "colbert", "quotes": [...],
     "why": ...}

The general one names several, each with its own quotes:

    {"id": ..., "question": ...,
     "sources": [{"doc_id": "colbert", "quotes": [...]},
                 {"doc_id": "colbertv2", "quotes": [...]}],
     "why": ...}

Both are kept because every question written before Phase 7 uses the first,
and rewriting them would risk moving the numbers they produce — the one
thing the generalisation must not do. Using both in a single question is an
error rather than a guess about which document is the gold one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.loader import Document

REQUIRED_FIELDS = ("id", "question")
"""Checked before anything else, because the id is what every later error
names and the question is what is being asked. `why` and the document fields
are checked once the entry's spelling is known -- which of `doc_id`/`quotes`
or `sources` is required depends on it.
"""


@dataclass(frozen=True)
class GoldSpan:
    """One answering passage, and the document it lives in.

    The document travels with the span because a question may now have
    answers in several papers, and a bare (start, end) pair would silently
    credit a chunk at the same offset in the wrong one.
    """

    doc_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    sources: tuple[str, ...]
    """Every document holding an answer, in the order the file lists them."""
    quotes: tuple[str, ...]
    why: str
    spans: tuple[GoldSpan, ...]
    """Every passage that answers this question, as half-open char ranges.

    A question usually has more than one passage that answers it: a paper
    states its contribution in the abstract and then explains it properly in
    the body. Accepting only one makes a strategy that finds the better
    explanation score zero, which measures the gold set rather than the
    strategy.
    """
    expects: str = ""
    """The technique this question was predicted to favour, recorded when it
    was written and before anything was measured. Empty for the original ten,
    which predate the practice. Task 4 scores these predictions: a gold set
    written by someone who knows what each technique does is a gold set that
    can be fitted to the answer, and reporting how often the prediction held
    is what makes that risk visible instead of hidden.
    """

    @property
    def doc_id(self) -> str:
        """The first gold document.

        Kept so single-document call sites -- `routing_eval`'s topic lookup
        among them -- keep working without a sweep, and so the ten original
        questions behave exactly as before.
        """
        return self.sources[0]


def _resolve_quote(question_id: str, doc: Document, quote: str) -> GoldSpan:
    """Find `quote` in `doc`, insisting it occurs exactly once."""
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
    return GoldSpan(doc.doc_id, start, start + len(quote))


def _sources_of(question_id: str, entry: dict) -> list[dict]:
    """The entry's sources, in either spelling, as a uniform list.

    Refuses a question that uses both spellings: which document is the gold
    one would otherwise be a guess, and guessing here is the kind of thing
    that quietly scores against the wrong paper.
    """
    if "sources" in entry and ("doc_id" in entry or "quotes" in entry):
        raise ValueError(
            f"gold question {question_id}: use either 'sources' or "
            "'doc_id'/'quotes', not both -- which document is the gold one "
            "is otherwise ambiguous"
        )
    if "sources" in entry:
        sources = entry["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError(
                f"gold question {question_id}: 'sources' must be a non-empty list"
            )
        return sources
    missing = [f for f in ("doc_id", "quotes") if f not in entry]
    if missing:
        raise ValueError(
            f"gold question {question_id} is missing required field(s): "
            f"{', '.join(missing)}"
        )
    return [{"doc_id": entry["doc_id"], "quotes": entry["quotes"]}]


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

        sources = _sources_of(question_id, entry)
        if "why" not in entry:
            raise ValueError(
                f"gold question {question_id} in {path} is missing required "
                "field(s): why"
            )

        source_ids: list[str] = []
        quotes: list[str] = []
        spans: list[GoldSpan] = []
        for source in sources:
            if not isinstance(source, dict) or "doc_id" not in source:
                raise ValueError(
                    f"gold question {question_id}: every entry in 'sources' "
                    f"must be an object with a 'doc_id', got {source!r}"
                )
            doc = by_id.get(source["doc_id"])
            if doc is None:
                raise ValueError(
                    f"gold question {question_id} names unknown doc_id: "
                    f"{source['doc_id']}"
                )
            source_quotes = source.get("quotes")
            if isinstance(source_quotes, str) or not source_quotes:
                raise ValueError(
                    f"gold question {question_id}: 'quotes' must be a "
                    f"non-empty list of strings (in source {doc.doc_id})"
                )
            source_ids.append(doc.doc_id)
            for quote in source_quotes:
                spans.append(_resolve_quote(question_id, doc, quote))
                quotes.append(quote)

        gold.append(
            GoldQuestion(
                id=question_id,
                question=entry["question"],
                sources=tuple(source_ids),
                quotes=tuple(quotes),
                why=entry["why"],
                spans=tuple(spans),
                expects=entry.get("expects", ""),
            )
        )
    return gold
