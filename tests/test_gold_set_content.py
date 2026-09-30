"""Properties the gold set itself must hold.

Separate from test_gold.py, which tests the loader. This tests the data.
"""

from pathlib import Path

import pytest

from evaluation.gold import load_gold
from rag.__main__ import load_config
from rag.loader import load_documents

GOLD = Path(__file__).resolve().parent.parent / "evaluation" / "gold.json"


@pytest.fixture(scope="module")
def gold():
    config = load_config()
    return load_gold(GOLD, load_documents(config.corpus_dir, config.metadata_path))


def test_every_quote_resolves(gold):
    # load_gold raises on a quote that is missing or ambiguous, so reaching
    # here at all means all 30 resolved. This asserts the fixture ran.
    assert gold


def test_ids_are_unique(gold):
    ids = [q.id for q in gold]
    assert len(ids) == len(set(ids))


def test_every_question_has_a_reason(gold):
    assert all(q.why.strip() for q in gold)


def test_every_span_is_non_empty(gold):
    for q in gold:
        for span in q.spans:
            assert span.char_end > span.char_start, f"{q.id} has an empty span"


def test_there_are_cross_document_questions(gold):
    # Their absence is why Phase 5 could not judge RAPTOR fairly.
    multi = [q for q in gold if len(q.sources) > 1]
    assert len(multi) >= 10, f"only {len(multi)} cross-document questions"


def test_cross_document_questions_name_distinct_documents(gold):
    for q in gold:
        assert len(set(q.sources)) == len(q.sources), f"{q.id} repeats a document"


def test_new_questions_carry_a_prediction(gold):
    # Written before measuring. The original ten predate the practice and
    # are exempt by id, not by silence.
    original = {
        "colbert-maxsim", "raptor-clustering", "dpr-encoder", "selfrag-tokens",
        "hyde-problem", "rag-memory", "crag-quality", "longcontext-position",
        "colbertv2-tradeoff", "cot-limits",
    }
    missing = [q.id for q in gold if q.id not in original and not q.expects]
    assert not missing, f"no 'expects' prediction on: {missing}"
