import json

import pytest

from evaluation.gold import GoldQuestion, load_gold
from rag.loader import Document


def _docs():
    return [
        Document(
            doc_id="alpha",
            text="Cosine similarity ignores magnitude. It measures the angle only.",
            title="Alpha",
            source="test",
        ),
        Document(doc_id="beta", text="Rank fusion sums reciprocal ranks.", title="Beta", source="test"),
    ]


def _write(tmp_path, questions):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return path


def test_loads_a_question_and_resolves_its_quote(tmp_path):
    path = _write(tmp_path, [{
        "id": "q1",
        "question": "Does cosine similarity care about length?",
        "doc_id": "alpha",
        "quote": "Cosine similarity ignores magnitude.",
        "why": "states the property directly",
    }])
    gold = load_gold(path, _docs())
    assert len(gold) == 1
    assert isinstance(gold[0], GoldQuestion)
    assert gold[0].char_start == 0
    assert gold[0].char_end == len("Cosine similarity ignores magnitude.")


def test_resolved_span_reproduces_the_quote(tmp_path):
    path = _write(tmp_path, [{
        "id": "q1", "question": "q", "doc_id": "alpha",
        "quote": "measures the angle", "why": "w",
    }])
    gold = load_gold(path, _docs())
    text = _docs()[0].text
    assert text[gold[0].char_start:gold[0].char_end] == "measures the angle"


def test_a_quote_that_does_not_appear_is_an_error(tmp_path):
    path = _write(tmp_path, [{
        "id": "q1", "question": "q", "doc_id": "alpha",
        "quote": "this text is not in the document", "why": "w",
    }])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_an_ambiguous_quote_is_an_error(tmp_path):
    # Two occurrences means the span is undetermined, and scoring would
    # silently use whichever came first.
    docs = [Document(doc_id="alpha", text="repeat. repeat.", title="A", source="t")]
    path = _write(tmp_path, [{
        "id": "q1", "question": "q", "doc_id": "alpha", "quote": "repeat.", "why": "w",
    }])
    with pytest.raises(ValueError, match="twice|2 times|ambiguous"):
        load_gold(path, docs)


def test_an_unknown_doc_id_is_an_error(tmp_path):
    path = _write(tmp_path, [{
        "id": "q1", "question": "q", "doc_id": "nope", "quote": "x", "why": "w",
    }])
    with pytest.raises(ValueError, match="nope"):
        load_gold(path, _docs())


def test_duplicate_question_ids_are_an_error(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "a", "doc_id": "alpha", "quote": "angle", "why": "w"},
        {"id": "q1", "question": "b", "doc_id": "beta", "quote": "fusion", "why": "w"},
    ])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_a_missing_required_field_is_an_error(tmp_path):
    path = _write(tmp_path, [{"id": "q1", "question": "q", "doc_id": "alpha"}])
    with pytest.raises(ValueError, match="quote"):
        load_gold(path, _docs())


def test_questions_load_in_file_order(tmp_path):
    path = _write(tmp_path, [
        {"id": "q2", "question": "b", "doc_id": "beta", "quote": "fusion", "why": "w"},
        {"id": "q1", "question": "a", "doc_id": "alpha", "quote": "angle", "why": "w"},
    ])
    assert [g.id for g in load_gold(path, _docs())] == ["q2", "q1"]
