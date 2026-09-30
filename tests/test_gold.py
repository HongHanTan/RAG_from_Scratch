import json

import pytest

from evaluation.gold import GoldQuestion, GoldSpan, load_gold
from rag.loader import Document


def _docs():
    return [
        Document(
            doc_id="alpha",
            text="Cosine similarity ignores magnitude. It measures the angle only.",
            title="Alpha",
            source="test",
        ),
        Document(
            doc_id="beta",
            text="Rank fusion sums reciprocal ranks.",
            title="Beta",
            source="test",
        ),
    ]


def _write(tmp_path, questions):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return path


def test_loads_a_question_and_resolves_its_quote(tmp_path):
    path = _write(tmp_path, [
        {
            "id": "q1",
            "question": "Does cosine similarity care about length?",
            "doc_id": "alpha",
            "quotes": ["Cosine similarity ignores magnitude."],
            "why": "states the property directly",
        }
    ])
    gold = load_gold(path, _docs())
    assert len(gold) == 1
    assert isinstance(gold[0], GoldQuestion)
    assert gold[0].spans == (
        GoldSpan("alpha", 0, len("Cosine similarity ignores magnitude.")),
    )


def test_resolved_span_reproduces_the_quote(tmp_path):
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "q", "doc_id": "alpha",
            "quotes": ["measures the angle"], "why": "w",
        }
    ])
    gold = load_gold(path, _docs())
    span = gold[0].spans[0]
    assert span.doc_id == "alpha"
    assert _docs()[0].text[span.char_start:span.char_end] == "measures the angle"


def test_several_quotes_produce_several_spans(tmp_path):
    # A question usually has more than one passage that answers it: the
    # abstract states the contribution, the body explains it properly.
    # Accepting only one scores a strategy zero for finding the better
    # explanation of the same thing.
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "q", "doc_id": "alpha",
            "quotes": [
                "Cosine similarity ignores magnitude.",
                "measures the angle",
            ],
            "why": "w",
        }
    ])
    gold = load_gold(path, _docs())
    assert len(gold) == 1
    assert len(gold[0].spans) == 2


def test_a_quote_that_does_not_appear_is_an_error(tmp_path):
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "q", "doc_id": "alpha",
            "quotes": ["this text is not in the document"], "why": "w",
        }
    ])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_an_ambiguous_quote_is_an_error(tmp_path):
    # Two occurrences means the span is undetermined, and scoring would
    # silently use whichever came first.
    docs = [Document(doc_id="alpha", text="repeat. repeat.", title="A", source="t")]
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "q", "doc_id": "alpha",
            "quotes": ["repeat."], "why": "w",
        }
    ])
    with pytest.raises(ValueError, match="ambiguous"):
        load_gold(path, docs)


def test_an_unknown_doc_id_is_an_error(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "nope", "quotes": ["x"], "why": "w"}
    ])
    with pytest.raises(ValueError, match="nope"):
        load_gold(path, _docs())


def test_duplicate_question_ids_are_an_error(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "a", "doc_id": "alpha", "quotes": ["angle"], "why": "w"},
        {"id": "q1", "question": "b", "doc_id": "beta", "quotes": ["fusion"], "why": "w"},
    ])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_a_missing_required_field_is_an_error(tmp_path):
    path = _write(tmp_path, [{"id": "q1", "question": "q", "doc_id": "alpha"}])
    with pytest.raises(ValueError, match="quotes"):
        load_gold(path, _docs())


def test_an_empty_quotes_list_is_an_error(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha", "quotes": [], "why": "w"}
    ])
    with pytest.raises(ValueError, match="non-empty"):
        load_gold(path, _docs())


def test_a_bare_string_instead_of_a_list_is_an_error(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha", "quotes": "oops", "why": "w"}
    ])
    with pytest.raises(ValueError, match="list"):
        load_gold(path, _docs())


def test_a_non_string_quote_in_the_list_is_an_error(tmp_path):
    # doc.text.count(quote) raises a bare TypeError for a non-string element
    # ("count() argument 1 must be str, not int"), naming nothing -- every
    # other malformed-gold-set case raises a ValueError naming the question
    # id, and this one should too.
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "q", "doc_id": "alpha",
            "quotes": ["measures the angle", 42], "why": "w",
        }
    ])
    with pytest.raises(ValueError, match="q1") as excinfo:
        load_gold(path, _docs())
    assert "42" in str(excinfo.value)


def test_questions_load_in_file_order(tmp_path):
    path = _write(tmp_path, [
        {"id": "q2", "question": "b", "doc_id": "beta", "quotes": ["fusion"], "why": "w"},
        {"id": "q1", "question": "a", "doc_id": "alpha", "quotes": ["angle"], "why": "w"},
    ])
    assert [g.id for g in load_gold(path, _docs())] == ["q2", "q1"]


def test_a_question_may_name_several_documents(tmp_path):
    # RAPTOR's whole claim is about answers that span documents. The old
    # schema could not state such a question, so the Phase 5 gold set
    # contained none and RAPTOR was judged only on cases it is not for.
    docs = _docs()
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "what do these share?",
            "sources": [
                {"doc_id": "alpha", "quotes": ["measures the angle"]},
                {"doc_id": "beta", "quotes": ["Rank fusion sums"]},
            ],
            "why": "spans two documents",
        }
    ])
    gold = load_gold(path, docs)
    assert gold[0].sources == ("alpha", "beta")
    assert {s.doc_id for s in gold[0].spans} == {"alpha", "beta"}


def test_each_span_reproduces_its_own_quote(tmp_path):
    docs = _docs()
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "q",
            "sources": [
                {"doc_id": "alpha", "quotes": ["measures the angle"]},
                {"doc_id": "beta", "quotes": ["Rank fusion sums"]},
            ],
            "why": "w",
        }
    ])
    by_id = {d.doc_id: d for d in docs}
    for span in load_gold(path, docs)[0].spans:
        assert by_id[span.doc_id].text[span.char_start:span.char_end] in (
            "measures the angle", "Rank fusion sums"
        )


def test_the_old_single_document_form_still_loads(tmp_path):
    # Every existing question uses it, and rewriting them all would risk
    # changing the numbers they produce.
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha",
         "quotes": ["measures the angle"], "why": "w"}
    ])
    gold = load_gold(path, _docs())
    assert gold[0].sources == ("alpha",)
    assert gold[0].doc_id == "alpha"


def test_doc_id_returns_the_first_source(tmp_path):
    # Kept so single-document call sites, and routing_eval's topic lookup,
    # keep working unchanged.
    path = _write(tmp_path, [
        {"id": "q1", "question": "q",
         "sources": [{"doc_id": "beta", "quotes": ["Rank fusion sums"]},
                     {"doc_id": "alpha", "quotes": ["measures the angle"]}],
         "why": "w"}
    ])
    assert load_gold(path, _docs())[0].doc_id == "beta"


def test_mixing_both_forms_in_one_question_is_an_error(tmp_path):
    # Ambiguous: which is the gold document? Better to refuse than guess.
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha", "quotes": ["angle"],
         "sources": [{"doc_id": "beta", "quotes": ["Rank fusion sums"]}],
         "why": "w"}
    ])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_an_empty_sources_list_is_an_error(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "sources": [], "why": "w"}
    ])
    with pytest.raises(ValueError, match="non-empty"):
        load_gold(path, _docs())


def test_a_bad_quote_in_the_second_source_names_the_question(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q",
         "sources": [{"doc_id": "alpha", "quotes": ["measures the angle"]},
                     {"doc_id": "beta", "quotes": ["not present at all"]}],
         "why": "w"}
    ])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_expects_is_loaded_when_present(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha",
         "quotes": ["measures the angle"], "why": "w", "expects": "raptor"}
    ])
    assert load_gold(path, _docs())[0].expects == "raptor"


def test_expects_defaults_to_empty(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha",
         "quotes": ["measures the angle"], "why": "w"}
    ])
    assert load_gold(path, _docs())[0].expects == ""
