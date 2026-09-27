from rag.chunking import Chunk, RetrievedChunk
from rag.prompts import ANSWER_TEMPLATE, build_answer_prompt, format_context


def _retrieved():
    return [
        RetrievedChunk(
            Chunk("alpha:0", "alpha", 0, "Cosine ignores magnitude.", 0, 5, 0, 25),
            0.91,
            1,
        ),
        RetrievedChunk(
            Chunk("beta:2", "beta", 2, "RRF sums reciprocal ranks.", 0, 5, 0, 26),
            0.77,
            2,
        ),
    ]


def test_template_has_both_placeholders():
    assert "{context}" in ANSWER_TEMPLATE
    assert "{question}" in ANSWER_TEMPLATE


def test_template_instructs_the_model_to_stay_in_context():
    assert "context" in ANSWER_TEMPLATE.lower()


def test_context_numbers_chunks_from_one():
    context = format_context(_retrieved())
    assert "[1]" in context
    assert "[2]" in context
    assert "[0]" not in context


def test_context_includes_chunk_text():
    context = format_context(_retrieved())
    assert "Cosine ignores magnitude." in context
    assert "RRF sums reciprocal ranks." in context


def test_context_labels_the_source_document():
    context = format_context(_retrieved())
    assert "alpha" in context
    assert "beta" in context


def test_context_shows_the_score():
    assert "0.910" in format_context(_retrieved())


def test_context_labels_a_cosine_score_as_cosine():
    chunk = Chunk("a:0", "a", 0, "text", 0, 5, 0, 4)
    context = format_context([RetrievedChunk(chunk=chunk, score=0.552, rank=1)])
    assert "cosine 0.552" in context


def test_context_labels_a_fused_score_as_rrf():
    # After fusion the number is ~0.03, not a similarity. Labelling both
    # "score" makes fusion look like a catastrophic quality drop.
    chunk = Chunk("a:0", "a", 0, "text", 0, 5, 0, 4)
    retrieved = [
        RetrievedChunk(chunk=chunk, score=0.0328, rank=1, score_kind="rrf")
    ]
    context = format_context(retrieved)
    assert "rrf 0.033" in context
    assert "cosine" not in context


def test_empty_retrieval_produces_an_explicit_marker_not_a_blank():
    # A blank context makes the model hallucinate freely; saying so does not.
    assert format_context([]) == "(no documents retrieved)"


def test_prompt_contains_the_question_and_the_context():
    prompt = build_answer_prompt("What is RRF?", _retrieved())
    assert "What is RRF?" in prompt
    assert "RRF sums reciprocal ranks." in prompt


def test_prompt_has_no_unfilled_placeholders():
    prompt = build_answer_prompt("q", _retrieved())
    assert "{context}" not in prompt
    assert "{question}" not in prompt


def test_braces_in_chunk_text_do_not_break_formatting():
    chunks = [RetrievedChunk(Chunk("a:0", "a", 0, "code: {'k': 1}", 0, 5, 0, 14), 0.5, 1)]
    prompt = build_answer_prompt("q", chunks)
    assert "{'k': 1}" in prompt
