from evaluation.gold import GoldQuestion
from evaluation.spans import chunks_overlapping, relevant_chunk_ids
from rag.chunking import Chunk


def _chunk(chunk_id, doc_id, start, end):
    return Chunk(chunk_id, doc_id, 0, "text", 0, 10, start, end)


def _chunks():
    return [
        _chunk("a:0", "alpha", 0, 100),
        _chunk("a:1", "alpha", 80, 180),
        _chunk("a:2", "alpha", 160, 260),
        _chunk("b:0", "beta", 0, 100),
    ]


def test_a_span_inside_one_chunk_matches_only_it():
    assert chunks_overlapping("alpha", 10, 50, _chunks()) == {"a:0"}


def test_a_span_straddling_a_boundary_matches_both_chunks():
    # Chunks overlap by design, so an answering sentence routinely spans two.
    # Requiring containment would score a chunk holding most of the answer
    # as a miss.
    assert chunks_overlapping("alpha", 90, 120, _chunks()) == {"a:0", "a:1"}


def test_a_span_covering_three_chunks_matches_all_three():
    assert chunks_overlapping("alpha", 10, 250, _chunks()) == {"a:0", "a:1", "a:2"}


def test_chunks_from_other_documents_never_match():
    assert "b:0" not in chunks_overlapping("alpha", 0, 300, _chunks())


def test_a_span_touching_only_the_exclusive_end_does_not_match():
    # char_end is exclusive, so a span starting exactly at a chunk's end
    # shares no characters with it.
    assert chunks_overlapping("alpha", 100, 110, _chunks()) == {"a:1"}


def test_a_span_in_an_unindexed_document_matches_nothing():
    assert chunks_overlapping("gamma", 0, 50, _chunks()) == set()


def test_an_empty_span_matches_nothing():
    assert chunks_overlapping("alpha", 50, 50, _chunks()) == set()


def test_relevant_chunk_ids_uses_the_questions_span():
    question = GoldQuestion(
        id="q1", question="q", doc_id="alpha", quotes=("x",), why="w",
        spans=((90, 120),),
    )
    assert relevant_chunk_ids(question, _chunks()) == {"a:0", "a:1"}


def test_relevant_chunk_ids_unions_every_span():
    question = GoldQuestion(
        id="q1", question="q", doc_id="alpha", quotes=("x", "y"), why="w",
        spans=((10, 20), (200, 210)),
    )
    assert relevant_chunk_ids(question, _chunks()) == {"a:0", "a:2"}
