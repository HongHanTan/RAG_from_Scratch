import pytest

from rag.chunking import Chunk, chunk_document, chunk_documents, window_bounds
from rag.loader import Document
from tests.conftest import FakeTokenizer


# --- pure window arithmetic -------------------------------------------------

def test_windows_advance_by_size_minus_overlap():
    assert window_bounds(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]


def test_final_window_is_clipped_not_padded():
    assert window_bounds(9, 4, 1) == [(0, 4), (3, 7), (6, 9)]


def test_single_window_when_document_is_shorter_than_the_window():
    assert window_bounds(3, 4, 1) == [(0, 3)]


def test_exact_fit_produces_one_window():
    assert window_bounds(4, 4, 1) == [(0, 4)]


def test_no_windows_for_empty_input():
    assert window_bounds(0, 4, 1) == []


def test_no_trailing_window_that_is_wholly_covered_by_its_predecessor():
    # With size=4, overlap=1, n=10: a naive range() would emit (9, 10), whose
    # single token already appears in (6, 10).
    bounds = window_bounds(10, 4, 1)
    assert bounds[-1] == (6, 10)
    assert all(end <= 10 for _, end in bounds)


def test_zero_overlap_produces_disjoint_windows():
    assert window_bounds(9, 3, 0) == [(0, 3), (3, 6), (6, 9)]


def test_overlap_equal_to_size_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        window_bounds(10, 4, 4)


def test_nonpositive_size_is_rejected():
    with pytest.raises(ValueError, match="size"):
        window_bounds(10, 0, 0)


def test_every_token_appears_in_at_least_one_window():
    bounds = window_bounds(37, 8, 3)
    covered = {i for start, end in bounds for i in range(start, end)}
    assert covered == set(range(37))


# --- chunking a document ----------------------------------------------------

def _doc(text: str) -> Document:
    return Document(doc_id="d", text=text, title="T", source="s")


def test_chunk_text_is_sliced_from_the_original_not_detokenized():
    doc = _doc("alpha beta gamma delta epsilon")
    chunks = chunk_document(doc, FakeTokenizer(), size=2, overlap=0)
    assert chunks[0].text == "alpha beta"
    assert doc.text[chunks[0].char_start:chunks[0].char_end] == chunks[0].text


def test_char_offsets_round_trip_for_every_chunk():
    doc = _doc(" ".join(f"word{i}" for i in range(30)))
    for chunk in chunk_document(doc, FakeTokenizer(), size=7, overlap=2):
        assert doc.text[chunk.char_start:chunk.char_end] == chunk.text


def test_chunk_ids_are_doc_id_colon_index():
    doc = _doc("a b c d e f")
    chunks = chunk_document(doc, FakeTokenizer(), size=2, overlap=0)
    assert [c.chunk_id for c in chunks] == ["d:0", "d:1", "d:2"]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_token_bounds_are_recorded():
    doc = _doc("a b c d e")
    chunks = chunk_document(doc, FakeTokenizer(), size=3, overlap=1)
    assert (chunks[0].token_start, chunks[0].token_end) == (0, 3)
    assert (chunks[1].token_start, chunks[1].token_end) == (2, 5)


def test_empty_document_yields_no_chunks():
    assert chunk_document(_doc(""), FakeTokenizer(), size=4, overlap=1) == []


def test_slow_tokenizer_is_rejected_because_offsets_are_required():
    class SlowTokenizer:
        is_fast = False

        def __call__(self, *a, **kw):
            raise AssertionError("should not be called")

    with pytest.raises(ValueError, match="fast tokenizer"):
        chunk_document(_doc("a b"), SlowTokenizer(), size=2, overlap=0)


def test_chunk_documents_concatenates_in_document_order():
    docs = [_doc("a b c"), Document(doc_id="e", text="x y z", title="T", source="s")]
    chunks = chunk_documents(docs, FakeTokenizer(), size=2, overlap=0)
    assert [c.doc_id for c in chunks] == ["d", "d", "e", "e"]


def test_chunks_are_returned_as_the_dataclass():
    chunks = chunk_document(_doc("a b"), FakeTokenizer(), size=2, overlap=0)
    assert isinstance(chunks[0], Chunk)
