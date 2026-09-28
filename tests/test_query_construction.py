import numpy as np
import pytest

from rag.chunking import Chunk
from rag.query_construction import MetadataFilter, compile_mask


def _chunks():
    return [
        Chunk("a:0", "a", 0, "t", 0, 5, 0, 1),
        Chunk("a:1", "a", 1, "t", 0, 5, 1, 2),
        Chunk("b:0", "b", 0, "t", 0, 5, 0, 1),
        Chunk("c:0", "c", 0, "t", 0, 5, 0, 1),
    ]


def _doc_meta():
    return {
        "a": {"topic": "retrieval-models", "author": "Khattab", "publish_date": "2020-04-27"},
        "b": {"topic": "rag-systems", "author": "Lewis", "publish_date": "2024-01-29"},
        "c": {"topic": "rag-systems", "author": "Sarthi", "publish_date": "2023-01-01"},
    }


# --- the filter itself -------------------------------------------------------

def test_an_empty_filter_is_empty():
    assert MetadataFilter().is_empty()


def test_a_filter_with_any_constraint_is_not_empty():
    assert not MetadataFilter(topics=("rag-systems",)).is_empty()
    assert not MetadataFilter(published_before="2024-01-01").is_empty()


def test_topic_matching():
    f = MetadataFilter(topics=("rag-systems",))
    assert f.matches({"topic": "rag-systems"})
    assert not f.matches({"topic": "foundations"})


def test_several_topics_are_an_or():
    f = MetadataFilter(topics=("rag-systems", "foundations"))
    assert f.matches({"topic": "foundations"})


def test_author_matching_is_case_insensitive_and_partial():
    # A question says "Khattab" where metadata says "Khattab and Zaharia".
    f = MetadataFilter(authors=("khattab",))
    assert f.matches({"author": "Khattab and Zaharia"})
    assert not f.matches({"author": "Lewis et al."})


def test_published_before_excludes_the_boundary_date():
    f = MetadataFilter(published_before="2024-01-01")
    assert f.matches({"publish_date": "2023-12-31"})
    assert not f.matches({"publish_date": "2024-01-01"})


def test_published_after_excludes_the_boundary_date():
    f = MetadataFilter(published_after="2023-12-31")
    assert f.matches({"publish_date": "2024-01-01"})
    assert not f.matches({"publish_date": "2023-12-31"})


def test_constraints_combine_with_and():
    f = MetadataFilter(topics=("rag-systems",), published_before="2024-01-01")
    assert f.matches({"topic": "rag-systems", "publish_date": "2023-01-01"})
    assert not f.matches({"topic": "rag-systems", "publish_date": "2024-06-01"})


def test_a_document_missing_the_constrained_field_does_not_match():
    # Silently keeping documents with no date would make a date filter
    # quietly weaker than it claims to be.
    assert not MetadataFilter(published_before="2024-01-01").matches({"topic": "x"})


def test_describe_is_human_readable():
    f = MetadataFilter(topics=("rag-systems",), published_before="2024-01-01")
    described = f.describe()
    assert "rag-systems" in described
    assert "2024-01-01" in described


def test_describe_of_an_empty_filter_says_so():
    assert "no filter" in MetadataFilter().describe().lower()


# --- mask compilation --------------------------------------------------------

def test_an_empty_filter_keeps_everything():
    mask = compile_mask(MetadataFilter(), _chunks(), _doc_meta())
    assert mask.dtype == bool
    assert mask.tolist() == [True, True, True, True]


def test_a_topic_filter_masks_by_document():
    mask = compile_mask(
        MetadataFilter(topics=("rag-systems",)), _chunks(), _doc_meta()
    )
    assert mask.tolist() == [False, False, True, True]


def test_every_chunk_of_a_matching_document_is_kept():
    mask = compile_mask(
        MetadataFilter(topics=("retrieval-models",)), _chunks(), _doc_meta()
    )
    assert mask.tolist() == [True, True, False, False]


def test_a_date_filter_masks_correctly():
    mask = compile_mask(
        MetadataFilter(published_before="2024-01-01"), _chunks(), _doc_meta()
    )
    assert mask.tolist() == [True, True, False, True]


def test_a_chunk_whose_document_has_no_metadata_is_excluded():
    chunks = _chunks() + [Chunk("z:0", "z", 0, "t", 0, 5, 0, 1)]
    mask = compile_mask(MetadataFilter(topics=("rag-systems",)), chunks, _doc_meta())
    assert mask[-1] == False  # noqa: E712


def test_a_filter_matching_nothing_gives_an_all_false_mask():
    mask = compile_mask(MetadataFilter(topics=("nope",)), _chunks(), _doc_meta())
    assert not mask.any()


def test_mask_length_matches_the_chunk_count():
    assert len(compile_mask(MetadataFilter(), _chunks(), _doc_meta())) == 4
