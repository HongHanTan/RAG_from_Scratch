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


# --- build_filter -------------------------------------------------------

from rag.llm import LLMError
from rag.query_construction import build_filter
from rag.trace import Trace

TOPICS = ("retrieval-models", "rag-systems", "foundations")


class ReplyLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply

    def structured(self, prompt, schema):
        from rag.llm import GeminiLLM

        return GeminiLLM.structured(self, prompt, schema)


def test_build_filter_extracts_a_date_constraint():
    llm = ReplyLLM('{"published_before": "2024-01-01"}')
    f = build_filter("anything published before 2024?", llm, TOPICS, Trace(question="q"))
    assert f.published_before == "2024-01-01"


def test_build_filter_extracts_a_topic():
    llm = ReplyLLM('{"topics": ["rag-systems"]}')
    f = build_filter("what do the RAG papers say?", llm, TOPICS, Trace(question="q"))
    assert f.topics == ("rag-systems",)


def test_build_filter_drops_a_topic_that_does_not_exist():
    # A hallucinated collection would mask everything out and return nothing.
    llm = ReplyLLM('{"topics": ["rag-systems", "invented-topic"]}')
    f = build_filter("q", llm, TOPICS, Trace(question="q"))
    assert f.topics == ("rag-systems",)


def test_build_filter_degrades_when_every_proposed_topic_is_unknown():
    # When some proposed topics survive, the filter is merely smaller than
    # proposed -- not this case. Here every proposed topic is hallucinated,
    # so the topic constraint the model meant to add vanishes entirely: that
    # is the exact silent fallback the "degraded" contract exists to catch.
    trace = Trace(question="q")
    llm = ReplyLLM('{"topics": ["invented-one", "invented-two"]}')
    f = build_filter("q", llm, TOPICS, trace)
    assert f.topics == ()
    assert any("degraded" in n for n in trace.notes)


def test_build_filter_rejects_author_constraints_shorter_than_three_characters():
    # MetadataFilter.matches does case-insensitive substring matching, and
    # build_filter is the only producer of author constraints, so a one- or
    # two-character name would match almost any author field in the corpus.
    llm = ReplyLLM('{"authors": ["Li", "Khattab"]}')
    f = build_filter("q", llm, TOPICS, Trace(question="q"))
    assert f.authors == ("Khattab",)


def test_build_filter_returns_an_empty_filter_for_an_unconstrained_question():
    llm = ReplyLLM("{}")
    assert build_filter("how does ColBERT work?", llm, TOPICS, Trace(question="q")).is_empty()


def test_build_filter_records_the_filter_on_the_trace():
    trace = Trace(question="q")
    build_filter("before 2024?", ReplyLLM('{"published_before": "2024-01-01"}'), TOPICS, trace)
    assert any(s.kind == "filter" for s in trace.translation)


def test_build_filter_degrades_when_the_llm_fails():
    class Failing:
        def generate(self, prompt):
            raise LLMError("rate limited")

        def structured(self, prompt, schema):
            raise LLMError("rate limited")

    trace = Trace(question="q")
    assert build_filter("q", Failing(), TOPICS, trace).is_empty()
    assert any("degraded" in n for n in trace.notes)


def test_build_filter_degrades_on_malformed_json():
    trace = Trace(question="q")
    assert build_filter("q", ReplyLLM("not json"), TOPICS, trace).is_empty()
    assert any("degraded" in n for n in trace.notes)


def test_build_filter_ignores_a_malformed_date():
    # "2024" is not a date the mask can compare against ISO strings. The
    # model tried to constrain the answer and the constraint disappeared, so
    # that must show up on the trace -- not vanish with no note at all.
    trace = Trace(question="q")
    f = build_filter("q", ReplyLLM('{"published_before": "2024"}'), TOPICS, trace)
    assert f.published_before is None
    assert any("degraded" in n for n in trace.notes)


def test_build_filter_needs_no_llm_gracefully():
    trace = Trace(question="q")
    assert build_filter("q", None, TOPICS, trace).is_empty()
    assert any("degraded" in n for n in trace.notes)
