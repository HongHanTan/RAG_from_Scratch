import pytest

from rag.chunking import RetrievedChunk
from rag.indexing.multi_representation import (
    build_multi_representation,
    expand_to_documents,
)
from rag.loader import Document
from rag.summarise import SummaryError
from rag.trace import Trace
from tests.conftest import FakeEmbedder


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "fallback summary"


def _docs():
    return [
        Document("alpha", "Alpha body text about cosine similarity.", "Alpha", "arxiv", topic="t"),
        Document("beta", "Beta body text about rank fusion.", "Beta", "arxiv", topic="t"),
    ]


def test_one_node_per_document():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert len(store) == 2
    assert {c.doc_id for c in store.chunks} == {"alpha", "beta"}


def test_nodes_hold_the_summary_not_the_document():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["SUMMARY A", "SUMMARY B"]))
    texts = {c.text for c in store.chunks}
    assert texts == {"SUMMARY A", "SUMMARY B"}


def test_nodes_are_synthetic_at_level_one():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert all(c.is_synthetic and c.level == 1 for c in store.chunks)


def test_nodes_keep_their_documents_id_so_filters_still_work():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert sorted(c.doc_id for c in store.chunks) == ["alpha", "beta"]


def test_the_docstore_holds_the_full_documents():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert store.docstore["alpha"] == "Alpha body text about cosine similarity."


def test_document_metadata_is_carried_over():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert store.doc_meta["alpha"]["title"] == "Alpha"


def test_a_document_whose_summary_fails_is_skipped_and_noted():
    class OneFails:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                from rag.llm import LLMError

                raise LLMError("rate limited")
            return "s2"

    trace = Trace(question="build")
    store = build_multi_representation(_docs(), FakeEmbedder(), OneFails(), trace)
    assert len(store) == 1
    assert any("degraded" in n for n in trace.notes)


def test_every_summary_failing_raises_rather_than_returning_an_empty_index():
    class AllFail:
        def generate(self, prompt):
            from rag.llm import LLMError

            raise LLMError("down")

    with pytest.raises(SummaryError, match="no documents"):
        build_multi_representation(_docs(), FakeEmbedder(), AllFail(), Trace(question="b"))


# --- expansion ----------------------------------------------------------------

def _hit(doc_id, text):
    from rag.chunking import make_summary_chunk

    return RetrievedChunk(
        chunk=make_summary_chunk(f"{doc_id}:summary", doc_id, 0, text, level=1),
        score=0.5,
        rank=1,
    )


def test_expansion_replaces_the_summary_with_the_document():
    docstore = {"alpha": "the whole document"}
    out = expand_to_documents([_hit("alpha", "the summary")], docstore)
    assert out[0].chunk.text == "the whole document"


def test_expansion_preserves_score_and_rank():
    out = expand_to_documents([_hit("alpha", "s")], {"alpha": "full"})
    assert out[0].score == 0.5
    assert out[0].rank == 1


def test_expansion_leaves_a_hit_alone_when_the_document_is_missing():
    out = expand_to_documents([_hit("ghost", "s")], {})
    assert out[0].chunk.text == "s"
