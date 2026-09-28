"""Offline tests for evaluation/routing_eval.py.

A small, fully hand-built world: three topics, four gold questions, a
scripted LLM that returns a fixed reply per question. Every expected number
in `score_routing`'s output is computed by hand in the test itself, so this
pins the actual arithmetic (what counts as a hit, what an abstention costs
the mean) rather than just checking the code runs.
"""

from __future__ import annotations

import pytest

from evaluation.gold import GoldQuestion
from evaluation.routing_eval import RoutingScore, format_routing_table, score_routing
from rag.chunking import Chunk
from rag.store import VectorStore
from rag.trace import Trace


class ScriptedLLM:
    """Returns a canned `structured()` reply keyed by a substring of the
    prompt (the question text `ROUTE_TEMPLATE` embeds verbatim)."""

    def __init__(self, replies: dict[str, str]) -> None:
        self._replies = replies
        self._next_reply = ""

    def generate(self, prompt: str) -> str:
        return self._next_reply

    def structured(self, prompt: str, schema: dict) -> dict:
        from rag.llm import GeminiLLM

        # Longest match wins: one question's text can be a substring of
        # another's ("question about A" inside "second question about A"),
        # and the more specific needle is the correct one.
        matches = [needle for needle in self._replies if needle in prompt]
        if not matches:
            raise AssertionError(f"no scripted reply matches prompt: {prompt!r}")
        best = max(matches, key=len)
        self._next_reply = self._replies[best]
        return GeminiLLM.structured(self, prompt, schema)


def _world():
    """Three single-chunk documents, one per topic, and four gold questions.

    - q1 names docA (topic t1); the router correctly narrows to {t1} -> hit.
    - q2 names docB (topic t2); the router narrows to {t1}, missing t2 -> miss.
    - q3 names docC (topic t3); the router abstains ({}) -> hit by
      construction, and the abstention counts as choosing all 3 topics.
    - q4 names docA (topic t1) again; the router "chooses" all three topics,
      which is not narrowing at all but still contains t1 -> hit.
    """
    chunks = [
        Chunk("docA:0", "docA", 0, "a", 0, 1, 0, 10),
        Chunk("docB:0", "docB", 0, "b", 0, 1, 0, 10),
        Chunk("docC:0", "docC", 0, "c", 0, 1, 0, 10),
    ]
    store = VectorStore(vectors=[[1.0], [1.0], [1.0]], chunks=chunks)
    store.doc_meta = {
        "docA": {"topic": "t1"},
        "docB": {"topic": "t2"},
        "docC": {"topic": "t3"},
    }

    gold = [
        GoldQuestion(
            id="q1", question="question about A", doc_id="docA",
            quotes=("x",), why="w", spans=((0, 1),),
        ),
        GoldQuestion(
            id="q2", question="question about B", doc_id="docB",
            quotes=("x",), why="w", spans=((0, 1),),
        ),
        GoldQuestion(
            id="q3", question="question about C", doc_id="docC",
            quotes=("x",), why="w", spans=((0, 1),),
        ),
        GoldQuestion(
            id="q4", question="second question about A", doc_id="docA",
            quotes=("x",), why="w", spans=((0, 1),),
        ),
    ]

    # Needles must not be substrings of one another -- ROUTE_TEMPLATE embeds
    # the question text verbatim, and ScriptedLLM matches by substring, so an
    # overlapping needle would silently pick the wrong scripted reply.
    llm = ScriptedLLM(
        {
            "question about A": '{"topics": ["t1"]}',
            "question about B": '{"topics": ["t1"]}',
            "question about C": '{"topics": []}',
            "second question about A": '{"topics": ["t1", "t2", "t3"]}',
        }
    )
    return gold, store, llm


def test_score_routing_matches_hand_computed_values():
    gold, store, llm = _world()
    score = score_routing(gold, store, llm)

    # hits: q1 (t1 in {t1}), q3 (abstention, always a hit), q4 (t1 in all
    # three) -- q2 misses (t2 not in {t1}). 3/4.
    assert score.recall == pytest.approx(0.75)
    # only q3 abstains: 1/4.
    assert score.abstention_rate == pytest.approx(0.25)
    # topic counts: q1=1, q2=1, q3=3 (abstention costs all 3), q4=3 -> mean 2.0
    assert score.mean_topics_chosen == pytest.approx(2.0)
    assert score.topics_available == 3
    assert score.questions == 4


def test_score_routing_uses_a_fresh_trace_per_question():
    gold, store, llm = _world()
    traces = []

    def factory(question: str) -> Trace:
        trace = Trace(question=question)
        traces.append(trace)
        return trace

    score_routing(gold, store, llm, trace_factory=factory)
    assert len(traces) == len(gold)
    # q3's abstention must be visible only on its own trace.
    q3_trace = traces[2]
    assert any("degraded" in n for n in q3_trace.notes)
    q1_trace = traces[0]
    assert not any("degraded" in n for n in q1_trace.notes)


def test_score_routing_raises_when_a_gold_doc_has_no_topic():
    gold, store, llm = _world()
    store.doc_meta["docA"] = {}  # no topic recorded
    with pytest.raises(ValueError, match="no topic"):
        score_routing(gold, store, llm)


def test_score_routing_is_deterministic_across_repeat_runs():
    gold, store, llm = _world()
    first = score_routing(gold, store, llm)
    second = score_routing(gold, store, llm)
    assert first == second


def test_score_routing_handles_an_empty_gold_set():
    _, store, llm = _world()
    score = score_routing([], store, llm)
    assert score.recall == 0.0
    assert score.abstention_rate == 0.0
    assert score.mean_topics_chosen == 0.0
    assert score.questions == 0


# --- format_routing_table ----------------------------------------------------

def _score():
    return RoutingScore(
        recall=0.75,
        abstention_rate=0.25,
        mean_topics_chosen=2.0,
        topics_available=3,
        questions=4,
    )


def test_format_routing_table_is_markdown():
    lines = format_routing_table(_score()).splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}


def test_format_routing_table_reports_every_field():
    table = format_routing_table(_score())
    assert "0.750" in table
    assert "0.250" in table
    assert "2.00" in table
    assert "3" in table
    assert "4" in table
