import json

from rag.chunking import Chunk
from rag.trace import RetrievedChunk, StageTiming, Trace


def _chunk() -> Chunk:
    return Chunk("d:0", "d", 0, "text", 0, 10, 0, 40)


def test_new_trace_starts_empty_apart_from_the_question():
    trace = Trace(question="what is RRF?")
    assert trace.question == "what is RRF?"
    assert trace.queries == []
    assert trace.retrieved == []
    assert trace.prompt is None
    assert trace.answer is None
    assert trace.timings == []
    assert trace.notes == []


def test_stage_records_a_timing():
    trace = Trace(question="q")
    with trace.stage("embed"):
        pass
    assert len(trace.timings) == 1
    assert trace.timings[0].name == "embed"
    assert trace.timings[0].ms >= 0.0


def test_stages_accumulate_in_order():
    trace = Trace(question="q")
    with trace.stage("embed"):
        pass
    with trace.stage("search"):
        pass
    assert [t.name for t in trace.timings] == ["embed", "search"]


def test_stage_records_a_timing_even_when_the_body_raises():
    trace = Trace(question="q")
    try:
        with trace.stage("generate"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert [t.name for t in trace.timings] == ["generate"]


def test_total_ms_sums_the_stages():
    trace = Trace(question="q")
    trace.timings = [StageTiming("a", 1.5), StageTiming("b", 2.5)]
    assert trace.total_ms == 4.0


def test_total_ms_of_an_empty_trace_is_zero():
    assert Trace(question="q").total_ms == 0.0


def test_note_appends_a_message():
    trace = Trace(question="q")
    trace.note("llm disabled")
    trace.note("degraded to direct retrieval")
    assert trace.notes == ["llm disabled", "degraded to direct retrieval"]


def test_to_dict_is_json_serialisable():
    trace = Trace(question="q")
    trace.queries = ["q"]
    trace.retrieved = [RetrievedChunk(chunk=_chunk(), score=0.75, rank=1)]
    trace.prompt = "prompt text"
    trace.answer = "answer text"
    trace.timings = [StageTiming("embed", 1.25)]
    trace.note("a note")

    encoded = json.dumps(trace.to_dict())      # must not raise
    decoded = json.loads(encoded)

    assert decoded["question"] == "q"
    assert decoded["answer"] == "answer text"
    assert decoded["notes"] == ["a note"]
    assert decoded["retrieved"][0]["score"] == 0.75
    assert decoded["retrieved"][0]["rank"] == 1
    assert decoded["retrieved"][0]["chunk"]["chunk_id"] == "d:0"
    assert decoded["timings"][0] == {"name": "embed", "ms": 1.25}
    assert decoded["total_ms"] == 1.25
