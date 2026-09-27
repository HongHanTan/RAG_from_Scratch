from rag.chunking import Chunk
from rag.generation import generate_answer
from rag.trace import Trace
from tests.conftest import FakeLLM


def _retrieved():
    return [(Chunk("a:0", "a", 0, "Some context.", 0, 5, 0, 13), 0.9)]


def test_returns_the_model_answer():
    trace = Trace(question="q")
    assert generate_answer(FakeLLM("the answer"), "q", _retrieved(), trace) == "the answer"


def test_records_the_prompt_on_the_trace():
    trace = Trace(question="q")
    generate_answer(FakeLLM(), "q", _retrieved(), trace)
    assert trace.prompt is not None
    assert "Some context." in trace.prompt


def test_records_the_answer_on_the_trace():
    trace = Trace(question="q")
    generate_answer(FakeLLM("the answer"), "q", _retrieved(), trace)
    assert trace.answer == "the answer"


def test_records_a_generate_timing():
    trace = Trace(question="q")
    generate_answer(FakeLLM(), "q", _retrieved(), trace)
    assert [t.name for t in trace.timings] == ["generate"]


def test_sends_exactly_one_prompt():
    llm = FakeLLM()
    generate_answer(llm, "q", _retrieved(), Trace(question="q"))
    assert len(llm.prompts) == 1


def test_llm_failure_is_recorded_as_a_note_and_yields_no_answer():
    class FailingLLM:
        def generate(self, prompt):
            from rag.llm import LLMError
            raise LLMError("rate limited")

    trace = Trace(question="q")
    answer = generate_answer(FailingLLM(), "q", _retrieved(), trace)
    assert answer is None
    assert trace.answer is None
    assert any("rate limited" in n for n in trace.notes)


# --- cache visibility ---------------------------------------------------------

class CacheReportingLLM:
    """An LLM stand-in that reports whether its last call was cached."""

    def __init__(self, response: str = "the answer", cached: bool = False) -> None:
        self.response = response
        self.last_call_cached = cached
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_cache_hit_is_noted_on_the_trace():
    trace = Trace(question="q")
    generate_answer(CacheReportingLLM(cached=True), "q", _retrieved(), trace)
    assert any("cache" in n for n in trace.notes)


def test_cache_miss_is_not_noted_on_the_trace():
    trace = Trace(question="q")
    generate_answer(CacheReportingLLM(cached=False), "q", _retrieved(), trace)
    assert not any("cache" in n for n in trace.notes)


def test_llm_without_cache_attribute_still_works():
    # FakeLLM has no last_call_cached attribute; generate_answer must not
    # blow up looking for it, and must not fabricate a cache note.
    trace = Trace(question="q")
    answer = generate_answer(FakeLLM("the answer"), "q", _retrieved(), trace)
    assert answer == "the answer"
    assert not any("cache" in n for n in trace.notes)
