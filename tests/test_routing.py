import pytest

from rag.llm import LLMError
from rag.routing import SemanticRouter, logical_route
from rag.trace import Trace
from tests.conftest import FakeEmbedder

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


# --- logical routing ---------------------------------------------------------

def test_logical_route_returns_the_chosen_topics():
    llm = ReplyLLM('{"topics": ["retrieval-models"]}')
    assert logical_route("how does ColBERT score?", llm, TOPICS, Trace(question="q")) == (
        "retrieval-models",
    )


def test_logical_route_can_choose_several():
    llm = ReplyLLM('{"topics": ["retrieval-models", "rag-systems"]}')
    chosen = logical_route("q", llm, TOPICS, Trace(question="q"))
    assert set(chosen) == {"retrieval-models", "rag-systems"}


def test_logical_route_shows_the_llm_the_available_topics():
    llm = ReplyLLM('{"topics": ["foundations"]}')
    logical_route("q", llm, TOPICS, Trace(question="q"))
    for topic in TOPICS:
        assert topic in llm.prompts[0]


def test_logical_route_records_its_choice_on_the_trace():
    trace = Trace(question="q")
    logical_route("q", ReplyLLM('{"topics": ["foundations"]}'), TOPICS, trace)
    assert any(s.kind == "route" for s in trace.translation)


def test_logical_route_drops_invented_topics():
    llm = ReplyLLM('{"topics": ["foundations", "not-a-topic"]}')
    assert logical_route("q", llm, TOPICS, Trace(question="q")) == ("foundations",)


def test_logical_route_returns_everything_when_the_llm_fails():
    class Failing:
        def generate(self, prompt):
            raise LLMError("rate limited")

        def structured(self, prompt, schema):
            raise LLMError("rate limited")

    trace = Trace(question="q")
    assert logical_route("q", Failing(), TOPICS, trace) == ()
    assert any("degraded" in n for n in trace.notes)


def test_logical_route_returns_everything_when_nothing_valid_was_chosen():
    # An empty tuple means "search everything", which is the safe default:
    # routing narrows the search, so failing to route should widen it back.
    trace = Trace(question="q")
    assert logical_route("q", ReplyLLM('{"topics": []}'), TOPICS, trace) == ()


def test_logical_route_without_an_llm_returns_everything():
    trace = Trace(question="q")
    assert logical_route("q", None, TOPICS, trace) == ()
    assert any("degraded" in n for n in trace.notes)


# --- semantic routing --------------------------------------------------------

DESCRIPTIONS = {
    "definition": "explaining what a term or concept means",
    "mechanism": "describing how a method works step by step",
    "comparison": "contrasting two approaches against each other",
}


def test_semantic_router_returns_a_known_prompt_name():
    router = SemanticRouter(FakeEmbedder(), DESCRIPTIONS)
    assert router.route("what is late interaction?", Trace(question="q")) in DESCRIPTIONS


def test_semantic_router_is_deterministic():
    router = SemanticRouter(FakeEmbedder(), DESCRIPTIONS)
    trace = Trace(question="q")
    first = router.route("how does ColBERT work?", trace)
    second = router.route("how does ColBERT work?", trace)
    assert first == second


def test_semantic_router_records_its_choice():
    router = SemanticRouter(FakeEmbedder(), DESCRIPTIONS)
    trace = Trace(question="q")
    chosen = router.route("q", trace)
    assert any(s.kind == "route" and chosen in s.text for s in trace.translation)


def test_semantic_router_uses_no_llm():
    # Semantic routing is cosine similarity over embeddings, nothing else.
    # If it needed an LLM it would be logical routing with extra steps.
    router = SemanticRouter(FakeEmbedder(), DESCRIPTIONS)
    assert router.route("q", Trace(question="q"))


def test_semantic_router_embeds_descriptions_once():
    embedder = FakeEmbedder()
    calls = []
    original = embedder.encode

    def counting(texts, batch_size=32):
        calls.append(len(texts))
        return original(texts, batch_size=batch_size)

    embedder.encode = counting
    router = SemanticRouter(embedder, DESCRIPTIONS)
    trace = Trace(question="q")
    router.route("a", trace)
    router.route("b", trace)
    # one batch of 3 descriptions at construction, then one query each
    assert calls == [3, 1, 1]


def test_semantic_router_rejects_an_empty_description_set():
    with pytest.raises(ValueError, match="at least one"):
        SemanticRouter(FakeEmbedder(), {})
