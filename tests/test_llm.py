import pytest

from rag.llm import GeminiLLM, LLMError, cache_key


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, owner: "FakeClient") -> None:
        self._owner = owner

    def generate_content(self, model, contents, config=None):
        self._owner.calls.append({"model": model, "contents": contents})
        outcome = self._owner.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


class FakeClient:
    """Stands in for google.genai.Client."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.models = FakeModels(self)


def _llm(tmp_path, outcomes, **kwargs) -> GeminiLLM:
    return GeminiLLM(
        model="fake-model",
        api_key="key",
        cache_dir=tmp_path / "cache",
        client=FakeClient(outcomes),
        sleep=lambda seconds: None,
        **kwargs,
    )


# --- cache key --------------------------------------------------------------

def test_cache_key_is_stable_for_identical_input():
    assert cache_key("m", "p", 0.0) == cache_key("m", "p", 0.0)


def test_cache_key_changes_with_the_prompt():
    assert cache_key("m", "a", 0.0) != cache_key("m", "b", 0.0)


def test_cache_key_changes_with_the_model():
    assert cache_key("m1", "p", 0.0) != cache_key("m2", "p", 0.0)


def test_cache_key_changes_with_the_temperature():
    assert cache_key("m", "p", 0.0) != cache_key("m", "p", 0.7)


def test_cache_key_is_a_hex_digest():
    key = cache_key("m", "p", 0.0)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


# --- generation -------------------------------------------------------------

def test_generate_returns_the_model_text(tmp_path):
    llm = _llm(tmp_path, ["hello"])
    assert llm.generate("prompt") == "hello"


def test_generate_passes_the_prompt_through(tmp_path):
    llm = _llm(tmp_path, ["hello"])
    llm.generate("my prompt")
    assert llm._client.calls[0]["contents"] == "my prompt"


def test_repeated_prompts_are_served_from_cache(tmp_path):
    llm = _llm(tmp_path, ["hello"])
    assert llm.generate("prompt") == "hello"
    assert llm.generate("prompt") == "hello"
    assert llm.call_count == 1          # only one API call was made


def test_cache_survives_a_new_client_instance(tmp_path):
    first = _llm(tmp_path, ["hello"])
    first.generate("prompt")

    second = _llm(tmp_path, [])         # no outcomes left: must not call out
    assert second.generate("prompt") == "hello"
    assert second.call_count == 0


def test_different_prompts_do_not_share_a_cache_entry(tmp_path):
    llm = _llm(tmp_path, ["one", "two"])
    assert llm.generate("a") == "one"
    assert llm.generate("b") == "two"


# --- retry ------------------------------------------------------------------

def test_transient_failure_is_retried(tmp_path):
    llm = _llm(tmp_path, [RuntimeError("429 rate limit"), "recovered"])
    assert llm.generate("prompt") == "recovered"
    assert llm.call_count == 2


def test_gives_up_after_max_retries_and_raises_llm_error(tmp_path):
    llm = _llm(
        tmp_path,
        [RuntimeError("429 rate limit")] * 3,
        max_retries=3,
    )
    with pytest.raises(LLMError, match="3 attempts"):
        llm.generate("prompt")


def test_backoff_delays_grow(tmp_path):
    delays: list[float] = []
    llm = GeminiLLM(
        model="m",
        api_key="key",
        cache_dir=tmp_path / "cache",
        client=FakeClient([RuntimeError("429"), RuntimeError("429"), "ok"]),
        sleep=delays.append,
    )
    llm.generate("prompt")
    assert delays == sorted(delays)
    assert len(delays) == 2
    assert delays[1] > delays[0]


def test_a_failed_call_is_not_cached(tmp_path):
    llm = _llm(tmp_path, [RuntimeError("429")] * 5, max_retries=5)
    with pytest.raises(LLMError):
        llm.generate("prompt")

    recovered = _llm(tmp_path, ["fresh"])
    assert recovered.generate("prompt") == "fresh"


def test_empty_response_is_an_error(tmp_path):
    llm = _llm(tmp_path, [""], max_retries=1)
    with pytest.raises(LLMError, match="empty"):
        llm.generate("prompt")


def test_missing_api_key_is_rejected_at_construction(tmp_path):
    with pytest.raises(LLMError, match="GOOGLE_API_KEY"):
        GeminiLLM(model="m", api_key=None, cache_dir=tmp_path)
