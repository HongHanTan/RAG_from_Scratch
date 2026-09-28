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


# --- cache visibility ---------------------------------------------------------

def test_last_call_cached_is_false_on_a_fresh_call(tmp_path):
    llm = _llm(tmp_path, ["hello"])
    llm.generate("prompt")
    assert llm.last_call_cached is False


def test_last_call_cached_is_true_on_a_cache_hit(tmp_path):
    llm = _llm(tmp_path, ["hello"])
    llm.generate("prompt")
    llm.generate("prompt")
    assert llm.last_call_cached is True


def test_last_call_cached_updates_across_calls(tmp_path):
    llm = _llm(tmp_path, ["one", "two"])
    llm.generate("a")
    assert llm.last_call_cached is False
    llm.generate("a")
    assert llm.last_call_cached is True
    llm.generate("b")
    assert llm.last_call_cached is False


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


# --- structured output --------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["topics"],
}


def test_structured_parses_a_json_object(tmp_path):
    llm = _llm(tmp_path, ['{"topics": ["a", "b"], "reason": "because"}'])
    assert llm.structured("p", SCHEMA) == {"topics": ["a", "b"], "reason": "because"}


def test_structured_tolerates_a_fenced_code_block(tmp_path):
    # Models wrap JSON in ```json fences regardless of instructions.
    reply = '```json\n{"topics": ["a"]}\n```'
    llm = _llm(tmp_path, [reply])
    assert llm.structured("p", SCHEMA) == {"topics": ["a"]}


def test_structured_tolerates_surrounding_prose(tmp_path):
    reply = 'Here is the result:\n{"topics": ["a"]}\nHope that helps!'
    llm = _llm(tmp_path, [reply])
    assert llm.structured("p", SCHEMA) == {"topics": ["a"]}


def test_structured_rejects_unparseable_json(tmp_path):
    llm = _llm(tmp_path, ["not json at all"], max_retries=1)
    with pytest.raises(LLMError, match="JSON"):
        llm.structured("p", SCHEMA)


def test_structured_rejects_a_missing_required_field(tmp_path):
    llm = _llm(tmp_path, ['{"reason": "no topics here"}'], max_retries=1)
    with pytest.raises(LLMError, match="topics"):
        llm.structured("p", SCHEMA)


def test_structured_rejects_a_non_object(tmp_path):
    llm = _llm(tmp_path, ['["a", "b"]'], max_retries=1)
    with pytest.raises(LLMError, match="object"):
        llm.structured("p", SCHEMA)


def test_structured_rejects_a_wrongly_typed_field(tmp_path):
    llm = _llm(tmp_path, ['{"topics": "a"}'], max_retries=1)
    with pytest.raises(LLMError, match="topics"):
        llm.structured("p", SCHEMA)


def test_structured_rejects_a_boolean_for_an_integer_field(tmp_path):
    # isinstance(True, int) is True in Python, so an "integer" schema field
    # would otherwise silently accept a boolean. Phase 5's RAPTOR schemas
    # declare "integer" fields, so this must be caught now.
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    llm = _llm(tmp_path, ['{"count": true}'], max_retries=1)
    with pytest.raises(LLMError, match="count"):
        llm.structured("p", schema)


def test_structured_uses_the_same_cache_as_generate(tmp_path):
    llm = _llm(tmp_path, ['{"topics": ["a"]}'])
    llm.structured("p", SCHEMA)
    llm.structured("p", SCHEMA)
    assert llm.call_count == 1


# --- a failed structured() reply must not poison the cache forever ----------
#
# generate() writes the reply to disk before structured() gets a chance to
# parse or shape-check it, so a malformed reply used to be cached under a
# deterministic key: every later call -- even from a fresh process with a
# working API -- would re-read the same bad text and re-fail identically,
# never making another API call.


def test_structured_does_not_permanently_cache_unparseable_json(tmp_path):
    cache_dir = tmp_path / "cache"
    first = GeminiLLM(
        model="fake-model",
        api_key="key",
        cache_dir=cache_dir,
        client=FakeClient(["not json at all"]),
        sleep=lambda seconds: None,
        max_retries=1,
    )
    with pytest.raises(LLMError, match="JSON"):
        first.structured("p", SCHEMA)

    second = GeminiLLM(
        model="fake-model",
        api_key="key",
        cache_dir=cache_dir,
        client=FakeClient(['{"topics": ["a"]}']),
        sleep=lambda seconds: None,
        max_retries=1,
    )
    assert second.structured("p", SCHEMA) == {"topics": ["a"]}
    assert second.call_count == 1  # the retry actually reached the model


def test_structured_does_not_permanently_cache_a_shape_failure(tmp_path):
    cache_dir = tmp_path / "cache"
    first = GeminiLLM(
        model="fake-model",
        api_key="key",
        cache_dir=cache_dir,
        client=FakeClient(['{"reason": "no topics here"}']),
        sleep=lambda seconds: None,
        max_retries=1,
    )
    with pytest.raises(LLMError, match="topics"):
        first.structured("p", SCHEMA)

    second = GeminiLLM(
        model="fake-model",
        api_key="key",
        cache_dir=cache_dir,
        client=FakeClient(['{"topics": ["a"]}']),
        sleep=lambda seconds: None,
        max_retries=1,
    )
    assert second.structured("p", SCHEMA) == {"topics": ["a"]}
    assert second.call_count == 1


def test_a_request_timeout_is_configured_on_the_real_client():
    # The failure that hurts during an overload is a hang, not an error:
    # the API can hold a call open for ten minutes, and five retries of that
    # is an hour for one summary. Without a timeout a long build stalls
    # instead of converging.
    from pathlib import Path

    from google.genai import types

    from rag.llm import GeminiLLM

    llm = GeminiLLM(
        model="m", api_key="k", cache_dir=Path("."), request_timeout=12.0
    )
    options = llm._client._api_client._http_options
    assert isinstance(options, types.HttpOptions)
    assert options.timeout == 12000  # milliseconds


def test_the_timeout_defaults_to_thirty_seconds():
    from pathlib import Path

    from rag.llm import GeminiLLM

    llm = GeminiLLM(model="m", api_key="k", cache_dir=Path("."))
    assert llm._client._api_client._http_options.timeout == 30000
