"""Gemini client: caching, retry, and nothing else.

The cache is not an optimisation, it is what makes the project usable. One
benchmark run is hundreds of calls, and re-running it after a metric tweak
should cost nothing. Cache entries are keyed by model, prompt and temperature,
so changing any of them correctly misses.

The client and sleep function are injected so retry behaviour is testable
without the network and without actually waiting.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable


class LLMError(Exception):
    """Raised when the model cannot be reached or returns nothing usable."""


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """Pull the JSON object out of a reply that may be wrapped in prose.

    Models add ```json fences and conversational padding whatever the prompt
    says. Parsing the first balanced-looking object is more robust than
    trusting the instruction, and this is the whole reason structured output
    is worth having over prose parsing.
    """
    match = _JSON_OBJECT.search(text)
    return match.group(0) if match else text


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def _check_shape(value: dict, schema: dict) -> None:
    """Validate a parsed object against a small subset of JSON Schema.

    Only what this project's schemas use: a top-level object, `required`
    keys, and one-level `type` checks on properties. Written out rather than
    pulling in jsonschema, which is not on the dependency list.
    """
    if not isinstance(value, dict):
        raise LLMError(f"expected a JSON object, got {type(value).__name__}")
    for key in schema.get("required", []):
        if key not in value:
            raise LLMError(f"missing required field in model reply: {key}")
    for key, spec in schema.get("properties", {}).items():
        if key not in value:
            continue
        expected = _TYPES.get(spec.get("type"))
        # bool is a subclass of int, so isinstance(True, int) is True; without
        # this, a boolean would silently pass as a valid "integer" field.
        is_bool_masquerading_as_int = spec.get("type") == "integer" and isinstance(
            value[key], bool
        )
        if is_bool_masquerading_as_int or (expected and not isinstance(value[key], expected)):
            raise LLMError(
                f"field {key} should be {spec['type']}, "
                f"got {type(value[key]).__name__}"
            )


def cache_key(model: str, prompt: str, temperature: float) -> str:
    """Stable content hash identifying one request."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evict_cache_for(llm: object, prompt: str) -> None:
    """Remove `llm`'s cache entry for `prompt`, if it has one.

    `generate` writes its reply to the cache before `structured` gets a
    chance to check it, so a malformed reply would otherwise be cached under
    a deterministic key forever -- every later call, including one from a
    fresh process with a working API, would keep re-reading the same bad
    text and re-failing identically. Deleting it here means a retry can
    actually reach the model.

    Best-effort and tolerant of objects that are not a real `GeminiLLM`: test
    doubles share `structured`'s parsing logic via
    `GeminiLLM.structured(self, ...)` but duck-type only `generate`, with no
    `model`/`temperature`/cache directory of their own. Any of that being
    missing is swallowed exactly like a missing cache file.
    """
    try:
        key = cache_key(llm.model, prompt, llm.temperature)
        llm._cache_path(key).unlink()
    except (AttributeError, FileNotFoundError):
        pass


class GeminiLLM:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        cache_dir: Path,
        temperature: float = 0.0,
        max_retries: int = 5,
        client: object | None = None,
        sleep: Callable[[float], None] = time.sleep,
        request_timeout: float = 30.0,
    ) -> None:
        if client is None and not api_key:
            raise LLMError(
                "no API key: set GOOGLE_API_KEY in the environment or in .env"
            )
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sleep = sleep
        self.call_count = 0
        self.last_call_cached = False

        if client is not None:
            self._client = client
        else:
            from google import genai
            from google.genai import types

            # A request timeout, because the failure that actually hurts is
            # not a fast error but a hang. During a 503 overload the API can
            # leave a call open for ten minutes; five retries of that is an
            # hour for one summary, and a 730-call build never finishes.
            # Failing fast lets the retry land in a window where the model is
            # up, which -- when it is intermittently available -- is the
            # difference between converging and stalling.
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(request_timeout * 1000)  # milliseconds
                ),
            )

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except (json.JSONDecodeError, KeyError):
            return None   # a corrupt entry is a miss, not a crash

    def _write_cache(self, key: str, prompt: str, response: str) -> None:
        self._cache_path(key).write_text(
            json.dumps(
                {"model": self.model, "prompt": prompt, "response": response},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def generate(self, prompt: str) -> str:
        """Return the model's text for prompt, from cache when available."""
        key = cache_key(self.model, prompt, self.temperature)
        cached = self._read_cache(key)
        if cached is not None:
            self.last_call_cached = True
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self.call_count += 1
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": self.temperature},
                )
                text = (response.text or "").strip()
                if not text:
                    raise LLMError("model returned an empty response")
                self._write_cache(key, prompt, text)
                self.last_call_cached = False
                return text
            except LLMError:
                raise
            except Exception as exc:       # transport, rate limit, server error
                last_error = exc
                if attempt < self.max_retries - 1:
                    self._sleep(2.0 ** attempt)

        raise LLMError(
            f"gave up after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def structured(self, prompt: str, schema: dict) -> dict:
        """Return a parsed, shape-checked JSON object from the model.

        Shares `generate`'s cache, retry and backoff. Raises LLMError when the
        reply cannot be parsed or does not match the schema — callers are
        expected to catch that and fall back to a safe default, because a
        routing or filtering step that fails should not lose the answer.

        A reply that fails either check is evicted from the cache rather than
        left there: `generate` already wrote it before this method could
        judge it, and leaving a bad reply cached would make every future call
        with this prompt fail the same way, permanently, even once the model
        would answer correctly.
        """
        raw = self.generate(prompt)
        try:
            parsed = json.loads(_extract_json_object(raw))
        except json.JSONDecodeError as exc:
            _evict_cache_for(self, prompt)
            raise LLMError(
                f"model reply was not valid JSON: {exc}; got {raw[:120]!r}"
            ) from exc
        try:
            _check_shape(parsed, schema)
        except LLMError:
            _evict_cache_for(self, prompt)
            raise
        return parsed
