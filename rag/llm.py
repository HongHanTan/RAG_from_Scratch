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
import time
from pathlib import Path
from typing import Callable


class LLMError(Exception):
    """Raised when the model cannot be reached or returns nothing usable."""


def cache_key(model: str, prompt: str, temperature: float) -> str:
    """Stable content hash identifying one request."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
            self._client = genai.Client(api_key=api_key)

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
