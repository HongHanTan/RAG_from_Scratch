"""Shared fixtures.

Every fake here is deterministic and loads no model, so the default test suite
runs in seconds with no network and no API key.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from rag.config import Config

ALPHA_TEXT = (
    "Cosine similarity measures the angle between two vectors and ignores their "
    "magnitude. Two documents about the same topic point in a similar direction "
    "even when one is much longer than the other."
)
BETA_TEXT = (
    "Reciprocal rank fusion combines several ranked lists into a single ranking. "
    "Each document receives one divided by a constant plus its rank in each list, "
    "and the sums are sorted to produce the final order."
)


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Config:
    """A two-document corpus on disk, with a Config pointing at it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.txt").write_text(ALPHA_TEXT, encoding="utf-8")
    (corpus / "beta.txt").write_text(BETA_TEXT, encoding="utf-8")

    metadata = {
        "documents": {
            "alpha": {
                "title": "Cosine Similarity",
                "source": "notes",
                "publish_date": "2023-05-01",
                "author": "A. Author",
                "url": "https://example.invalid/alpha",
            },
            "beta": {
                "title": "Reciprocal Rank Fusion",
                "source": "notes",
                "publish_date": "2024-02-11",
                "author": "B. Author",
                "url": "https://example.invalid/beta",
            },
        }
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    return Config(
        corpus_dir=corpus,
        metadata_path=metadata_path,
        index_path=tmp_path / "index.npz",
        cache_dir=tmp_path / "cache",
        chunk_tokens=8,
        chunk_overlap=2,
        max_seq_tokens=8,
        top_k=3,
    )


class FakeTokenizer:
    """Whitespace tokenizer with real character offsets.

    Mimics the parts of a HuggingFace fast tokenizer that chunking.py uses, so
    chunking can be tested without downloading a model.
    """

    is_fast = True

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        **kwargs: object,
    ) -> dict[str, list]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for match in re.finditer(r"\S+", text):
            ids.append(zlib.crc32(match.group().encode("utf-8")) % 30000)
            offsets.append((match.start(), match.end()))
        result: dict[str, list] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class FakeEmbedder:
    """Deterministic pseudo-embeddings derived from a CRC of the text.

    CRC rather than hash(): Python randomises str hashing per process, which
    would make these vectors differ between test runs.
    """

    dim = 8

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = zlib.crc32(text.encode("utf-8"))
            out[i] = np.random.default_rng(seed).normal(size=self.dim)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


class FakeLLM:
    """Records the prompts it is given and returns a canned answer."""

    def __init__(self, response: str = "A stub answer. [1]") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response

    def structured(self, prompt: str, schema: dict) -> dict:
        # Shares GeminiLLM's own parsing/shape-checking so routing and query
        # construction tests can hand this a canned JSON reply directly,
        # rather than reimplementing that logic in a second fake.
        from rag.llm import GeminiLLM

        return GeminiLLM.structured(self, prompt, schema)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
