"""Configuration for the RAG pipeline.

One frozen dataclass holds every tunable. The CLI overrides fields by keyword;
nothing reads os.environ except from_env, which makes the whole package testable
without touching the real environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal .env file. Missing file yields an empty mapping.

    Deliberately not python-dotenv: this is ten lines and one less dependency.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


@dataclass(frozen=True)
class Config:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gemini-3.5-flash-lite"
    chunk_tokens: int = 200
    chunk_overlap: int = 50
    max_seq_tokens: int = 256
    top_k: int = 5
    retrieval_depth: int = 20
    """How many chunks each individual query retrieves, before combination.

    Distinct from `top_k`, which is how many survive into the answer. Keeping
    them separate matters for the multi-query strategies: if each of five
    rewrites retrieved only `top_k` chunks, fusion would have almost nothing
    to fuse and RAG-Fusion could not differ meaningfully from a plain union.
    """
    raptor_max_depth: int = 3
    """How many summary levels RAPTOR builds above the raw chunks.

    Each level costs one LLM call per cluster, so depth is the main lever on
    build cost. Three levels over 5,116 chunks is roughly 640 + 80 + 10 calls.
    """

    raptor_cluster_size: int = 8
    """Target chunks per cluster, which sets how fast the tree narrows."""
    corpus_dir: Path = Path("data/corpus")
    metadata_path: Path = Path("data/metadata.json")
    index_path: Path = Path("data/index.npz")
    cache_dir: Path = Path(".cache/llm")
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_tokens ({self.chunk_tokens}), or windows never advance"
            )
        if self.chunk_tokens > self.max_seq_tokens:
            raise ValueError(
                f"chunk_tokens ({self.chunk_tokens}) exceeds max_seq_tokens "
                f"({self.max_seq_tokens}); chunks would be silently truncated"
            )
        if self.retrieval_depth < self.top_k:
            raise ValueError(
                f"retrieval_depth ({self.retrieval_depth}) must be at least "
                f"top_k ({self.top_k}); each query retrieves at depth and the "
                "combined result is then truncated to top_k"
            )
        if self.raptor_max_depth < 1:
            raise ValueError(
                f"raptor_max_depth must be at least 1, got {self.raptor_max_depth}"
            )
        if self.raptor_cluster_size < 2:
            raise ValueError(
                f"raptor_cluster_size must be at least 2, got "
                f"{self.raptor_cluster_size}"
            )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        env_file: Path | None = None,
        **overrides: object,
    ) -> "Config":
        """Build a Config from the environment, with .env as a fallback layer."""
        env = os.environ if env is None else env
        file_values = load_env_file(env_file) if env_file is not None else {}
        api_key = env.get("GOOGLE_API_KEY") or file_values.get("GOOGLE_API_KEY")
        base = cls(api_key=api_key)
        return replace(base, **overrides) if overrides else base
