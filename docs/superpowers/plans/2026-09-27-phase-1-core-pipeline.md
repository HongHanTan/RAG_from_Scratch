# Phase 1: Core Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working retrieval-augmented question answering pipeline built without any RAG framework, driven by `python -m rag ask "..."`.

**Architecture:** Documents are loaded from committed `.txt` files with sidecar metadata, split by a token-aware sliding window, embedded by a local HuggingFace model with hand-written mean pooling and L2 normalisation, and stored as a dense NumPy matrix. Retrieval is an exact brute-force cosine matmul plus top-k. The retrieved chunks are formatted into a prompt template and sent to Gemini. Every stage writes into one `Trace` object, which is what the CLI prints and what later phases consume.

**Tech Stack:** Python 3.14, NumPy 2.4, PyTorch 2.13 (CPU), Transformers 5.14, google-genai 1.68, pytest. Standard library for everything else.

## Global Constraints

These apply to every task. Do not restate them per task; do not violate them.

- **No RAG framework.** LangChain, `langchain-*`, LlamaIndex, and `sentence-transformers` are installed in this environment but MUST NOT be imported anywhere in this project. Task 14 adds a test that enforces this.
- **Allowed third-party imports:** `numpy`, `torch`, `transformers`, `google.genai`, `pytest`. Nothing else. Specifically not `scikit-learn`, not `beautifulsoup4`, not `requests`, not `python-dotenv` — the standard library covers each of these needs here.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`. This is a model identifier on the HuggingFace Hub, loaded through `transformers`. Loading a model published by the sentence-transformers org is not the same as importing the `sentence_transformers` library, and is allowed.
- **Chunking:** 200 tokens, 50 token overlap, model max sequence length 256.
- **Determinism:** LLM temperature 0. No unseeded randomness anywhere.
- **Test-driven.** Write the test, run it, watch it fail for the expected reason, then implement. A test that passes before the implementation exists is a broken test.
- **Tests never hit the network and never load the embedding model** unless marked `@pytest.mark.slow` (real model) or `@pytest.mark.live` (real API). Both are deselected by default.
- **Platform is Windows.** Use `pathlib`, never string path concatenation. Write files with `encoding="utf-8"` explicitly — the Windows default is cp1252 and will corrupt the corpus.
- **Commit after every task.** Commit messages end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## File Structure

| File | Responsibility |
|---|---|
| `rag/config.py` | `Config` dataclass; env and `.env` loading |
| `rag/loader.py` | `Document`; read corpus dir + metadata.json |
| `rag/chunking.py` | `Chunk`; pure window arithmetic; tokenizer-driven chunking |
| `rag/embedding.py` | `mean_pool`, `l2_normalize`, `Embedder` |
| `rag/similarity.py` | `cosine_similarity`, `top_k` |
| `rag/store.py` | `VectorStore`: matrix + chunks, search, npz persistence |
| `rag/trace.py` | `StageTiming`, `RetrievedChunk`, `Trace` |
| `rag/llm.py` | `cache_key`, `LLMError`, `GeminiLLM` |
| `rag/prompts.py` | `ANSWER_TEMPLATE`, `format_context` |
| `rag/generation.py` | `generate_answer` |
| `rag/pipeline.py` | `build_index`, `load_index`, `ask` |
| `rag/__main__.py` | CLI: `index`, `ask` |
| `scripts/html_text.py` | `HTMLTextExtractor`, `html_to_text`, `normalize_whitespace` |
| `scripts/fetch_corpus.py` | source list, fetch driver, metadata writer |
| `tests/conftest.py` | `tiny_corpus`, `FakeTokenizer`, `FakeEmbedder`, `FakeLLM` |

Dependency direction is strictly one-way: `config` ← everything; `loader` → `chunking` → `embedding`/`store`; `pipeline` wires them; `__main__` only calls `pipeline`. Nothing imports `__main__`.

---

### Task 1: Scaffolding, Config, and test fixtures

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `rag/__init__.py`, `rag/config.py`, `tests/__init__.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Config` — frozen dataclass with fields `embedding_model: str`, `llm_model: str`, `chunk_tokens: int`, `chunk_overlap: int`, `max_seq_tokens: int`, `top_k: int`, `corpus_dir: Path`, `metadata_path: Path`, `index_path: Path`, `cache_dir: Path`, `api_key: str | None`
  - `Config.from_env(env: Mapping[str, str] | None = None, env_file: Path | None = None, **overrides) -> Config`
  - `load_env_file(path: Path) -> dict[str, str]`
  - Fixtures `tiny_corpus -> Config`, and classes `FakeTokenizer`, `FakeEmbedder`, `FakeLLM` importable from `tests.conftest`

- [ ] **Step 1: Create the project skeleton**

`pyproject.toml`:

```toml
[project]
name = "rag-from-scratch"
version = "0.1.0"
description = "Retrieval-augmented generation implemented without a RAG framework"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "torch>=2.0",
    "transformers>=4.40",
    "google-genai>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not slow and not live'"
markers = [
    "slow: loads the real embedding model (deselected by default)",
    "live: calls the real Gemini API (deselected by default)",
]
```

`.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
.env
data/index.npz
.cache/
```

`.env.example`:

```dotenv
# Copy to .env and fill in. .env is gitignored.
GOOGLE_API_KEY=your-key-here
```

Create empty `rag/__init__.py` and `tests/__init__.py`.

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:

```python
from pathlib import Path

from rag.config import Config, load_env_file


def test_defaults_match_the_spec():
    cfg = Config()
    assert cfg.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.chunk_tokens == 200
    assert cfg.chunk_overlap == 50
    assert cfg.max_seq_tokens == 256
    assert cfg.top_k == 5


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    import pytest
    with pytest.raises(ValueError, match="overlap"):
        Config(chunk_tokens=100, chunk_overlap=100)


def test_chunk_size_must_fit_the_model_window():
    import pytest
    with pytest.raises(ValueError, match="max_seq_tokens"):
        Config(chunk_tokens=300, max_seq_tokens=256)


def test_from_env_reads_the_api_key():
    cfg = Config.from_env(env={"GOOGLE_API_KEY": "abc123"})
    assert cfg.api_key == "abc123"


def test_from_env_leaves_api_key_none_when_unset():
    cfg = Config.from_env(env={})
    assert cfg.api_key is None


def test_overrides_beat_the_environment():
    cfg = Config.from_env(env={"GOOGLE_API_KEY": "abc123"}, top_k=9)
    assert cfg.top_k == 9
    assert cfg.api_key == "abc123"


def test_load_env_file_parses_pairs_and_ignores_comments(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "\n"
        "GOOGLE_API_KEY=secret\n"
        "QUOTED=\"with spaces\"\n"
        "  SPACED  =  padded  \n",
        encoding="utf-8",
    )
    assert load_env_file(p) == {
        "GOOGLE_API_KEY": "secret",
        "QUOTED": "with spaces",
        "SPACED": "padded",
    }


def test_load_env_file_returns_empty_when_missing(tmp_path: Path):
    assert load_env_file(tmp_path / "nope.env") == {}


def test_env_file_is_a_fallback_not_an_override(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text("GOOGLE_API_KEY=from-file\n", encoding="utf-8")
    cfg = Config.from_env(env={"GOOGLE_API_KEY": "from-env"}, env_file=p)
    assert cfg.api_key == "from-env"
```

- [ ] **Step 3: Run the test and verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'rag.config'`.

- [ ] **Step 4: Implement `rag/config.py`**

```python
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
    llm_model: str = "gemini-2.0-flash"
    chunk_tokens: int = 200
    chunk_overlap: int = 50
    max_seq_tokens: int = 256
    top_k: int = 5
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
```

Note on `replace` with a frozen dataclass: `__post_init__` runs again on the copy, so validation still applies to overrides.

- [ ] **Step 5: Run the test and verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 9 passed.

- [ ] **Step 6: Write the shared test fixtures**

`tests/conftest.py`:

```python
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


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
```

- [ ] **Step 7: Verify the fixtures import cleanly**

Run: `python -m pytest tests/ -v`
Expected: 9 passed, no collection errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore .env.example rag/ tests/
git commit -m "feat: project scaffolding, Config, and test fixtures

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: HTML text extraction

**Files:**
- Create: `scripts/__init__.py`, `scripts/html_text.py`
- Test: `tests/test_html_text.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `normalize_whitespace(text: str) -> str`
  - `html_to_text(html: str) -> str`

Why a hand-written parser: BeautifulSoup is installed but outside the allowed dependency list, and `html.parser.HTMLParser` does the job in forty lines. The interesting part is what to *drop* — script, style, and LaTeX math artifacts that ar5iv leaves in the markup.

- [ ] **Step 1: Write the failing test**

`tests/test_html_text.py`:

```python
from scripts.html_text import html_to_text, normalize_whitespace


def test_normalize_collapses_runs_of_spaces():
    assert normalize_whitespace("a   b\t\tc") == "a b c"


def test_normalize_keeps_paragraph_breaks_but_collapses_bigger_gaps():
    assert normalize_whitespace("one\n\n\n\n\ntwo") == "one\n\ntwo"


def test_normalize_strips_leading_and_trailing_space():
    assert normalize_whitespace("  hello  ") == "hello"


def test_extracts_visible_text():
    html = "<html><body><p>Hello world</p></body></html>"
    assert html_to_text(html) == "Hello world"


def test_drops_script_and_style_content():
    html = (
        "<html><head><style>p { color: red }</style></head>"
        "<body><p>Keep me</p><script>alert('no')</script></body></html>"
    )
    assert html_to_text(html) == "Keep me"


def test_block_elements_become_paragraph_breaks():
    html = "<p>First para</p><p>Second para</p>"
    assert html_to_text(html) == "First para\n\nSecond para"


def test_inline_elements_do_not_split_words():
    html = "<p>re<em>trie</em>val</p>"
    assert html_to_text(html) == "retrieval"


def test_headings_are_kept_as_their_own_paragraphs():
    html = "<h2>Method</h2><p>We propose</p>"
    assert html_to_text(html) == "Method\n\nWe propose"


def test_entities_are_decoded():
    assert html_to_text("<p>alpha &amp; beta &lt;tag&gt;</p>") == "alpha & beta <tag>"


def test_br_becomes_a_single_newline_not_a_paragraph_break():
    assert html_to_text("<p>line one<br>line two</p>") == "line one\nline two"


def test_empty_input_yields_empty_string():
    assert html_to_text("") == ""
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_html_text.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.html_text'`.

- [ ] **Step 3: Implement `scripts/html_text.py`**

Create empty `scripts/__init__.py` first, then:

```python
"""Turn HTML into clean plain text using only the standard library.

BeautifulSoup would be one line, but it is outside this project's dependency
budget and hides exactly the kind of mechanic the project exists to show.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Content inside these never belongs in the text.
DROPPED_TAGS = frozenset({"script", "style", "noscript", "head", "svg", "math"})

# These force a paragraph break around their content.
BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "header", "footer", "li", "tr",
        "blockquote", "pre", "figure", "figcaption", "table",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
)

PARAGRAPH_BREAK = "\n\n"


def normalize_whitespace(text: str) -> str:
    """Collapse spaces and runs of blank lines, preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)          # spaces/tabs, but not newlines
    text = re.sub(r" *\n *", "\n", text)           # trim around newlines
    text = re.sub(r"\n{3,}", PARAGRAPH_BREAK, text)  # 3+ newlines -> exactly one break
    return text.strip()


class HTMLTextExtractor(HTMLParser):
    """Collects visible text, inserting breaks at block boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in DROPPED_TAGS:
            self._drop_depth += 1
        elif tag == "br":
            self._parts.append("\n")
        elif tag in BLOCK_TAGS:
            self._parts.append(PARAGRAPH_BREAK)

    def handle_endtag(self, tag: str) -> None:
        if tag in DROPPED_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
        elif tag in BLOCK_TAGS:
            self._parts.append(PARAGRAPH_BREAK)

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return normalize_whitespace("".join(self._parts))


def html_to_text(html: str) -> str:
    """Extract normalised plain text from an HTML document."""
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    extractor.close()
    return extractor.text()
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_html_text.py -v`
Expected: 11 passed.

If `test_br_becomes_a_single_newline_not_a_paragraph_break` fails because `normalize_whitespace` collapses the single newline: confirm the regex `\n{3,}` is used, not `\n{2,}`. A single `\n` must survive.

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/test_html_text.py
git commit -m "feat: standard-library HTML to text extraction

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Corpus fetch script, and fetch the corpus

**Files:**
- Create: `scripts/fetch_corpus.py`, `data/corpus/*.txt` (generated), `data/metadata.json` (generated)
- Test: `tests/test_fetch_corpus.py`

**Interfaces:**
- Consumes: `scripts.html_text.html_to_text`
- Produces:
  - `SOURCES: list[Source]` where `Source` is a frozen dataclass with `doc_id: str`, `kind: str` (`"arxiv"` or `"web"`), `ref: str`, `title: str`, `author: str`, `publish_date: str`
  - `source_url(source: Source) -> str`
  - `metadata_record(source: Source) -> dict[str, str]`
  - `fetch_text(url: str, opener=urllib.request.urlopen) -> str`
  - `main(argv: list[str] | None = None) -> int`

The pure functions are tested; the network call is injected so tests never touch it.

- [ ] **Step 1: Write the failing test**

`tests/test_fetch_corpus.py`:

```python
import pytest

from scripts.fetch_corpus import (
    SOURCES,
    Source,
    fetch_text,
    metadata_record,
    source_url,
)


def test_arxiv_sources_resolve_to_ar5iv_html():
    src = Source(
        doc_id="raptor",
        kind="arxiv",
        ref="2401.18059",
        title="RAPTOR",
        author="Sarthi et al.",
        publish_date="2024-01-31",
    )
    assert source_url(src) == "https://ar5iv.labs.arxiv.org/html/2401.18059"


def test_web_sources_use_their_ref_verbatim():
    src = Source(
        doc_id="blog",
        kind="web",
        ref="https://example.invalid/post",
        title="A Post",
        author="Someone",
        publish_date="2024-03-01",
    )
    assert source_url(src) == "https://example.invalid/post"


def test_unknown_kind_is_rejected():
    src = Source("x", "carrier-pigeon", "ref", "T", "A", "2024-01-01")
    with pytest.raises(ValueError, match="carrier-pigeon"):
        source_url(src)


def test_metadata_record_has_every_field_the_loader_expects():
    src = Source(
        doc_id="raptor",
        kind="arxiv",
        ref="2401.18059",
        title="RAPTOR",
        author="Sarthi et al.",
        publish_date="2024-01-31",
    )
    record = metadata_record(src)
    assert record == {
        "title": "RAPTOR",
        "source": "arxiv",
        "publish_date": "2024-01-31",
        "author": "Sarthi et al.",
        "url": "https://ar5iv.labs.arxiv.org/html/2401.18059",
    }


def test_fetch_text_strips_markup_via_the_injected_opener():
    class FakeResponse:
        def read(self):
            return b"<html><body><p>Hello</p><script>x</script></body></html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_opener(request, timeout=0):
        return FakeResponse()

    assert fetch_text("https://example.invalid", opener=fake_opener) == "Hello"


def test_doc_ids_are_unique():
    ids = [s.doc_id for s in SOURCES]
    assert len(ids) == len(set(ids))


def test_doc_ids_are_safe_filenames():
    import re
    for s in SOURCES:
        assert re.fullmatch(r"[a-z0-9_]+", s.doc_id), s.doc_id


def test_corpus_is_large_enough_for_raptor_clustering():
    assert len(SOURCES) >= 35


def test_publish_dates_are_iso_and_span_the_2024_boundary():
    import datetime
    dates = [datetime.date.fromisoformat(s.publish_date) for s in SOURCES]
    assert any(d.year < 2024 for d in dates)
    assert any(d.year >= 2024 for d in dates)
```

The last test exists because Phase 4's query construction demo needs `publish_date < 2024` to actually partition the corpus. A corpus where every document is from 2024 would make that feature undemonstrable.

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_fetch_corpus.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.fetch_corpus'`.

- [ ] **Step 3: Implement `scripts/fetch_corpus.py`**

```python
"""Download the corpus and write data/corpus/*.txt plus data/metadata.json.

Run once; the output is committed so the repository works offline and the
evaluation gold set stays pinned to a corpus that cannot drift.

    python -m scripts.fetch_corpus
    python -m scripts.fetch_corpus --only raptor --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from scripts.html_text import html_to_text

AR5IV = "https://ar5iv.labs.arxiv.org/html/"
USER_AGENT = "rag-from-scratch/0.1 (educational project)"
MIN_CHARS = 2000  # below this, ar5iv probably served an error page


@dataclass(frozen=True)
class Source:
    doc_id: str
    kind: str          # "arxiv" | "web"
    ref: str           # arXiv id, or a full URL
    title: str
    author: str
    publish_date: str  # ISO YYYY-MM-DD


# NOTE: these arXiv ids and dates are a starting list and are NOT verified.
# Step 5 of this task prints the resolved title for each one; correct any row
# whose printed title does not match, and delete any that will not fetch.
SOURCES: list[Source] = [
    Source("rag", "arxiv", "2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "Lewis et al.", "2020-05-22"),
    Source("dpr", "arxiv", "2004.04906", "Dense Passage Retrieval for Open-Domain Question Answering", "Karpukhin et al.", "2020-04-10"),
    Source("realm", "arxiv", "2002.08909", "REALM: Retrieval-Augmented Language Model Pre-Training", "Guu et al.", "2020-02-10"),
    Source("colbert", "arxiv", "2004.12832", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction", "Khattab and Zaharia", "2020-04-27"),
    Source("colbertv2", "arxiv", "2112.01488", "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction", "Santhanam et al.", "2021-12-02"),
    Source("hyde", "arxiv", "2212.10496", "Precise Zero-Shot Dense Retrieval without Relevance Labels", "Gao et al.", "2022-12-20"),
    Source("raptor", "arxiv", "2401.18059", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", "Sarthi et al.", "2024-01-31"),
    Source("self_rag", "arxiv", "2310.11511", "Self-RAG: Learning to Retrieve, Generate, and Critique", "Asai et al.", "2023-10-17"),
    Source("step_back", "arxiv", "2310.06117", "Take a Step Back: Evoking Reasoning via Abstraction", "Zheng et al.", "2023-10-09"),
    Source("ircot", "arxiv", "2212.10509", "Interleaving Retrieval with Chain-of-Thought Reasoning", "Trivedi et al.", "2022-12-20"),
    Source("cot", "arxiv", "2201.11903", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "Wei et al.", "2022-01-28"),
    Source("least_to_most", "arxiv", "2205.10625", "Least-to-Most Prompting Enables Complex Reasoning", "Zhou et al.", "2022-05-21"),
    Source("react", "arxiv", "2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models", "Yao et al.", "2022-10-06"),
    Source("rag_survey", "arxiv", "2312.10997", "Retrieval-Augmented Generation for Large Language Models: A Survey", "Gao et al.", "2023-12-18"),
    Source("beir", "arxiv", "2104.08663", "BEIR: A Heterogeneous Benchmark for Zero-shot Information Retrieval", "Thakur et al.", "2021-04-17"),
    Source("sbert", "arxiv", "1908.10084", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "Reimers and Gurevych", "2019-08-27"),
    Source("bert", "arxiv", "1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers", "Devlin et al.", "2018-10-11"),
    Source("fid", "arxiv", "2007.01282", "Leveraging Passage Retrieval with Generative Models", "Izacard and Grave", "2020-07-02"),
    Source("ragas", "arxiv", "2309.15217", "RAGAS: Automated Evaluation of Retrieval Augmented Generation", "Es et al.", "2023-09-26"),
    Source("crag", "arxiv", "2401.15884", "Corrective Retrieval Augmented Generation", "Yan et al.", "2024-01-29"),
    Source("query_rewriting", "arxiv", "2305.14283", "Query Rewriting for Retrieval-Augmented Large Language Models", "Ma et al.", "2023-05-23"),
    Source("ir_llm_survey", "arxiv", "2308.07107", "Large Language Models for Information Retrieval: A Survey", "Zhu et al.", "2023-08-14"),
    Source("splade", "arxiv", "2107.05720", "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking", "Formal et al.", "2021-07-12"),
    Source("contriever", "arxiv", "2112.09118", "Unsupervised Dense Information Retrieval with Contrastive Learning", "Izacard et al.", "2021-12-16"),
    Source("atlas", "arxiv", "2208.03299", "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "Izacard et al.", "2022-08-05"),
    Source("replug", "arxiv", "2301.12652", "REPLUG: Retrieval-Augmented Black-Box Language Models", "Shi et al.", "2023-01-30"),
    Source("lost_in_middle", "arxiv", "2307.03172", "Lost in the Middle: How Language Models Use Long Contexts", "Liu et al.", "2023-07-06"),
    Source("hotpotqa", "arxiv", "1809.09600", "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", "Yang et al.", "2018-09-25"),
    Source("natural_questions", "arxiv", "1906.00300", "Latent Retrieval for Weakly Supervised Open Domain Question Answering", "Lee et al.", "2019-06-01"),
    Source("rankgpt", "arxiv", "2304.09542", "Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents", "Sun et al.", "2023-04-19"),
    Source("multi_vector", "arxiv", "2005.00181", "Sparse, Dense, and Attentional Representations for Text Retrieval", "Luan et al.", "2020-05-01"),
    Source("prompt_survey", "arxiv", "2107.13586", "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods", "Liu et al.", "2021-07-28"),
    Source("instructor", "arxiv", "2212.09741", "One Embedder, Any Task: Instruction-Finetuned Text Embeddings", "Su et al.", "2022-12-19"),
    Source("gtr", "arxiv", "2112.07899", "Large Dual Encoders Are Generalizable Retrievers", "Ni et al.", "2021-12-15"),
    Source("mteb", "arxiv", "2210.07316", "MTEB: Massive Text Embedding Benchmark", "Muennighoff et al.", "2022-10-13"),
    Source("longformer", "arxiv", "2004.05150", "Longformer: The Long-Document Transformer", "Beltagy et al.", "2020-04-10"),
    Source("t5", "arxiv", "1910.10683", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "Raffel et al.", "2019-10-23"),
    Source("attention", "arxiv", "1706.03762", "Attention Is All You Need", "Vaswani et al.", "2017-06-12"),
]


def source_url(source: Source) -> str:
    if source.kind == "arxiv":
        return f"{AR5IV}{source.ref}"
    if source.kind == "web":
        return source.ref
    raise ValueError(f"unknown source kind: {source.kind}")


def metadata_record(source: Source) -> dict[str, str]:
    return {
        "title": source.title,
        "source": source.kind,
        "publish_date": source.publish_date,
        "author": source.author,
        "url": source_url(source),
    }


def fetch_text(url: str, opener=urllib.request.urlopen) -> str:
    """Fetch a URL and return its visible text."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(request, timeout=60) as response:
        raw = response.read()
    return html_to_text(raw.decode("utf-8", errors="replace"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the corpus.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.json"))
    parser.add_argument("--only", action="append", help="fetch only these doc_ids")
    parser.add_argument("--force", action="store_true", help="refetch existing files")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    args = parser.parse_args(argv)

    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    selected = [s for s in SOURCES if not args.only or s.doc_id in args.only]

    documents: dict[str, dict[str, str]] = {}
    if args.metadata.is_file():
        documents = json.loads(args.metadata.read_text(encoding="utf-8"))["documents"]

    failures: list[str] = []
    for source in selected:
        target = args.corpus_dir / f"{source.doc_id}.txt"
        if target.is_file() and not args.force:
            print(f"skip   {source.doc_id} (exists)")
            documents[source.doc_id] = metadata_record(source)
            continue
        try:
            text = fetch_text(source_url(source))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"FAIL   {source.doc_id}: {exc}", file=sys.stderr)
            failures.append(source.doc_id)
            continue
        if len(text) < MIN_CHARS:
            print(f"FAIL   {source.doc_id}: only {len(text)} chars", file=sys.stderr)
            failures.append(source.doc_id)
            continue
        target.write_text(text, encoding="utf-8")
        documents[source.doc_id] = metadata_record(source)
        print(f"ok     {source.doc_id}  {len(text):>7} chars  {source.title[:60]}")
        time.sleep(args.delay)

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps({"documents": documents}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{len(documents)} documents in {args.metadata}")
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_fetch_corpus.py -v`
Expected: 9 passed.

- [ ] **Step 5: Fetch the corpus for real**

This is the one step in Phase 1 that needs the network. It takes several minutes because of the polite 2-second delay.

Run: `python -m scripts.fetch_corpus`

Expected: a line per document with a character count and the resolved title. Then check three things:

1. **Any `FAIL` lines.** Re-run just those with `--only <doc_id>`. If a paper genuinely has no ar5iv rendering, delete its row from `SOURCES` and find a replacement — the corpus must stay at 35 or more or `test_corpus_is_large_enough_for_raptor_clustering` fails.
2. **Character counts.** Anything under ~10,000 characters is probably an abstract page rather than full text. Investigate before accepting it.
3. **Titles.** The printed title comes from the hardcoded `SOURCES` row, not from the fetched page — so it does *not* verify the arXiv id. Spot-check five documents by opening `data/corpus/<doc_id>.txt` and confirming the first heading matches the title in `SOURCES`. Fix any mismatched ids and refetch with `--force`.

- [ ] **Step 6: Sanity-check the fetched text**

```bash
python -c "
from pathlib import Path
files = sorted(Path('data/corpus').glob('*.txt'))
print(f'{len(files)} files')
for f in files[:3]:
    text = f.read_text(encoding='utf-8')
    print(f'--- {f.stem} ({len(text)} chars) ---')
    print(text[:300].replace(chr(10), ' '))
"
```

Expected: readable prose. If you see runs of LaTeX macros or navigation boilerplate, add the offending tag to `DROPPED_TAGS` in `scripts/html_text.py`, add a regression test for it in `tests/test_html_text.py`, and refetch with `--force`.

- [ ] **Step 7: Commit the script and the corpus**

```bash
git add scripts/fetch_corpus.py tests/test_fetch_corpus.py data/corpus data/metadata.json
git commit -m "feat: corpus fetch script, and fetch the corpus

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Document loader

**Files:**
- Create: `rag/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `tiny_corpus` fixture.
- Produces:
  - `Document` — frozen dataclass: `doc_id: str`, `text: str`, `title: str`, `source: str`, `publish_date: str | None`, `author: str | None`, `url: str | None`
  - `load_documents(corpus_dir: Path, metadata_path: Path) -> list[Document]` — sorted by `doc_id`

- [ ] **Step 1: Write the failing test**

`tests/test_loader.py`:

```python
import json

import pytest

from rag.config import Config
from rag.loader import Document, load_documents


def test_loads_every_document(tiny_corpus: Config):
    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert [d.doc_id for d in docs] == ["alpha", "beta"]


def test_attaches_metadata_to_text(tiny_corpus: Config):
    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    alpha = docs[0]
    assert isinstance(alpha, Document)
    assert alpha.title == "Cosine Similarity"
    assert alpha.publish_date == "2023-05-01"
    assert alpha.author == "A. Author"
    assert alpha.source == "notes"
    assert "Cosine similarity" in alpha.text


def test_result_is_sorted_by_doc_id_for_reproducible_chunk_ids(tiny_corpus: Config):
    (tiny_corpus.corpus_dir / "aardvark.txt").write_text("Zebra text.", encoding="utf-8")
    meta = json.loads(tiny_corpus.metadata_path.read_text(encoding="utf-8"))
    meta["documents"]["aardvark"] = {
        "title": "Aardvark", "source": "notes",
        "publish_date": "2022-01-01", "author": "C. Author", "url": None,
    }
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")

    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert [d.doc_id for d in docs] == ["aardvark", "alpha", "beta"]


def test_text_file_without_metadata_is_an_error(tiny_corpus: Config):
    (tiny_corpus.corpus_dir / "orphan.txt").write_text("No metadata.", encoding="utf-8")
    with pytest.raises(ValueError, match="orphan"):
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)


def test_metadata_without_text_file_is_an_error(tiny_corpus: Config):
    meta = json.loads(tiny_corpus.metadata_path.read_text(encoding="utf-8"))
    meta["documents"]["ghost"] = {
        "title": "Ghost", "source": "notes",
        "publish_date": "2022-01-01", "author": None, "url": None,
    }
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="ghost"):
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)


def test_empty_text_file_is_an_error(tiny_corpus: Config):
    (tiny_corpus.corpus_dir / "alpha.txt").write_text("   \n  ", encoding="utf-8")
    with pytest.raises(ValueError, match="alpha"):
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)


def test_missing_corpus_directory_is_an_error(tiny_corpus: Config):
    with pytest.raises(FileNotFoundError):
        load_documents(tiny_corpus.corpus_dir / "nope", tiny_corpus.metadata_path)


def test_optional_metadata_fields_default_to_none(tiny_corpus: Config):
    meta = {"documents": {"alpha": {"title": "T", "source": "notes"}, }}
    (tiny_corpus.corpus_dir / "beta.txt").unlink()
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert docs[0].publish_date is None
    assert docs[0].author is None
    assert docs[0].url is None
```

The strictness matters: a silently skipped document produces an index that is quietly missing content, and the symptom appears much later as a retrieval miss that looks like a ranking problem.

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_loader.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.loader'`.

- [ ] **Step 3: Implement `rag/loader.py`**

```python
"""Load the committed corpus into Document objects.

Mismatches between the text files and the metadata raise rather than warn: a
silently dropped document shows up much later as a retrieval miss that looks
like a ranking bug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str
    source: str
    publish_date: str | None = None
    author: str | None = None
    url: str | None = None


def load_documents(corpus_dir: Path, metadata_path: Path) -> list[Document]:
    """Read every .txt in corpus_dir, pairing it with its metadata entry.

    Returns documents sorted by doc_id, so chunk ids are stable across runs.
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["documents"]
    text_files = {p.stem: p for p in corpus_dir.glob("*.txt")}

    orphans = sorted(set(text_files) - set(metadata))
    if orphans:
        raise ValueError(
            f"text files with no metadata entry: {', '.join(orphans)}"
        )
    ghosts = sorted(set(metadata) - set(text_files))
    if ghosts:
        raise ValueError(
            f"metadata entries with no text file: {', '.join(ghosts)}"
        )

    documents: list[Document] = []
    for doc_id in sorted(text_files):
        text = text_files[doc_id].read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"document is empty: {doc_id}")
        record = metadata[doc_id]
        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                title=record["title"],
                source=record["source"],
                publish_date=record.get("publish_date"),
                author=record.get("author"),
                url=record.get("url"),
            )
        )
    return documents
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_loader.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/loader.py tests/test_loader.py
git commit -m "feat: corpus loader with strict metadata pairing

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Token-aware chunking

**Files:**
- Create: `rag/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `rag.loader.Document`, `tests.conftest.FakeTokenizer`
- Produces:
  - `Chunk` — frozen dataclass: `chunk_id: str`, `doc_id: str`, `index: int`, `text: str`, `token_start: int`, `token_end: int`, `char_start: int`, `char_end: int`
  - `window_bounds(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]` — half-open `[start, end)` pairs
  - `chunk_document(doc: Document, tokenizer, size: int, overlap: int) -> list[Chunk]`
  - `chunk_documents(docs: list[Document], tokenizer, size: int, overlap: int) -> list[Chunk]`

`window_bounds` is pure integer arithmetic with no tokenizer, which is what makes the off-by-one cases cheap to test exhaustively.

- [ ] **Step 1: Write the failing test**

`tests/test_chunking.py`:

```python
import pytest

from rag.chunking import Chunk, chunk_document, chunk_documents, window_bounds
from rag.loader import Document
from tests.conftest import FakeTokenizer


# --- pure window arithmetic -------------------------------------------------

def test_windows_advance_by_size_minus_overlap():
    assert window_bounds(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]


def test_final_window_is_clipped_not_padded():
    assert window_bounds(9, 4, 1) == [(0, 4), (3, 7), (6, 9)]


def test_single_window_when_document_is_shorter_than_the_window():
    assert window_bounds(3, 4, 1) == [(0, 3)]


def test_exact_fit_produces_one_window():
    assert window_bounds(4, 4, 1) == [(0, 4)]


def test_no_windows_for_empty_input():
    assert window_bounds(0, 4, 1) == []


def test_no_trailing_window_that_is_wholly_covered_by_its_predecessor():
    # With size=4, overlap=1, n=10: a naive range() would emit (9, 10), whose
    # single token already appears in (6, 10).
    bounds = window_bounds(10, 4, 1)
    assert bounds[-1] == (6, 10)
    assert all(end <= 10 for _, end in bounds)


def test_zero_overlap_produces_disjoint_windows():
    assert window_bounds(9, 3, 0) == [(0, 3), (3, 6), (6, 9)]


def test_overlap_equal_to_size_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        window_bounds(10, 4, 4)


def test_nonpositive_size_is_rejected():
    with pytest.raises(ValueError, match="size"):
        window_bounds(10, 0, 0)


def test_every_token_appears_in_at_least_one_window():
    bounds = window_bounds(37, 8, 3)
    covered = {i for start, end in bounds for i in range(start, end)}
    assert covered == set(range(37))


# --- chunking a document ----------------------------------------------------

def _doc(text: str) -> Document:
    return Document(doc_id="d", text=text, title="T", source="s")


def test_chunk_text_is_sliced_from_the_original_not_detokenized():
    doc = _doc("alpha beta gamma delta epsilon")
    chunks = chunk_document(doc, FakeTokenizer(), size=2, overlap=0)
    assert chunks[0].text == "alpha beta"
    assert doc.text[chunks[0].char_start:chunks[0].char_end] == chunks[0].text


def test_char_offsets_round_trip_for_every_chunk():
    doc = _doc(" ".join(f"word{i}" for i in range(30)))
    for chunk in chunk_document(doc, FakeTokenizer(), size=7, overlap=2):
        assert doc.text[chunk.char_start:chunk.char_end] == chunk.text


def test_chunk_ids_are_doc_id_colon_index():
    doc = _doc("a b c d e f")
    chunks = chunk_document(doc, FakeTokenizer(), size=2, overlap=0)
    assert [c.chunk_id for c in chunks] == ["d:0", "d:1", "d:2"]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_token_bounds_are_recorded():
    doc = _doc("a b c d e")
    chunks = chunk_document(doc, FakeTokenizer(), size=3, overlap=1)
    assert (chunks[0].token_start, chunks[0].token_end) == (0, 3)
    assert (chunks[1].token_start, chunks[1].token_end) == (2, 5)


def test_empty_document_yields_no_chunks():
    assert chunk_document(_doc(""), FakeTokenizer(), size=4, overlap=1) == []


def test_slow_tokenizer_is_rejected_because_offsets_are_required():
    class SlowTokenizer:
        is_fast = False

        def __call__(self, *a, **kw):
            raise AssertionError("should not be called")

    with pytest.raises(ValueError, match="fast tokenizer"):
        chunk_document(_doc("a b"), SlowTokenizer(), size=2, overlap=0)


def test_chunk_documents_concatenates_in_document_order():
    docs = [_doc("a b c"), Document(doc_id="e", text="x y z", title="T", source="s")]
    chunks = chunk_documents(docs, FakeTokenizer(), size=2, overlap=0)
    assert [c.doc_id for c in chunks] == ["d", "d", "e", "e"]


def test_chunks_are_returned_as_the_dataclass():
    chunks = chunk_document(_doc("a b"), FakeTokenizer(), size=2, overlap=0)
    assert isinstance(chunks[0], Chunk)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_chunking.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.chunking'`.

- [ ] **Step 3: Implement `rag/chunking.py`**

```python
"""Token-aware sliding-window chunking.

Character-based splitting is the standard beginner bug here: MiniLM truncates
at 256 tokens, so a 2000-character chunk loses its tail with no error raised.
Sliding over token ids instead means every chunk fits the model by construction.

Character offsets come from the tokenizer's offset mapping, so chunk text is a
slice of the original document rather than a detokenised approximation. Phase 3
needs those exact offsets to score span-tagged gold questions.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    token_start: int
    token_end: int   # exclusive
    char_start: int
    char_end: int    # exclusive


def window_bounds(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Half-open [start, end) windows covering n_tokens.

    Stops as soon as a window reaches the end, so no trailing stub window is
    emitted whose content is already wholly contained in its predecessor.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size}), "
            "or the window never advances"
        )
    if n_tokens <= 0:
        return []

    step = size - overlap
    bounds: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + size, n_tokens)
        bounds.append((start, end))
        if end >= n_tokens:
            return bounds
        start += step


def chunk_document(doc: Document, tokenizer, size: int, overlap: int) -> list[Chunk]:
    """Split one document into overlapping token windows."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "chunking needs a fast tokenizer for offset_mapping; "
            "load it with AutoTokenizer.from_pretrained(..., use_fast=True)"
        )

    encoded = tokenizer(
        doc.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]

    chunks: list[Chunk] = []
    for index, (token_start, token_end) in enumerate(
        window_bounds(len(offsets), size, overlap)
    ):
        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{index}",
                doc_id=doc.doc_id,
                index=index,
                text=doc.text[char_start:char_end],
                token_start=token_start,
                token_end=token_end,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return chunks


def chunk_documents(
    docs: list[Document], tokenizer, size: int, overlap: int
) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, tokenizer, size, overlap))
    return chunks
```

`add_special_tokens=False` is deliberate: `[CLS]` and `[SEP]` are added later at encode time, and counting them here would make each chunk two tokens larger than intended. With `chunk_tokens=200` and `max_seq_tokens=256` there is ample headroom either way, but the arithmetic should mean what it says.

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_chunking.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/chunking.py tests/test_chunking.py
git commit -m "feat: token-aware sliding window chunking with char offsets

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Embeddings

**Files:**
- Create: `rag/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `rag.config.Config`
- Produces:
  - `mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray` — `(batch, seq, dim)` and `(batch, seq)` to `(batch, dim)`
  - `l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray`
  - `Embedder(model_name: str, max_length: int = 256, device: str = "cpu")` with `.tokenizer`, `.dim: int`, `.encode(texts: list[str], batch_size: int = 32) -> np.ndarray`

The two pooling functions are plain NumPy and get hand-computed tests. The `Embedder` class is thin glue and gets one `@pytest.mark.slow` test that actually loads the model.

- [ ] **Step 1: Write the failing test**

`tests/test_embedding.py`:

```python
import numpy as np
import pytest

from rag.embedding import Embedder, l2_normalize, mean_pool


# --- mean pooling -----------------------------------------------------------

def test_mean_pool_averages_only_unmasked_positions():
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = np.array([[1, 1, 0]])
    # (1+3)/2 = 2, (2+4)/2 = 3 — the third position is padding and must not count.
    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 3.0]])


def test_mean_pool_with_all_positions_unmasked():
    hidden = np.array([[[1.0, 1.0], [3.0, 5.0]]])
    mask = np.array([[1, 1]])
    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 3.0]])


def test_mean_pool_handles_a_fully_masked_row_without_dividing_by_zero():
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = np.array([[0, 0]])
    result = mean_pool(hidden, mask)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [[0.0, 0.0]])


def test_mean_pool_is_independent_across_batch_rows():
    hidden = np.array([
        [[1.0, 1.0], [3.0, 3.0]],
        [[10.0, 0.0], [0.0, 0.0]],
    ])
    mask = np.array([[1, 1], [1, 0]])
    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 2.0], [10.0, 0.0]])


def test_mean_pool_returns_float32():
    hidden = np.ones((1, 2, 2), dtype=np.float64)
    mask = np.ones((1, 2), dtype=np.int64)
    assert mean_pool(hidden, mask).dtype == np.float32


# --- L2 normalisation -------------------------------------------------------

def test_l2_normalize_produces_unit_vectors():
    np.testing.assert_allclose(l2_normalize(np.array([[3.0, 4.0]])), [[0.6, 0.8]])


def test_l2_normalize_leaves_a_zero_vector_at_zero():
    result = l2_normalize(np.array([[0.0, 0.0]]))
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [[0.0, 0.0]])


def test_l2_normalize_acts_per_row():
    result = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 1.0]])


def test_l2_normalize_is_idempotent():
    once = l2_normalize(np.array([[3.0, 4.0], [1.0, 1.0]]))
    np.testing.assert_allclose(l2_normalize(once), once, atol=1e-6)


# --- the real model ---------------------------------------------------------

@pytest.mark.slow
def test_real_embedder_shape_and_normalisation():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedder.encode(["cosine similarity", "rank fusion"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-5)


@pytest.mark.slow
def test_real_embedder_places_related_text_closer_than_unrelated():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    v = embedder.encode([
        "cosine similarity between two vectors",
        "measuring the angle between vectors",
        "a recipe for banana bread",
    ])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


@pytest.mark.slow
def test_real_embedder_batching_matches_single_pass():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    texts = [f"sentence number {i}" for i in range(5)]
    np.testing.assert_allclose(
        embedder.encode(texts, batch_size=2),
        embedder.encode(texts, batch_size=32),
        atol=1e-5,
    )


@pytest.mark.slow
def test_real_embedder_returns_empty_matrix_for_empty_input():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    assert embedder.encode([]).shape == (0, 384)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_embedding.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.embedding'`. The four `slow` tests are deselected.

- [ ] **Step 3: Implement `rag/embedding.py`**

```python
"""Local embeddings via HuggingFace transformers.

Pooling and normalisation are written out rather than imported. A transformer
returns one vector per token; turning that into one vector per text is a choice,
and mean pooling over the attention mask is the choice all-MiniLM-L6-v2 was
trained with. Getting the mask wrong — averaging over padding — quietly degrades
every downstream similarity score, which is exactly the class of bug a framework
would hide.

Vectors are L2-normalised on the way out, which makes cosine similarity a plain
dot product later.
"""

from __future__ import annotations

import numpy as np


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average token vectors, counting only unmasked positions.

    hidden: (batch, seq, dim); mask: (batch, seq) of 0/1.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)[..., None]   # (batch, seq, 1)
    summed = (hidden * mask).sum(axis=1)                   # (batch, dim)
    counts = np.clip(mask.sum(axis=1), 1.0, None)          # never divide by zero
    return (summed / counts).astype(np.float32)


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit length; all-zero rows are left alone."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.maximum(norms, eps)).astype(np.float32)


class Embedder:
    """Wraps a HuggingFace encoder. Loads the model once, on construction."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 256,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts into an (n, dim) float32 matrix of unit vectors."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        pooled_batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
            pooled_batches.append(
                mean_pool(
                    hidden.cpu().numpy(),
                    encoded["attention_mask"].cpu().numpy(),
                )
            )
        return l2_normalize(np.vstack(pooled_batches))
```

`torch` and `transformers` are imported inside `__init__` so that importing `rag.embedding` for the pure functions costs nothing — `import torch` alone takes several seconds, and the fast test suite imports this module.

- [ ] **Step 4: Run the fast tests and verify they pass**

Run: `python -m pytest tests/test_embedding.py -v`
Expected: 9 passed, 4 deselected.

- [ ] **Step 5: Run the slow tests once, against the real model**

This downloads roughly 90MB on first run and needs the network.

Run: `python -m pytest tests/test_embedding.py -v -m slow`
Expected: 4 passed. First run takes a minute or two; afterwards the model is cached locally.

If `test_real_embedder_shape_and_normalisation` reports a dim other than 384, the wrong model was loaded — check the identifier for a typo.

- [ ] **Step 6: Commit**

```bash
git add rag/embedding.py tests/test_embedding.py
git commit -m "feat: embeddings with hand-written mean pooling and L2 norm

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Similarity and top-k

**Files:**
- Create: `rag/similarity.py`
- Test: `tests/test_similarity.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `cosine_similarity(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray` — `(n_queries, n_items)`
  - `top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]` — `(indices, scores)`, each `(n_queries, k')` where `k' = min(k, n_items)`

- [ ] **Step 1: Write the failing test**

`tests/test_similarity.py`:

```python
import numpy as np
import pytest

from rag.similarity import cosine_similarity, top_k


# --- cosine -----------------------------------------------------------------

def test_identical_vectors_score_one():
    v = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(cosine_similarity(v, v), [[1.0]], atol=1e-6)


def test_orthogonal_vectors_score_zero():
    q = np.array([[1.0, 0.0]])
    m = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(cosine_similarity(q, m), [[0.0]], atol=1e-6)


def test_opposite_vectors_score_minus_one():
    q = np.array([[1.0, 0.0]])
    m = np.array([[-1.0, 0.0]])
    np.testing.assert_allclose(cosine_similarity(q, m), [[-1.0]], atol=1e-6)


def test_magnitude_is_ignored():
    q = np.array([[1.0, 0.0]])
    m = np.array([[100.0, 0.0]])
    np.testing.assert_allclose(cosine_similarity(q, m), [[1.0]], atol=1e-6)


def test_matches_the_textbook_formula():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 5))
    m = rng.normal(size=(7, 5))
    expected = np.array([
        [
            float(qi @ mj / (np.linalg.norm(qi) * np.linalg.norm(mj)))
            for mj in m
        ]
        for qi in q
    ])
    np.testing.assert_allclose(cosine_similarity(q, m), expected, atol=1e-5)


def test_output_shape_is_queries_by_items():
    assert cosine_similarity(np.ones((3, 4)), np.ones((7, 4))).shape == (3, 7)


def test_a_one_dimensional_query_is_treated_as_a_single_query():
    assert cosine_similarity(np.ones(4), np.ones((7, 4))).shape == (1, 7)


def test_zero_vector_scores_zero_rather_than_nan():
    result = cosine_similarity(np.zeros((1, 3)), np.ones((2, 3)))
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [[0.0, 0.0]])


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity(np.ones((1, 4)), np.ones((2, 5)))


# --- top-k ------------------------------------------------------------------

def test_returns_highest_scores_in_descending_order():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    indices, values = top_k(scores, 2)
    np.testing.assert_array_equal(indices, [[1, 3]])
    np.testing.assert_allclose(values, [[0.9, 0.7]])


def test_ties_are_broken_by_lower_index_for_determinism():
    scores = np.array([[0.5, 0.9, 0.9, 0.1]])
    indices, _ = top_k(scores, 2)
    np.testing.assert_array_equal(indices, [[1, 2]])


def test_k_larger_than_the_corpus_returns_everything():
    scores = np.array([[0.1, 0.9]])
    indices, values = top_k(scores, 10)
    assert indices.shape == (1, 2)
    np.testing.assert_array_equal(indices, [[1, 0]])


def test_k_of_zero_returns_empty():
    indices, values = top_k(np.array([[0.1, 0.9]]), 0)
    assert indices.shape == (1, 0)
    assert values.shape == (1, 0)


def test_negative_k_is_rejected():
    with pytest.raises(ValueError, match="k"):
        top_k(np.array([[0.1]]), -1)


def test_each_query_row_is_ranked_independently():
    scores = np.array([[0.1, 0.9], [0.9, 0.1]])
    indices, _ = top_k(scores, 1)
    np.testing.assert_array_equal(indices, [[1], [0]])


def test_empty_corpus_yields_empty_results():
    indices, values = top_k(np.zeros((2, 0)), 5)
    assert indices.shape == (2, 0)
    assert values.shape == (2, 0)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_similarity.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.similarity'`.

- [ ] **Step 3: Implement `rag/similarity.py`**

```python
"""Similarity search: exact, brute-force, and written out.

This is the whole of "the vector database". Scoring every query against every
chunk is one matrix multiply, which at a few thousand chunks is faster than the
overhead of an approximate index would be. It is O(n) per query, and past
roughly a million vectors an ANN index (FAISS, HNSW) becomes the right answer.

Inputs are normalised defensively even though Embedder already returns unit
vectors, so these functions are correct in isolation.
"""

from __future__ import annotations

import numpy as np


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_similarity(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of every query against every row of matrix.

    Returns an (n_queries, n_items) array. All-zero rows score 0, not NaN.
    """
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    if queries.shape[1] != matrix.shape[1]:
        raise ValueError(
            f"dimension mismatch: queries are {queries.shape[1]}-d, "
            f"items are {matrix.shape[1]}-d"
        )
    return _unit_rows(queries) @ _unit_rows(matrix).T


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the k highest scores per row, descending.

    Ties break toward the lower index: argsort is stable, so equal scores keep
    their original order. Without that, two runs could rank identical chunks
    differently and the benchmark would drift for no reason.
    """
    if k < 0:
        raise ValueError(f"k must not be negative, got {k}")
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float32))
    k = min(k, scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(scores, order, axis=1)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_similarity.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/similarity.py tests/test_similarity.py
git commit -m "feat: cosine similarity and stable top-k in NumPy

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Vector store

**Files:**
- Create: `rag/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `rag.chunking.Chunk`, `rag.similarity.cosine_similarity`, `rag.similarity.top_k`
- Produces:
  - `VectorStore(vectors: np.ndarray, chunks: list[Chunk])` with `__len__`, `dim`
  - `VectorStore.search(query_vectors: np.ndarray, k: int) -> list[list[tuple[Chunk, float]]]` — one result list per query row
  - `VectorStore.save(path: Path) -> None`
  - `VectorStore.load(path: Path) -> VectorStore` (classmethod)

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:

```python
import numpy as np
import pytest

from rag.chunking import Chunk
from rag.store import VectorStore


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"d:{i}",
        doc_id="d",
        index=i,
        text=f"chunk {i}",
        token_start=i * 10,
        token_end=i * 10 + 10,
        char_start=i * 50,
        char_end=i * 50 + 50,
    )


def _store(n: int = 3) -> VectorStore:
    vectors = np.eye(n, dtype=np.float32)
    return VectorStore(vectors=vectors, chunks=[_chunk(i) for i in range(n)])


def test_length_is_the_chunk_count():
    assert len(_store(3)) == 3


def test_dim_reports_the_vector_width():
    assert _store(3).dim == 3


def test_vector_and_chunk_counts_must_agree():
    with pytest.raises(ValueError, match="3 vectors"):
        VectorStore(vectors=np.eye(3, dtype=np.float32), chunks=[_chunk(0)])


def test_search_returns_the_nearest_chunk_first():
    store = _store(3)
    query = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    results = store.search(query, k=1)
    assert len(results) == 1
    chunk, score = results[0][0]
    assert chunk.chunk_id == "d:1"
    assert score == pytest.approx(1.0, abs=1e-6)


def test_search_returns_one_result_list_per_query():
    store = _store(3)
    queries = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    results = store.search(queries, k=2)
    assert [r[0][0].chunk_id for r in results] == ["d:0", "d:2"]


def test_search_results_are_ordered_by_descending_score():
    store = _store(3)
    query = np.array([[0.9, 0.4, 0.1]], dtype=np.float32)
    scores = [score for _, score in store.search(query, k=3)[0]]
    assert scores == sorted(scores, reverse=True)


def test_search_k_larger_than_the_store_returns_everything():
    assert len(_store(2).search(np.array([[1.0, 0.0]], dtype=np.float32), k=10)[0]) == 2


def test_search_on_an_empty_store_returns_empty_results():
    store = VectorStore(vectors=np.zeros((0, 4), dtype=np.float32), chunks=[])
    assert store.search(np.ones((1, 4), dtype=np.float32), k=5) == [[]]


def test_scores_are_plain_floats_not_numpy_scalars():
    # The trace is serialised to JSON for the dashboard; np.float32 is not
    # JSON-serialisable and the failure would surface far from here.
    _, score = _store(2).search(np.array([[1.0, 0.0]], dtype=np.float32), k=1)[0][0]
    assert type(score) is float


def test_save_and_load_round_trip(tmp_path):
    store = _store(3)
    path = tmp_path / "index.npz"
    store.save(path)
    loaded = VectorStore.load(path)
    np.testing.assert_allclose(loaded.vectors, store.vectors)
    assert loaded.chunks == store.chunks


def test_loaded_vectors_are_float32(tmp_path):
    path = tmp_path / "index.npz"
    _store(3).save(path)
    assert VectorStore.load(path).vectors.dtype == np.float32


def test_save_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "index.npz"
    _store(2).save(path)
    assert path.is_file()


def test_loading_a_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope.npz")
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.store'`.

- [ ] **Step 3: Implement `rag/store.py`**

```python
"""The vector store: a dense NumPy matrix and a parallel list of chunks.

Row i of `vectors` is the embedding of `chunks[i]`. That invariant is the whole
data structure — there is no index, no graph, no quantisation. Persistence is
.npz with the chunk metadata alongside as a JSON blob, loaded with
allow_pickle=False so the index file is never an arbitrary-code vector.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.similarity import cosine_similarity, top_k


@dataclass
class VectorStore:
    vectors: np.ndarray       # (n_chunks, dim) float32
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if self.vectors.shape[0] != len(self.chunks):
            raise ValueError(
                f"{self.vectors.shape[0]} vectors but {len(self.chunks)} chunks; "
                "row i must be the embedding of chunks[i]"
            )

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(
        self, query_vectors: np.ndarray, k: int
    ) -> list[list[tuple[Chunk, float]]]:
        """Nearest chunks for each query row, best first."""
        query_vectors = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        if len(self.chunks) == 0:
            return [[] for _ in range(query_vectors.shape[0])]
        scores = cosine_similarity(query_vectors, self.vectors)
        indices, values = top_k(scores, k)
        return [
            [(self.chunks[int(i)], float(s)) for i, s in zip(row_i, row_s)]
            for row_i, row_s in zip(indices, values)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False)
        np.savez_compressed(path, vectors=self.vectors, chunks=np.array(payload))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        if not path.is_file():
            raise FileNotFoundError(
                f"index not found: {path} — run `python -m rag index` first"
            )
        with np.load(path, allow_pickle=False) as data:
            vectors = data["vectors"].astype(np.float32)
            chunks = [Chunk(**record) for record in json.loads(str(data["chunks"]))]
        return cls(vectors=vectors, chunks=chunks)
```

Note `float(s)` and `int(i)` in `search`: NumPy scalars leak into the trace otherwise, and `json.dumps` rejects `np.float32` with an error that surfaces in the dashboard rather than here.

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/store.py tests/test_store.py
git commit -m "feat: NumPy vector store with npz persistence

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Trace

**Files:**
- Create: `rag/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `rag.chunking.Chunk`
- Produces:
  - `StageTiming(name: str, ms: float)`
  - `RetrievedChunk(chunk: Chunk, score: float, rank: int)`
  - `Trace(question: str)` with mutable fields `queries: list[str]`, `retrieved: list[RetrievedChunk]`, `prompt: str | None`, `answer: str | None`, `timings: list[StageTiming]`, `notes: list[str]`
  - `Trace.stage(name: str)` — context manager appending a `StageTiming`
  - `Trace.note(message: str) -> None`
  - `Trace.total_ms -> float`
  - `Trace.to_dict() -> dict` — JSON-serialisable

Phase 1 only fills `queries` with the single original question. Later phases add rewritten queries without changing the shape.

- [ ] **Step 1: Write the failing test**

`tests/test_trace.py`:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_trace.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.trace'`.

- [ ] **Step 3: Implement `rag/trace.py`**

```python
"""The Trace: one record of everything a single question caused.

Three consumers read this object and nothing else — the CLI's --trace output,
the Phase 6 dashboard, and the Phase 3 benchmark. Keeping them behind one
representation is what stops the dashboard growing logic of its own.

Later phases add fields (rewritten queries, the inferred metadata filter, the
routing decision) without changing this shape.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Iterator

from rag.chunking import Chunk


@dataclass
class StageTiming:
    name: str
    ms: float


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int   # 1-based


@dataclass
class Trace:
    question: str
    queries: list[str] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    prompt: str | None = None
    answer: str | None = None
    timings: list[StageTiming] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and record it, whether or not the block raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.timings.append(StageTiming(name=name, ms=elapsed_ms))

    def note(self, message: str) -> None:
        """Record something the reader needs to know, such as a degradation."""
        self.notes.append(message)

    @property
    def total_ms(self) -> float:
        return sum(t.ms for t in self.timings)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "queries": list(self.queries),
            "retrieved": [
                {
                    "chunk": asdict(r.chunk),
                    "score": float(r.score),
                    "rank": int(r.rank),
                }
                for r in self.retrieved
            ],
            "prompt": self.prompt,
            "answer": self.answer,
            "timings": [{"name": t.name, "ms": t.ms} for t in self.timings],
            "notes": list(self.notes),
            "total_ms": self.total_ms,
        }
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_trace.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/trace.py tests/test_trace.py
git commit -m "feat: Trace object recording every pipeline stage

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Gemini client with caching and retry

**Files:**
- Create: `rag/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `rag.config.Config`
- Produces:
  - `LLMError(Exception)`
  - `cache_key(model: str, prompt: str, temperature: float) -> str` — 64-char sha256 hex
  - `GeminiLLM(model, api_key, cache_dir, temperature=0.0, max_retries=5, client=None, sleep=time.sleep)` with `.generate(prompt: str) -> str` and `.call_count: int`

The client and the sleep function are injected so retry and cache behaviour can be tested without the network and without actually waiting.

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_llm.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.llm'`.

- [ ] **Step 3: Implement `rag/llm.py`**

```python
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
```

Note that `LLMError` from an empty response is re-raised rather than retried: an empty completion at temperature 0 will be empty again, so retrying only burns quota.

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_llm.py -v`
Expected: 17 passed.

- [ ] **Step 5: Confirm the default model id against the real account**

The `llm_model` default in `rag/config.py` was written without verifying it exists for your key. Check it and list the alternatives:

```bash
python -c "
import os
from google import genai
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
for m in client.models.list():
    if 'generateContent' in getattr(m, 'supported_actions', []) or True:
        print(m.name)
" | grep -i flash
```

Pick the cheapest current flash model from that list. If it differs from `gemini-2.0-flash`, update the `llm_model` default in `rag/config.py` and add its exact name to `tests/test_config.py::test_defaults_match_the_spec`.

If the command errors with an auth failure, set the key first: `$env:GOOGLE_API_KEY = "..."` in PowerShell, or create `.env` from `.env.example`.

- [ ] **Step 6: Commit**

```bash
git add rag/llm.py rag/config.py tests/test_llm.py tests/test_config.py
git commit -m "feat: Gemini client with on-disk cache and backoff

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Prompt template and answer generation

**Files:**
- Create: `rag/prompts.py`, `rag/generation.py`
- Test: `tests/test_prompts.py`, `tests/test_generation.py`

**Interfaces:**
- Consumes: `rag.chunking.Chunk`, `rag.trace.Trace`, `tests.conftest.FakeLLM`
- Produces:
  - `ANSWER_TEMPLATE: str` — contains `{context}` and `{question}`
  - `format_context(retrieved: list[tuple[Chunk, float]]) -> str`
  - `build_answer_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str`
  - `generate_answer(llm, question: str, retrieved: list[tuple[Chunk, float]], trace: Trace) -> str | None`

- [ ] **Step 1: Write the failing tests**

`tests/test_prompts.py`:

```python
from rag.chunking import Chunk
from rag.prompts import ANSWER_TEMPLATE, build_answer_prompt, format_context


def _retrieved():
    return [
        (Chunk("alpha:0", "alpha", 0, "Cosine ignores magnitude.", 0, 5, 0, 25), 0.91),
        (Chunk("beta:2", "beta", 2, "RRF sums reciprocal ranks.", 0, 5, 0, 26), 0.77),
    ]


def test_template_has_both_placeholders():
    assert "{context}" in ANSWER_TEMPLATE
    assert "{question}" in ANSWER_TEMPLATE


def test_template_instructs_the_model_to_stay_in_context():
    assert "context" in ANSWER_TEMPLATE.lower()


def test_context_numbers_chunks_from_one():
    context = format_context(_retrieved())
    assert "[1]" in context
    assert "[2]" in context
    assert "[0]" not in context


def test_context_includes_chunk_text():
    context = format_context(_retrieved())
    assert "Cosine ignores magnitude." in context
    assert "RRF sums reciprocal ranks." in context


def test_context_labels_the_source_document():
    context = format_context(_retrieved())
    assert "alpha" in context
    assert "beta" in context


def test_context_shows_the_score():
    assert "0.910" in format_context(_retrieved())


def test_empty_retrieval_produces_an_explicit_marker_not_a_blank():
    # A blank context makes the model hallucinate freely; saying so does not.
    assert format_context([]) == "(no documents retrieved)"


def test_prompt_contains_the_question_and_the_context():
    prompt = build_answer_prompt("What is RRF?", _retrieved())
    assert "What is RRF?" in prompt
    assert "RRF sums reciprocal ranks." in prompt


def test_prompt_has_no_unfilled_placeholders():
    prompt = build_answer_prompt("q", _retrieved())
    assert "{context}" not in prompt
    assert "{question}" not in prompt


def test_braces_in_chunk_text_do_not_break_formatting():
    chunks = [(Chunk("a:0", "a", 0, "code: {'k': 1}", 0, 5, 0, 14), 0.5)]
    prompt = build_answer_prompt("q", chunks)
    assert "{'k': 1}" in prompt
```

`tests/test_generation.py`:

```python
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
```

The last test is the error-handling rule from the spec in miniature: generation failing must leave a usable trace with retrieved chunks and a recorded reason, not an exception out of the CLI.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_prompts.py tests/test_generation.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.prompts'`.

- [ ] **Step 3: Implement `rag/prompts.py`**

```python
"""Every prompt template in the project, in one file.

Prompts are the interface to the model and they change often; keeping them
together means a change is one diff in one place rather than a hunt through the
strategies. Later phases append their templates here.
"""

from __future__ import annotations

from rag.chunking import Chunk

NO_CONTEXT = "(no documents retrieved)"

ANSWER_TEMPLATE = """You are answering a question using retrieved excerpts.

Rules:
- Use only the context below. Do not use prior knowledge.
- If the context does not answer the question, say so plainly and stop.
- Cite the excerpts you used with their bracketed numbers, like [1] or [2].
- Be concise.

Context:
{context}

Question: {question}

Answer:"""


def format_context(retrieved: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as a numbered, citable block."""
    if not retrieved:
        return NO_CONTEXT
    blocks = []
    for number, (chunk, score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[{number}] source: {chunk.doc_id}, chunk {chunk.index}, "
            f"score {score:.3f}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_answer_prompt(
    question: str, retrieved: list[tuple[Chunk, float]]
) -> str:
    """Fill the answer template. Chunk text may contain braces; it is not
    re-formatted, so `{}` in a document cannot break or inject anything."""
    return ANSWER_TEMPLATE.format(
        context=format_context(retrieved), question=question
    )
```

- [ ] **Step 4: Implement `rag/generation.py`**

```python
"""Turn retrieved chunks into an answer."""

from __future__ import annotations

from rag.chunking import Chunk
from rag.llm import LLMError
from rag.prompts import build_answer_prompt
from rag.trace import Trace


def generate_answer(
    llm,
    question: str,
    retrieved: list[tuple[Chunk, float]],
    trace: Trace,
) -> str | None:
    """Generate an answer, recording prompt, answer and timing on the trace.

    Returns None and records a note if the model call fails. Retrieval already
    succeeded at this point, so the trace is still worth showing — failing hard
    would throw away the useful half of the result.
    """
    prompt = build_answer_prompt(question, retrieved)
    trace.prompt = prompt
    with trace.stage("generate"):
        try:
            answer = llm.generate(prompt)
        except LLMError as exc:
            trace.note(f"generation failed: {exc}")
            return None
    trace.answer = answer
    return answer
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest tests/test_prompts.py tests/test_generation.py -v`
Expected: 10 passed, 6 passed.

- [ ] **Step 6: Commit**

```bash
git add rag/prompts.py rag/generation.py tests/test_prompts.py tests/test_generation.py
git commit -m "feat: answer prompt template and generation with graceful failure

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Pipeline

**Files:**
- Create: `rag/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `build_index(config: Config, embedder) -> VectorStore` — loads, chunks, embeds, saves
  - `load_index(config: Config) -> VectorStore`
  - `ask(question: str, store: VectorStore, embedder, llm, config: Config, k: int | None = None) -> Trace` — `llm=None` means retrieval only

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:

```python
import pytest

from rag.config import Config
from rag.pipeline import ask, build_index, load_index
from rag.store import VectorStore
from tests.conftest import FakeEmbedder, FakeLLM


# --- indexing ---------------------------------------------------------------

def test_build_index_covers_every_document(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    assert {c.doc_id for c in store.chunks} == {"alpha", "beta"}


def test_build_index_produces_one_vector_per_chunk(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    assert store.vectors.shape[0] == len(store.chunks)
    assert store.vectors.shape[1] == FakeEmbedder.dim


def test_build_index_writes_the_index_file(tiny_corpus: Config):
    build_index(tiny_corpus, FakeEmbedder())
    assert tiny_corpus.index_path.is_file()


def test_build_index_is_reproducible(tiny_corpus: Config):
    first = build_index(tiny_corpus, FakeEmbedder())
    second = build_index(tiny_corpus, FakeEmbedder())
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]


def test_load_index_round_trips(tiny_corpus: Config):
    built = build_index(tiny_corpus, FakeEmbedder())
    loaded = load_index(tiny_corpus)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in built.chunks]


def test_load_index_without_building_first_raises(tiny_corpus: Config):
    with pytest.raises(FileNotFoundError, match="rag index"):
        load_index(tiny_corpus)


# --- asking -----------------------------------------------------------------

def test_ask_returns_a_trace_with_the_question(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("what is cosine?", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert trace.question == "what is cosine?"


def test_ask_records_the_original_question_as_the_only_query(tiny_corpus: Config):
    # Phase 2 adds rewritten queries here; in Phase 1 there is exactly one.
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert trace.queries == ["q"]


def test_ask_retrieves_top_k_chunks(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, k=2)
    assert len(trace.retrieved) == 2


def test_ask_ranks_retrieved_chunks_from_one(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, k=3)
    assert [r.rank for r in trace.retrieved] == [1, 2, 3]


def test_ask_orders_retrieved_chunks_by_descending_score(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, k=3)
    scores = [r.score for r in trace.retrieved]
    assert scores == sorted(scores, reverse=True)


def test_ask_defaults_k_to_the_config(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert len(trace.retrieved) == tiny_corpus.top_k


def test_ask_records_the_answer(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM("an answer"), tiny_corpus)
    assert trace.answer == "an answer"


def test_ask_times_every_stage(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert [t.name for t in trace.timings] == ["embed", "search", "generate"]


def test_ask_without_an_llm_retrieves_but_does_not_generate(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), None, tiny_corpus)
    assert trace.retrieved
    assert trace.answer is None
    assert any("retrieval only" in n for n in trace.notes)
    assert [t.name for t in trace.timings] == ["embed", "search"]


def test_ask_gives_the_llm_the_retrieved_context(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM()
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, k=1)
    assert trace.retrieved[0].chunk.text in llm.prompts[0]
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.pipeline'`.

- [ ] **Step 3: Implement `rag/pipeline.py`**

```python
"""Wiring. This module composes the others and owns no algorithm of its own.

Later phases insert routing, query construction and translation between the
question arriving and the search running; the Trace shape does not change.
"""

from __future__ import annotations

from rag.chunking import chunk_documents
from rag.config import Config
from rag.generation import generate_answer
from rag.loader import load_documents
from rag.store import VectorStore
from rag.trace import RetrievedChunk, Trace


def build_index(config: Config, embedder) -> VectorStore:
    """Load the corpus, chunk it, embed it, and persist the result."""
    documents = load_documents(config.corpus_dir, config.metadata_path)
    chunks = chunk_documents(
        documents, embedder.tokenizer, config.chunk_tokens, config.chunk_overlap
    )
    vectors = embedder.encode([chunk.text for chunk in chunks])
    store = VectorStore(vectors=vectors, chunks=chunks)
    store.save(config.index_path)
    return store


def load_index(config: Config) -> VectorStore:
    return VectorStore.load(config.index_path)


def ask(
    question: str,
    store: VectorStore,
    embedder,
    llm,
    config: Config,
    k: int | None = None,
) -> Trace:
    """Answer one question. Pass llm=None to retrieve without generating."""
    trace = Trace(question=question)
    trace.queries = [question]
    k = config.top_k if k is None else k

    with trace.stage("embed"):
        query_vectors = embedder.encode([question])

    with trace.stage("search"):
        results = store.search(query_vectors, k)[0]

    trace.retrieved = [
        RetrievedChunk(chunk=chunk, score=score, rank=rank)
        for rank, (chunk, score) in enumerate(results, start=1)
    ]

    if llm is None:
        trace.note("retrieval only: no LLM configured")
        return trace

    generate_answer(llm, question, results, trace)
    return trace
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline wiring index build and question answering

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: CLI

**Files:**
- Create: `rag/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `rag.pipeline`, `rag.config.Config`, `rag.embedding.Embedder`, `rag.llm.GeminiLLM`
- Produces:
  - `main(argv: list[str] | None = None, embedder_factory=..., llm_factory=...) -> int`
  - `format_trace(trace: Trace, verbose: bool) -> str`

The two factories are injected so the CLI is testable without loading a model or holding a key.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
import pytest

from rag.__main__ import format_trace, main
from rag.chunking import Chunk
from rag.config import Config
from rag.trace import RetrievedChunk, StageTiming, Trace
from tests.conftest import FakeEmbedder, FakeLLM


def _trace() -> Trace:
    trace = Trace(question="what is RRF?")
    trace.queries = ["what is RRF?"]
    trace.retrieved = [
        RetrievedChunk(
            chunk=Chunk("beta:1", "beta", 1, "RRF sums reciprocal ranks.", 0, 5, 0, 26),
            score=0.83,
            rank=1,
        )
    ]
    trace.answer = "It fuses ranked lists. [1]"
    trace.timings = [StageTiming("embed", 12.5), StageTiming("search", 0.4)]
    return trace


# --- output formatting ------------------------------------------------------

def test_output_includes_the_answer():
    assert "It fuses ranked lists. [1]" in format_trace(_trace(), verbose=False)


def test_output_lists_sources_even_without_verbose():
    # Sources are the point of RAG; hiding them behind a flag defeats it.
    assert "beta" in format_trace(_trace(), verbose=False)


def test_verbose_output_includes_scores():
    assert "0.83" in format_trace(_trace(), verbose=True)


def test_verbose_output_includes_timings():
    output = format_trace(_trace(), verbose=True)
    assert "embed" in output
    assert "12.5" in output


def test_verbose_output_includes_the_prompt():
    trace = _trace()
    trace.prompt = "THE PROMPT TEXT"
    assert "THE PROMPT TEXT" in format_trace(trace, verbose=True)


def test_non_verbose_output_omits_the_prompt():
    trace = _trace()
    trace.prompt = "THE PROMPT TEXT"
    assert "THE PROMPT TEXT" not in format_trace(trace, verbose=False)


def test_notes_are_always_shown():
    trace = _trace()
    trace.note("generation failed: rate limited")
    assert "rate limited" in format_trace(trace, verbose=False)


def test_missing_answer_is_reported_not_printed_as_none():
    trace = _trace()
    trace.answer = None
    output = format_trace(trace, verbose=False)
    assert "None" not in output


# --- command dispatch -------------------------------------------------------

def _factories():
    return {
        "embedder_factory": lambda config: FakeEmbedder(),
        "llm_factory": lambda config: FakeLLM("stub answer"),
    }


def test_index_command_builds_the_index(tiny_corpus: Config, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    assert main(["index"], **_factories()) == 0
    assert tiny_corpus.index_path.is_file()
    assert "chunks" in capsys.readouterr().out


def test_ask_command_prints_an_answer(tiny_corpus: Config, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "what is cosine?"], **_factories()) == 0
    assert "stub answer" in capsys.readouterr().out


def test_ask_without_an_index_fails_with_guidance(
    tiny_corpus: Config, monkeypatch, capsys
):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    assert main(["ask", "q"], **_factories()) == 1
    assert "rag index" in capsys.readouterr().err


def test_no_llm_flag_skips_generation(tiny_corpus: Config, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())

    def exploding_llm_factory(config):
        raise AssertionError("--no-llm must not construct an LLM")

    assert main(
        ["ask", "q", "--no-llm"],
        embedder_factory=lambda config: FakeEmbedder(),
        llm_factory=exploding_llm_factory,
    ) == 0
    assert "retrieval only" in capsys.readouterr().out


def test_no_subcommand_prints_usage_and_fails():
    with pytest.raises(SystemExit):
        main([])
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: `ImportError` / `ModuleNotFoundError` for `rag.__main__`.

- [ ] **Step 3: Implement `rag/__main__.py`**

```python
"""Command line entry point.

    python -m rag index
    python -m rag ask "what is reciprocal rank fusion?"
    python -m rag ask "..." --k 8 --trace
    python -m rag ask "..." --no-llm       # retrieval only, no API key needed

The embedder and LLM are built by injected factories so the CLI can be tested
without loading a model or holding a key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import Config
from rag.embedding import Embedder
from rag.llm import GeminiLLM, LLMError
from rag.pipeline import ask, build_index, load_index
from rag.trace import Trace


def load_config(**overrides) -> Config:
    return Config.from_env(env_file=Path(".env"), **overrides)


def default_embedder(config: Config) -> Embedder:
    return Embedder(config.embedding_model, max_length=config.max_seq_tokens)


def default_llm(config: Config) -> GeminiLLM:
    return GeminiLLM(
        model=config.llm_model,
        api_key=config.api_key,
        cache_dir=config.cache_dir,
    )


def format_trace(trace: Trace, verbose: bool) -> str:
    lines: list[str] = []

    if trace.answer:
        lines.append(trace.answer)
    else:
        lines.append("(no answer generated)")
    lines.append("")

    lines.append("Sources:")
    for item in trace.retrieved:
        chunk = item.chunk
        if verbose:
            lines.append(
                f"  [{item.rank}] {chunk.chunk_id}  score {item.score:.3f}  "
                f"chars {chunk.char_start}-{chunk.char_end}"
            )
            lines.append(f"      {chunk.text[:160]}")
        else:
            lines.append(f"  [{item.rank}] {chunk.doc_id} (chunk {chunk.index})")

    for note in trace.notes:
        lines.append(f"note: {note}")

    if verbose:
        lines.append("")
        lines.append("Timings:")
        for timing in trace.timings:
            lines.append(f"  {timing.name:<10} {timing.ms:8.1f} ms")
        lines.append(f"  {'total':<10} {trace.total_ms:8.1f} ms")
        if trace.prompt:
            lines.append("")
            lines.append("Prompt sent:")
            lines.append(trace.prompt)

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="RAG from scratch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build the vector index")
    index_parser.add_argument("--chunk-tokens", type=int)
    index_parser.add_argument("--chunk-overlap", type=int)

    ask_parser = subparsers.add_parser("ask", help="answer a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--k", type=int, help="number of chunks to retrieve")
    ask_parser.add_argument("--trace", action="store_true", help="show scores, timings, prompt")
    ask_parser.add_argument("--no-llm", action="store_true", help="retrieve only")

    return parser


def main(
    argv: list[str] | None = None,
    embedder_factory=default_embedder,
    llm_factory=default_llm,
) -> int:
    args = _build_parser().parse_args(argv)

    overrides = {}
    for field in ("chunk_tokens", "chunk_overlap"):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    config = load_config(**overrides)

    if args.command == "index":
        embedder = embedder_factory(config)
        store = build_index(config, embedder)
        documents = len({c.doc_id for c in store.chunks})
        print(
            f"indexed {documents} documents into {len(store)} chunks "
            f"({store.dim}-d) -> {config.index_path}"
        )
        return 0

    try:
        store = load_index(config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    embedder = embedder_factory(config)
    llm = None
    if not args.no_llm:
        try:
            llm = llm_factory(config)
        except LLMError as exc:
            print(f"{exc}\nretrieving without generation", file=sys.stderr)

    trace = ask(args.question, store, embedder, llm, config, k=args.k)
    print(format_trace(trace, verbose=args.trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 13 passed.

- [ ] **Step 5: Build the real index**

Run: `python -m rag index`

Expected: something like `indexed 38 documents into 2400 chunks (384-d) -> data\index.npz`. Takes a few minutes on CPU.

Sanity-check the chunk count: roughly `total_characters / 4 / 150`. Wildly fewer chunks than that means chunking silently produced one chunk per document — check that `window_bounds` is being called with the config values and not defaults.

- [ ] **Step 6: Run the real pipeline end to end**

```bash
python -m rag ask "What is reciprocal rank fusion?" --trace
python -m rag ask "How does ColBERT score a document?" --trace
python -m rag ask "What problem does HyDE solve?" --no-llm
```

Expected: grounded answers citing bracketed sources, with the cited documents being plausibly the right ones — the RRF question should retrieve from `rag_survey` or `rankgpt`, the ColBERT question from `colbert` or `colbertv2`.

If retrieval looks random, check in this order: that `build_index` embedded `chunk.text` and not `doc.text`; that `VectorStore.search` uses `self.vectors` rows in the same order as `self.chunks`; that the query is embedded by the same model as the chunks.

- [ ] **Step 7: Commit**

```bash
git add rag/__main__.py tests/test_cli.py
git commit -m "feat: CLI with index and ask commands

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: README, dependency guard, and phase wrap-up

**Files:**
- Modify: `README.md`
- Test: `tests/test_no_frameworks.py`

**Interfaces:**
- Consumes: the whole package.
- Produces: nothing importable.

- [ ] **Step 1: Write the failing test**

`tests/test_no_frameworks.py`:

```python
"""The project's central claim, enforced.

A reviewer's first question is whether "no frameworks" is actually true. This
test answers it, and stops a future phase from quietly importing a wrapper.
"""

import re
from pathlib import Path

import pytest

FORBIDDEN = [
    "langchain",
    "llama_index",
    "llamaindex",
    "sentence_transformers",
    "haystack",
    "chromadb",
    "faiss",
    "pinecone",
    "sklearn",
    "bs4",
    "requests",
    "dotenv",
]

SOURCE_DIRS = ["rag", "scripts", "tests"]


def _python_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return [
        path
        for directory in SOURCE_DIRS
        for path in (root / directory).rglob("*.py")
    ]


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_forbidden_package_is_not_imported(forbidden: str):
    pattern = re.compile(
        rf"^\s*(?:import\s+{forbidden}|from\s+{forbidden}[\s.])", re.MULTILINE
    )
    offenders = [
        str(path)
        for path in _python_files()
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"{forbidden} imported in: {offenders}"


def test_there_are_python_files_to_check():
    # Guards against the glob silently matching nothing and the suite
    # passing vacuously.
    assert len(_python_files()) > 10
```

`sentence_transformers` with an underscore is the library; the model identifier `sentence-transformers/all-MiniLM-L6-v2` is a hyphenated string and is not matched by these patterns.

- [ ] **Step 2: Run the test and verify it passes immediately**

Run: `python -m pytest tests/test_no_frameworks.py -v`
Expected: 13 passed.

This is the one test in the plan that should pass on first run — it asserts an existing property rather than driving new code. If it fails, a forbidden import crept in and must be removed.

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest -v`
Expected: all tests pass, `slow` and `live` deselected. Confirm the deselected count is 4.

Then confirm the slow suite still passes: `python -m pytest -m slow -v` → 4 passed.

- [ ] **Step 4: Write the README**

Replace `README.md` entirely:

````markdown
# RAG from Scratch

A retrieval-augmented generation pipeline built without a RAG framework. No
LangChain, no LlamaIndex, no vector database, no `sentence-transformers`. The
text splitter, the embedding pooling, the similarity math and the prompt
assembly are all written out.

Implements the techniques in `RAG.pdf`, phase by phase. Phase 1 — the core
indexing, retrieval and generation loop — is complete.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env        # add your GOOGLE_API_KEY
python -m rag index
python -m rag ask "What is reciprocal rank fusion?" --trace
```

Retrieval works without an API key:

```bash
python -m rag ask "What problem does HyDE solve?" --no-llm
```

## How it works

| Stage | What happens | Where |
|---|---|---|
| Load | 38 documents on RAG and retrieval, fetched from ar5iv and committed | `rag/loader.py` |
| Chunk | Sliding window of 200 tokens with 50 overlap, over token ids | `rag/chunking.py` |
| Embed | all-MiniLM-L6-v2, mean pooling over the attention mask, L2 normalised | `rag/embedding.py` |
| Store | Dense `(n_chunks, 384)` float32 NumPy matrix | `rag/store.py` |
| Retrieve | Cosine similarity as one matmul, stable top-k | `rag/similarity.py` |
| Generate | Numbered context injected into a prompt template, sent to Gemini | `rag/prompts.py` |

Every stage writes into one `Trace` object (`rag/trace.py`), which is what
`--trace` prints and what the benchmark and dashboard will read in later phases.

## Three things worth pointing out

**Chunking is token-aware, not character-aware.** MiniLM truncates at 256
tokens. A character-based splitter produces chunks whose tails the model
silently discards — no error, just quietly worse retrieval. Sliding over token
ids instead means every chunk fits by construction.

**Embeddings are 384-dimensional, not 3-dimensional.** Diagrams draw vector
space in 3D because 3D is drawable. Nothing here assumes three dimensions.

**Search is exact brute force, and that is a deliberate limit.** Scoring every
query against every chunk is one `(n_queries, 384) @ (384, n_chunks)` matmul,
which at a few thousand chunks is sub-millisecond — faster than the overhead an
approximate index would add. It is O(n) per query. Past roughly a million
vectors, FAISS or HNSW becomes the right answer.

## Tests

```bash
python -m pytest              # fast: no network, no model, no API key
python -m pytest -m slow      # loads the real embedding model
```

`tests/test_no_frameworks.py` enforces the central claim: it fails if any
forbidden package is imported anywhere in the project.

## Roadmap

- [x] **Phase 1** — core pipeline
- [ ] **Phase 2** — query translation: multi-query, RAG-Fusion, decomposition, step-back, HyDE
- [ ] **Phase 3** — evaluation harness: Recall@k, MRR, nDCG
- [ ] **Phase 4** — routing and query construction
- [ ] **Phase 5** — multi-representation indexing and RAPTOR
- [ ] **Phase 6** — ColBERT-style late interaction, and a retrieval inspector dashboard
- [ ] **Phase 7** — full gold set and final benchmark table

Design: [`docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md`](docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md)
````

Two things to reconcile before committing:

- **Document count.** Use whatever Task 3 actually produced, not 38.
- **`RAG.pdf` is not in the repository.** Either copy the source notes in as
  `docs/RAG.pdf` and link them, or drop the reference and point only at the
  design spec. A README citing a file that is not there is the first thing a
  reviewer notices.

- [ ] **Step 5: Verify the README's commands actually work**

Run each command in the Quick start block, from a clean shell, in order. Fix the README, not your memory of it, if any command fails.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_no_frameworks.py
git commit -m "docs: README and a test enforcing the no-frameworks claim

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 1 Definition of Done

- [ ] `python -m pytest` passes with zero failures and no network access
- [ ] `python -m pytest -m slow` passes (4 tests, real model)
- [ ] `python -m rag index` produces an index over 35 or more documents
- [ ] `python -m rag ask "What is reciprocal rank fusion?" --trace` returns a grounded answer citing plausible sources
- [ ] `python -m rag ask "..." --no-llm` works with no API key set
- [ ] `tests/test_no_frameworks.py` passes
- [ ] README's Quick start block has been executed end to end as written
- [ ] Working tree clean

## What Phase 2 will need from this phase

Listed so that nothing here gets refactored away without noticing:

- `Trace.queries` is a list, already populated with the single original question. Multi-query appends to it.
- `VectorStore.search` accepts a `(n_queries, dim)` matrix and returns one result list per row — RAG-Fusion needs the multi-row form.
- `Chunk.chunk_id` is the identity used for deduplication and for reciprocal rank fusion.
- `rag/prompts.py` is where translation templates will be added.
- `generate_answer` takes `retrieved` as a plain `list[tuple[Chunk, float]]`, which is what fusion produces.
