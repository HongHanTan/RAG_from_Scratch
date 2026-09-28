# Phase 4: Routing and Query Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `RAG.pdf` Stages 2 and 3 — logical routing, semantic routing, and natural-language-to-metadata-filter query construction — and measure them with the Phase 3 benchmark.

**Architecture:** A structured-output helper on the Gemini client returns validated JSON instead of prose, and three features are built on it: logical routing picks which topical collections to search, query construction turns a question into a `MetadataFilter` compiled to a NumPy boolean mask applied *before* top-k, and semantic routing picks an answer prompt by cosine similarity between the question and embedded prompt descriptions. Every one degrades to its unfiltered, unrouted default and records that it did.

**Tech Stack:** Python 3.14, NumPy, google-genai structured output, pytest. No new dependencies.

## Global Constraints

- **No RAG framework.** `langchain`, `llama_index`, `sentence_transformers`, `sklearn`, `bs4`, `requests`, `dotenv` are installed and MUST NOT be imported. `tests/test_no_frameworks.py` enforces this across `rag/`, `scripts/`, `tests/` and `evaluation/`, and scans `pyproject.toml`.
- **Allowed third-party imports:** `numpy`, `torch`, `transformers`, `google.genai`, `pytest`.
- **Determinism.** LLM temperature 0, stable sorts. The benchmark must stay reproducible.
- **Test-driven.** Write the test, run it, watch it fail for the expected reason, then implement.
- **Every unit test runs offline.** No test may call the real API or load the real embedding model unless marked `@pytest.mark.slow` or `@pytest.mark.live`; both are deselected by default.
- **Platform is Windows.** `pathlib`, never string path concatenation. `encoding="utf-8"` explicit on every file read and write.
- **Degrade, but record it.** Every LLM-dependent step falls back to a safe default and calls `trace.note(...)` with the word `degraded`. Task 8 widens the benchmark's check to catch all of them.
- **Commit after every task.** Messages end with a blank line then exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## What Phases 1–3 provide

| Thing | Signature |
|---|---|
| `rag.loader.Document` | frozen: `doc_id`, `text`, `title`, `source`, `publish_date=None`, `author=None`, `url=None` |
| `rag.chunking.Chunk` | frozen: `chunk_id`, `doc_id`, `index`, `text`, `token_start`, `token_end`, `char_start`, `char_end` |
| `rag.chunking.RetrievedChunk` | `chunk`, `score`, `rank`, `score_kind` |
| `rag.store.VectorStore` | `vectors`, `chunks`, `meta`; `.search(query_vectors, k)`, `.save(path, meta=)`, `.load(path, expect_meta=)` |
| `rag.similarity` | `cosine_similarity`, `top_k`, `reciprocal_rank_fusion`, `merge_best_score` |
| `rag.trace.Trace` | `question`, `strategy`, `queries`, `translation`, `retrieved`, `prompt`, `answer`, `timings`, `notes`; `stage(name)` (depth-aware), `note(msg)`, `add_translation(kind, text)`, `to_dict()` |
| `rag.llm.GeminiLLM` | `.generate(prompt) -> str`, `.last_call_cached`, on-disk cache keyed by model+prompt+temperature; raises `LLMError` |
| `rag.strategies` | `StrategyContext(store, embedder, llm, config, trace)` with `.search(queries, k)`; `StrategyResult(retrieved, extra_context)`; `degrade_to_direct`; `STRATEGY_NAMES`; `get_strategy(name, **options)` |
| `rag.pipeline` | `build_index(config, embedder)`, `load_index(config)`, `ask(question, store, embedder, llm, config, k=None, strategy="direct", strategy_options=None, generate=True)` |
| `rag.config.Config` | frozen; `top_k=5`, `retrieval_depth=20`, `chunk_tokens=200`, `chunk_overlap=50`, `embedding_model`, `llm_model`, `corpus_dir`, `metadata_path`, `index_path`, `cache_dir`, `api_key` |
| `evaluation/` | `load_gold`, `relevant_chunk_ids`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `doc_precision_at_k`, `score_strategy`, `format_table`, `check_not_degraded` |

Corpus: 38 documents, 5,116 chunks. Current suite: 450 passed, 5 deselected.

## Two problems this phase has to solve first

**1. There is nothing to route between.** Every document in `data/metadata.json` has `source: "arxiv"`. Logical routing — which `RAG.pdf` describes as giving the model knowledge of the available data sources so it can pick one — has a single option, and a `source` filter matches all 38 documents. Task 1 adds a `topic` field partitioning the corpus into five collections. This is safe: metadata fields do not affect chunking, the index's provenance check (`embedding_model`, `chunk_tokens`, `chunk_overlap`, `dim`), or the gold set, which keys on `doc_id` plus a quote.

**2. The pipeline cannot see document metadata.** `VectorStore` holds chunks, and a `Chunk` knows only its `doc_id`. To mask by author or date, retrieval needs per-document metadata at query time, and `load_index` does not load documents. Task 2 persists a `doc_meta` map inside the index so `ask()` stays self-contained and a reloaded index can still filter.

## Carried note from Phase 2

Step-back's question extraction went through four attempts at parsing prose out of a model reply, and the code records that the principled fix is structured output — the mechanism this phase introduces. Task 9 adopts it there, closing that thread.

## File Structure

```
rag/
  llm.py                 MODIFY: structured(prompt, schema) -> dict
  loader.py              MODIFY: Document.topic
  store.py               MODIFY: doc_meta; search(..., mask=)
  query_construction.py  NEW: MetadataFilter, compile_mask, build_filter
  routing.py             NEW: logical_route, SemanticRouter
  prompts.py             MODIFY: FILTER_SCHEMA/TEMPLATE, ROUTE_SCHEMA/TEMPLATE, prompt variants
  pipeline.py            MODIFY: ask(..., route=, filter_=) wiring
  __main__.py            MODIFY: --route, --filter, --no-route flags; trace rendering
  strategies/base.py     MODIFY: StrategyContext carries an optional mask
  strategies/step_back.py MODIFY: use structured output
evaluation/
  benchmark.py           MODIFY: widen the degradation check; --route column
data/metadata.json       MODIFY: topic per document
tests/                   test_query_construction.py, test_routing.py (new); others modified
README.md                MODIFY
```

---

### Task 1: Topic metadata

**Files:**
- Modify: `data/metadata.json`, `rag/loader.py`
- Test: `tests/test_loader.py`, `tests/test_topics.py`

**Interfaces:**
- Produces: `Document.topic: str | None = None`; every document in `data/metadata.json` carries a `topic`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_loader.py`:

```python
def test_topic_is_read_from_metadata(tiny_corpus: Config):
    import json

    meta = json.loads(tiny_corpus.metadata_path.read_text(encoding="utf-8"))
    meta["documents"]["alpha"]["topic"] = "similarity"
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert docs[0].topic == "similarity"


def test_topic_defaults_to_none_when_absent(tiny_corpus: Config):
    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert docs[0].topic is None
```

Create `tests/test_topics.py`:

```python
"""Checks on the real corpus's topic assignment.

Logical routing needs something to route between. Every document in this
corpus has source "arxiv", so `topic` is the field that partitions it.
"""

from pathlib import Path

import pytest

from rag.config import Config
from rag.loader import load_documents

EXPECTED_TOPICS = {
    "foundations",
    "retrieval-models",
    "rag-systems",
    "prompting-reasoning",
    "evaluation-benchmarks",
}


@pytest.fixture(scope="module")
def corpus():
    cfg = Config()
    return load_documents(cfg.corpus_dir, cfg.metadata_path)


def test_every_document_has_a_topic(corpus):
    missing = [d.doc_id for d in corpus if not d.topic]
    assert not missing, f"documents with no topic: {missing}"


def test_topics_come_from_the_known_set(corpus):
    unknown = {d.topic for d in corpus} - EXPECTED_TOPICS
    assert not unknown, f"unexpected topics: {unknown}"


def test_every_topic_is_used(corpus):
    # A topic nobody is in is a routing option that can only ever be wrong.
    unused = EXPECTED_TOPICS - {d.topic for d in corpus}
    assert not unused, f"topics with no documents: {unused}"


def test_no_topic_holds_more_than_half_the_corpus(corpus):
    # A partition where one bucket is most of the corpus cannot route
    # usefully: picking it is barely different from searching everything.
    from collections import Counter

    counts = Counter(d.topic for d in corpus)
    biggest, n = counts.most_common(1)[0]
    assert n <= len(corpus) // 2, f"{biggest} holds {n} of {len(corpus)}"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_loader.py tests/test_topics.py -q`
Expected: `TypeError` for the unexpected `topic` keyword, and the topic tests failing because no document has one.

- [ ] **Step 3: Add `topic` to `Document`**

In `rag/loader.py`, add the field to the dataclass after `url`:

```python
    topic: str | None = None
    """Which topical collection this document belongs to.

    Every document in this corpus has source "arxiv", so `source` cannot
    partition it. `topic` is what logical routing chooses between.
    """
```

and read it in the loop that builds each `Document`:

```python
                topic=record.get("topic"),
```

- [ ] **Step 4: Assign topics in `data/metadata.json`**

Add a `"topic"` key to every document. Use exactly this assignment — it splits 38 documents into five collections of 4, 9, 9, 8 and 8, so no collection is more than half the corpus:

```
foundations           attention, bert, t5, longformer
retrieval-models      colbert, colbertv2, dpr, contriever, splade, gtr, sbert,
                      multi_vector, instructor
rag-systems           rag, realm, atlas, fid, replug, self_rag, crag,
                      rag_survey, raptor
prompting-reasoning   cot, least_to_most, react, ircot, step_back,
                      query_rewriting, prompt_survey, hyde
evaluation-benchmarks beir, mteb, ragas, hotpotqa, natural_questions,
                      ir_llm_survey, rankgpt, lost_in_middle
```

Write it with a script rather than by hand so the file's existing formatting and non-ASCII author names survive:

```python
python -c "
import json
from pathlib import Path
topics = {
 'foundations': ['attention','bert','t5','longformer'],
 'retrieval-models': ['colbert','colbertv2','dpr','contriever','splade','gtr','sbert','multi_vector','instructor'],
 'rag-systems': ['rag','realm','atlas','fid','replug','self_rag','crag','rag_survey','raptor'],
 'prompting-reasoning': ['cot','least_to_most','react','ircot','step_back','query_rewriting','prompt_survey','hyde'],
 'evaluation-benchmarks': ['beir','mteb','ragas','hotpotqa','natural_questions','ir_llm_survey','rankgpt','lost_in_middle'],
}
p = Path('data/metadata.json')
data = json.loads(p.read_text(encoding='utf-8'))
assigned = {d: t for t, ds in topics.items() for d in ds}
missing = set(data['documents']) - set(assigned)
extra = set(assigned) - set(data['documents'])
assert not missing, f'no topic for: {sorted(missing)}'
assert not extra, f'topic names a missing doc: {sorted(extra)}'
for doc_id, record in data['documents'].items():
    record['topic'] = assigned[doc_id]
p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + chr(10), encoding='utf-8')
print(f'assigned topics to {len(data[\"documents\"])} documents')
"
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`
Expected: all pass. Confirm nothing else moved:

```
python -c "
from pathlib import Path
from rag.config import Config
from rag.pipeline import load_index
from rag.loader import load_documents
cfg = Config()
print('documents:', len(load_documents(cfg.corpus_dir, cfg.metadata_path)))
print('index chunks:', len(load_index(cfg)))
"
```

Expect 38 and 5,116 — adding a metadata field must not change either.

- [ ] **Step 6: Commit**

```bash
git add data/metadata.json rag/loader.py tests/test_loader.py tests/test_topics.py
git commit -m "feat: topic metadata, so logical routing has something to choose

Every document has source \"arxiv\", so source cannot partition the
corpus and a source filter matches everything. Five topical collections
give routing a real decision.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The index carries document metadata

**Files:**
- Modify: `rag/store.py`, `rag/pipeline.py`
- Test: `tests/test_store.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `VectorStore.doc_meta: dict[str, dict]` — `doc_id` to `{"title","source","topic","publish_date","author","url"}`
  - persisted in the `.npz` and restored by `load`; an index written before this loads with `doc_meta == {}`
  - `build_index` populates it from the loaded documents

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py`:

```python
def _doc_meta():
    return {"d": {"title": "T", "source": "arxiv", "topic": "foundations",
                  "publish_date": "2020-01-01", "author": "A", "url": None}}


def test_doc_meta_round_trips(tmp_path):
    path = tmp_path / "index.npz"
    store = _store(3)
    store.doc_meta = _doc_meta()
    store.save(path, meta=None)
    assert VectorStore.load(path).doc_meta == _doc_meta()


def test_doc_meta_defaults_to_empty(tmp_path):
    path = tmp_path / "index.npz"
    _store(2).save(path)
    assert VectorStore.load(path).doc_meta == {}


def test_an_index_written_without_doc_meta_still_loads(tmp_path):
    # Indexes built before this field existed must not become unreadable.
    import json
    from dataclasses import asdict

    path = tmp_path / "index.npz"
    store = _store(2)
    np.savez_compressed(
        path,
        vectors=store.vectors,
        chunks=np.array(json.dumps([asdict(c) for c in store.chunks])),
    )
    loaded = VectorStore.load(path)
    assert len(loaded) == 2
    assert loaded.doc_meta == {}
```

Append to `tests/test_pipeline.py`:

```python
def test_build_index_records_document_metadata(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    assert set(store.doc_meta) == {"alpha", "beta"}
    assert store.doc_meta["alpha"]["title"] == "Cosine Similarity"


def test_load_index_restores_document_metadata(tiny_corpus: Config):
    build_index(tiny_corpus, FakeEmbedder())
    assert load_index(tiny_corpus).doc_meta["beta"]["publish_date"] == "2024-02-11"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_store.py tests/test_pipeline.py -q`
Expected: `AttributeError: 'VectorStore' object has no attribute 'doc_meta'`.

- [ ] **Step 3: Add `doc_meta` to `VectorStore`**

In `rag/store.py`, add the field after `meta`:

```python
    doc_meta: dict = field(default_factory=dict)
    """Per-document metadata, keyed by doc_id.

    Retrieval needs this to filter by author, date or topic, and a `Chunk`
    knows only its `doc_id`. Persisting it in the index keeps `ask()` able to
    filter from a reloaded index without also loading the corpus.
    """
```

In `save`, add it to the payload:

```python
                doc_meta=np.array(json.dumps(self.doc_meta, ensure_ascii=False)),
```

In `load`, read it back defensively, exactly as `meta` is:

```python
            doc_meta = (
                json.loads(str(data["doc_meta"])) if "doc_meta" in data.files else {}
            )
```

and pass it to the constructor.

- [ ] **Step 4: Populate it in `build_index`**

In `rag/pipeline.py`, after constructing the store:

```python
    store.doc_meta = {
        doc.doc_id: {
            "title": doc.title,
            "source": doc.source,
            "topic": doc.topic,
            "publish_date": doc.publish_date,
            "author": doc.author,
            "url": doc.url,
        }
        for doc in documents
    }
```

and make sure `store.save(...)` runs after that assignment.

- [ ] **Step 5: Run the tests, then rebuild the real index**

Run: `python -m pytest -q` — all passing.

Then rebuild so the committed workflow has metadata available. This takes a few minutes.

```
python -m rag index
python -c "
from rag.config import Config
from rag.pipeline import load_index
s = load_index(Config())
print(len(s), 'chunks |', len(s.doc_meta), 'documents in doc_meta')
print('sample:', s.doc_meta['colbert'])
"
```

Expect 5,116 chunks and 38 documents. `data/index.npz` is gitignored, so nothing to commit there.

- [ ] **Step 6: Commit**

```bash
git add rag/store.py rag/pipeline.py tests/test_store.py tests/test_pipeline.py
git commit -m "feat: persist per-document metadata in the index

Filtering by author, date or topic needs document metadata at query
time, and a Chunk knows only its doc_id. Storing the map in the index
keeps ask() able to filter from a reloaded index.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Structured output on the LLM client

**Files:**
- Modify: `rag/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `GeminiLLM.structured(prompt: str, schema: dict) -> dict` — returns a parsed, shape-checked dict; raises `LLMError` when the reply cannot be used

This is the mechanism `RAG.pdf` calls for in logical routing, and the one Phase 2's step-back parser concluded it needed after four attempts at parsing prose.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
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


def test_structured_uses_the_same_cache_as_generate(tmp_path):
    llm = _llm(tmp_path, ['{"topics": ["a"]}'])
    llm.structured("p", SCHEMA)
    llm.structured("p", SCHEMA)
    assert llm.call_count == 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_llm.py -q`
Expected: `AttributeError: 'GeminiLLM' object has no attribute 'structured'`.

- [ ] **Step 3: Implement `structured`**

Add to `rag/llm.py`:

```python
import re

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
        if expected and not isinstance(value[key], expected):
            raise LLMError(
                f"field {key} should be {spec['type']}, "
                f"got {type(value[key]).__name__}"
            )
```

and the method on `GeminiLLM`:

```python
    def structured(self, prompt: str, schema: dict) -> dict:
        """Return a parsed, shape-checked JSON object from the model.

        Shares `generate`'s cache, retry and backoff. Raises LLMError when the
        reply cannot be parsed or does not match the schema — callers are
        expected to catch that and fall back to a safe default, because a
        routing or filtering step that fails should not lose the answer.
        """
        raw = self.generate(prompt)
        try:
            parsed = json.loads(_extract_json_object(raw))
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"model reply was not valid JSON: {exc}; got {raw[:120]!r}"
            ) from exc
        _check_shape(parsed, schema)
        return parsed
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_llm.py -q`

- [ ] **Step 5: Commit**

```bash
git add rag/llm.py tests/test_llm.py
git commit -m "feat: structured output with schema validation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: MetadataFilter and mask compilation

**Files:**
- Create: `rag/query_construction.py`
- Test: `tests/test_query_construction.py`

**Interfaces:**
- Produces:
  - `MetadataFilter` — frozen: `topics: tuple[str, ...] = ()`, `authors: tuple[str, ...] = ()`, `published_before: str | None = None`, `published_after: str | None = None`
  - `MetadataFilter.is_empty() -> bool`
  - `MetadataFilter.describe() -> str` — human-readable, for the trace
  - `MetadataFilter.matches(record: dict) -> bool`
  - `compile_mask(filter_: MetadataFilter, chunks: list[Chunk], doc_meta: dict) -> np.ndarray` — bool array of length `len(chunks)`

Dates are ISO `YYYY-MM-DD` strings and compare lexicographically, which is why no date parsing is needed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_query_construction.py`:

```python
import numpy as np
import pytest

from rag.chunking import Chunk
from rag.query_construction import MetadataFilter, compile_mask


def _chunks():
    return [
        Chunk("a:0", "a", 0, "t", 0, 5, 0, 1),
        Chunk("a:1", "a", 1, "t", 0, 5, 1, 2),
        Chunk("b:0", "b", 0, "t", 0, 5, 0, 1),
        Chunk("c:0", "c", 0, "t", 0, 5, 0, 1),
    ]


def _doc_meta():
    return {
        "a": {"topic": "retrieval-models", "author": "Khattab", "publish_date": "2020-04-27"},
        "b": {"topic": "rag-systems", "author": "Lewis", "publish_date": "2024-01-29"},
        "c": {"topic": "rag-systems", "author": "Sarthi", "publish_date": "2023-01-01"},
    }


# --- the filter itself -------------------------------------------------------

def test_an_empty_filter_is_empty():
    assert MetadataFilter().is_empty()


def test_a_filter_with_any_constraint_is_not_empty():
    assert not MetadataFilter(topics=("rag-systems",)).is_empty()
    assert not MetadataFilter(published_before="2024-01-01").is_empty()


def test_topic_matching():
    f = MetadataFilter(topics=("rag-systems",))
    assert f.matches({"topic": "rag-systems"})
    assert not f.matches({"topic": "foundations"})


def test_several_topics_are_an_or():
    f = MetadataFilter(topics=("rag-systems", "foundations"))
    assert f.matches({"topic": "foundations"})


def test_author_matching_is_case_insensitive_and_partial():
    # A question says "Khattab" where metadata says "Khattab and Zaharia".
    f = MetadataFilter(authors=("khattab",))
    assert f.matches({"author": "Khattab and Zaharia"})
    assert not f.matches({"author": "Lewis et al."})


def test_published_before_excludes_the_boundary_date():
    f = MetadataFilter(published_before="2024-01-01")
    assert f.matches({"publish_date": "2023-12-31"})
    assert not f.matches({"publish_date": "2024-01-01"})


def test_published_after_excludes_the_boundary_date():
    f = MetadataFilter(published_after="2023-12-31")
    assert f.matches({"publish_date": "2024-01-01"})
    assert not f.matches({"publish_date": "2023-12-31"})


def test_constraints_combine_with_and():
    f = MetadataFilter(topics=("rag-systems",), published_before="2024-01-01")
    assert f.matches({"topic": "rag-systems", "publish_date": "2023-01-01"})
    assert not f.matches({"topic": "rag-systems", "publish_date": "2024-06-01"})


def test_a_document_missing_the_constrained_field_does_not_match():
    # Silently keeping documents with no date would make a date filter
    # quietly weaker than it claims to be.
    assert not MetadataFilter(published_before="2024-01-01").matches({"topic": "x"})


def test_describe_is_human_readable():
    f = MetadataFilter(topics=("rag-systems",), published_before="2024-01-01")
    described = f.describe()
    assert "rag-systems" in described
    assert "2024-01-01" in described


def test_describe_of_an_empty_filter_says_so():
    assert "no filter" in MetadataFilter().describe().lower()


# --- mask compilation --------------------------------------------------------

def test_an_empty_filter_keeps_everything():
    mask = compile_mask(MetadataFilter(), _chunks(), _doc_meta())
    assert mask.dtype == bool
    assert mask.tolist() == [True, True, True, True]


def test_a_topic_filter_masks_by_document():
    mask = compile_mask(
        MetadataFilter(topics=("rag-systems",)), _chunks(), _doc_meta()
    )
    assert mask.tolist() == [False, False, True, True]


def test_every_chunk_of_a_matching_document_is_kept():
    mask = compile_mask(
        MetadataFilter(topics=("retrieval-models",)), _chunks(), _doc_meta()
    )
    assert mask.tolist() == [True, True, False, False]


def test_a_date_filter_masks_correctly():
    mask = compile_mask(
        MetadataFilter(published_before="2024-01-01"), _chunks(), _doc_meta()
    )
    assert mask.tolist() == [True, True, False, True]


def test_a_chunk_whose_document_has_no_metadata_is_excluded():
    chunks = _chunks() + [Chunk("z:0", "z", 0, "t", 0, 5, 0, 1)]
    mask = compile_mask(MetadataFilter(topics=("rag-systems",)), chunks, _doc_meta())
    assert mask[-1] == False  # noqa: E712


def test_a_filter_matching_nothing_gives_an_all_false_mask():
    mask = compile_mask(MetadataFilter(topics=("nope",)), _chunks(), _doc_meta())
    assert not mask.any()


def test_mask_length_matches_the_chunk_count():
    assert len(compile_mask(MetadataFilter(), _chunks(), _doc_meta())) == 4
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_query_construction.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.query_construction'`.

- [ ] **Step 3: Implement**

Create `rag/query_construction.py`:

```python
"""PDF Stage 3: turn a natural-language constraint into a metadata filter.

"anything published before 2024" becomes `published_before="2024-01-01"`,
which becomes a NumPy boolean mask over chunks, applied *before* top-k so
that k results come back — rather than filtering the top k afterwards and
returning however many survived.

Dates are ISO `YYYY-MM-DD` strings throughout, so they compare correctly as
strings and nothing needs parsing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.chunking import Chunk


@dataclass(frozen=True)
class MetadataFilter:
    topics: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    published_before: str | None = None
    published_after: str | None = None

    def is_empty(self) -> bool:
        return not (
            self.topics or self.authors or self.published_before or self.published_after
        )

    def describe(self) -> str:
        """A one-line description, for the trace and the CLI."""
        parts: list[str] = []
        if self.topics:
            parts.append(f"topic in {', '.join(self.topics)}")
        if self.authors:
            parts.append(f"author matches {', '.join(self.authors)}")
        if self.published_before:
            parts.append(f"published before {self.published_before}")
        if self.published_after:
            parts.append(f"published after {self.published_after}")
        return "; ".join(parts) if parts else "no filter"

    def matches(self, record: dict) -> bool:
        """Whether one document's metadata satisfies every constraint.

        A document missing a constrained field does not match. Keeping it
        would make the filter quietly weaker than it says it is.
        """
        if self.topics and record.get("topic") not in self.topics:
            return False
        if self.authors:
            author = (record.get("author") or "").lower()
            if not any(wanted.lower() in author for wanted in self.authors):
                return False
        if self.published_before:
            date = record.get("publish_date")
            if not date or date >= self.published_before:
                return False
        if self.published_after:
            date = record.get("publish_date")
            if not date or date <= self.published_after:
                return False
        return True


def compile_mask(
    filter_: MetadataFilter, chunks: list[Chunk], doc_meta: dict
) -> np.ndarray:
    """A boolean mask over chunks, True where the chunk's document matches."""
    if filter_.is_empty():
        return np.ones(len(chunks), dtype=bool)
    allowed = {
        doc_id for doc_id, record in doc_meta.items() if filter_.matches(record)
    }
    return np.fromiter(
        (chunk.doc_id in allowed for chunk in chunks), dtype=bool, count=len(chunks)
    )
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_query_construction.py -q`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/query_construction.py tests/test_query_construction.py
git commit -m "feat: MetadataFilter and mask compilation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Apply the mask before top-k

**Files:**
- Modify: `rag/store.py`, `rag/strategies/base.py`
- Test: `tests/test_store.py`, `tests/test_strategies.py`

**Interfaces:**
- Produces:
  - `VectorStore.search(query_vectors, k, mask: np.ndarray | None = None)`
  - `StrategyContext.mask: np.ndarray | None = None`, passed through by `StrategyContext.search`

Masking before top-k is the point: filtering the top k afterwards returns however many survived, which is usually fewer than k and sometimes none.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_store.py`:

```python
def test_search_with_a_mask_excludes_masked_chunks():
    store = _store(3)
    mask = np.array([True, False, True])
    query = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # nearest is d:1
    ids = [r.chunk.chunk_id for r in store.search(query, 3, mask=mask)[0]]
    assert "d:1" not in ids


def test_a_mask_still_returns_k_results_when_enough_survive():
    # The point of masking before top-k: k results, not "however many of the
    # top k happened to survive".
    store = _store(5)
    mask = np.array([True, False, True, False, True])
    query = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    assert len(store.search(query, 3, mask=mask)[0]) == 3


def test_an_all_false_mask_returns_nothing():
    store = _store(3)
    mask = np.zeros(3, dtype=bool)
    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    assert store.search(query, 3, mask=mask)[0] == []


def test_an_all_true_mask_matches_an_unmasked_search():
    store = _store(4)
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    unmasked = [r.chunk.chunk_id for r in store.search(query, 3)[0]]
    masked = [
        r.chunk.chunk_id
        for r in store.search(query, 3, mask=np.ones(4, dtype=bool))[0]
    ]
    assert masked == unmasked


def test_ranks_restart_from_one_after_masking():
    store = _store(4)
    mask = np.array([False, True, True, True])
    query = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    assert [r.rank for r in store.search(query, 3, mask=mask)[0]] == [1, 2, 3]


def test_a_mask_of_the_wrong_length_is_rejected():
    store = _store(3)
    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="mask"):
        store.search(query, 3, mask=np.ones(2, dtype=bool))
```

Append to `tests/test_strategies.py`:

```python
def test_context_search_applies_the_mask(tiny_corpus: Config):
    import numpy as np

    ctx = build_context(tiny_corpus)
    ctx.mask = np.array([c.doc_id == "alpha" for c in ctx.store.chunks])
    results = ctx.search(["cosine"], k=3)[0]
    assert results
    assert all(r.chunk.doc_id == "alpha" for r in results)


def test_context_search_without_a_mask_is_unfiltered(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    assert ctx.mask is None
    assert ctx.search(["cosine"], k=3)[0]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_store.py tests/test_strategies.py -q`
Expected: `TypeError` for the unexpected `mask` keyword.

- [ ] **Step 3: Implement masking in `VectorStore.search`**

Replace the body of `search` in `rag/store.py`:

```python
    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
        mask: np.ndarray | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Nearest chunks for each query row, best first, ranked from one.

        `mask` is applied *before* top-k, by driving masked-out scores below
        any real cosine value. Filtering the top k afterwards would return
        however many survived — usually fewer than k, sometimes none.
        """
        query_vectors = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        if len(self.chunks) == 0:
            return [[] for _ in range(query_vectors.shape[0])]

        scores = l2_normalize(query_vectors) @ self.vectors.T

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (len(self.chunks),):
                raise ValueError(
                    f"mask has shape {mask.shape}, expected "
                    f"({len(self.chunks)},) — one entry per chunk"
                )
            if not mask.any():
                return [[] for _ in range(query_vectors.shape[0])]
            # Cosine of unit vectors is in [-1, 1]; -inf cannot be selected
            # and keeps the surviving order untouched.
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))

        indices, values = top_k(scores, k)
        return [
            [
                RetrievedChunk(chunk=self.chunks[int(i)], score=float(s), rank=rank)
                for rank, (i, s) in enumerate(zip(row_i, row_s), start=1)
            ]
            for row_i, row_s in zip(indices, values)
        ]
```

- [ ] **Step 4: Thread the mask through `StrategyContext`**

In `rag/strategies/base.py`, add the field and use it:

```python
    mask: np.ndarray | None = None
    """Chunks retrieval is allowed to return, or None for all of them.

    Set by query construction. Strategies never build it themselves — they
    just search, and the context applies whatever restriction is in force.
    """
```

and in `StrategyContext.search`:

```python
            return self.store.search(vectors, k, mask=self.mask)
```

Add `import numpy as np` to `base.py`.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add rag/store.py rag/strategies/base.py tests/test_store.py tests/test_strategies.py
git commit -m "feat: apply a metadata mask before top-k

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Query construction — question to filter

**Files:**
- Modify: `rag/prompts.py`, `rag/query_construction.py`
- Test: `tests/test_query_construction.py`

**Interfaces:**
- Produces:
  - `FILTER_SCHEMA`, `FILTER_TEMPLATE` in `rag/prompts.py`
  - `build_filter(question: str, llm, topics: tuple[str, ...], trace) -> MetadataFilter` in `rag/query_construction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_query_construction.py`:

```python
from rag.llm import LLMError
from rag.query_construction import build_filter
from rag.trace import Trace

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


def test_build_filter_extracts_a_date_constraint():
    llm = ReplyLLM('{"published_before": "2024-01-01"}')
    f = build_filter("anything published before 2024?", llm, TOPICS, Trace(question="q"))
    assert f.published_before == "2024-01-01"


def test_build_filter_extracts_a_topic():
    llm = ReplyLLM('{"topics": ["rag-systems"]}')
    f = build_filter("what do the RAG papers say?", llm, TOPICS, Trace(question="q"))
    assert f.topics == ("rag-systems",)


def test_build_filter_drops_a_topic_that_does_not_exist():
    # A hallucinated collection would mask everything out and return nothing.
    llm = ReplyLLM('{"topics": ["rag-systems", "invented-topic"]}')
    f = build_filter("q", llm, TOPICS, Trace(question="q"))
    assert f.topics == ("rag-systems",)


def test_build_filter_returns_an_empty_filter_for_an_unconstrained_question():
    llm = ReplyLLM("{}")
    assert build_filter("how does ColBERT work?", llm, TOPICS, Trace(question="q")).is_empty()


def test_build_filter_records_the_filter_on_the_trace():
    trace = Trace(question="q")
    build_filter("before 2024?", ReplyLLM('{"published_before": "2024-01-01"}'), TOPICS, trace)
    assert any(s.kind == "filter" for s in trace.translation)


def test_build_filter_degrades_when_the_llm_fails():
    class Failing:
        def generate(self, prompt):
            raise LLMError("rate limited")

        def structured(self, prompt, schema):
            raise LLMError("rate limited")

    trace = Trace(question="q")
    assert build_filter("q", Failing(), TOPICS, trace).is_empty()
    assert any("degraded" in n for n in trace.notes)


def test_build_filter_degrades_on_malformed_json():
    trace = Trace(question="q")
    assert build_filter("q", ReplyLLM("not json"), TOPICS, trace).is_empty()
    assert any("degraded" in n for n in trace.notes)


def test_build_filter_ignores_a_malformed_date():
    # "2024" is not a date the mask can compare against ISO strings.
    trace = Trace(question="q")
    f = build_filter("q", ReplyLLM('{"published_before": "2024"}'), TOPICS, trace)
    assert f.published_before is None


def test_build_filter_needs_no_llm_gracefully():
    trace = Trace(question="q")
    assert build_filter("q", None, TOPICS, trace).is_empty()
    assert any("degraded" in n for n in trace.notes)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_query_construction.py -q`
Expected: `ImportError: cannot import name 'build_filter'`.

- [ ] **Step 3: Add the schema and template to `rag/prompts.py`**

```python
FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "string"}},
        "authors": {"type": "array", "items": {"type": "string"}},
        "published_before": {"type": "string"},
        "published_after": {"type": "string"},
    },
    "required": [],
}

FILTER_TEMPLATE = """Extract any metadata constraints from the question below.

The documents are research papers with these fields:
- topic, one of: {topics}
- author, a surname or "Surname et al."
- publish_date, an ISO date

Reply with a JSON object containing only the constraints the question
actually states. Omit any field the question does not constrain. Do not
invent a topic that is not in the list above.

Dates must be full ISO dates. "before 2024" means
{{"published_before": "2024-01-01"}}.

Most questions state no constraint at all; for those, reply with {{}}.

Question: {question}

JSON:"""
```

Note the doubled braces: `FILTER_TEMPLATE` is filled with `.format()`, so a literal `{}` in the text must be written `{{}}`.

- [ ] **Step 4: Implement `build_filter`**

Append to `rag/query_construction.py`:

```python
import re

from rag.llm import LLMError
from rag.prompts import FILTER_SCHEMA, FILTER_TEMPLATE

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_filter(question: str, llm, topics: tuple[str, ...], trace) -> MetadataFilter:
    """Infer a metadata filter from the question, or an empty one.

    Every failure path returns an empty filter and records a note containing
    "degraded": a filter is an optimisation, and losing it costs precision,
    while wrongly applying one can mask the answer out entirely.
    """
    if llm is None:
        trace.note("query construction needs an LLM; degraded to no filter")
        return MetadataFilter()

    prompt = FILTER_TEMPLATE.format(question=question, topics=", ".join(topics))
    try:
        with trace.stage("construct"):
            parsed = llm.structured(prompt, FILTER_SCHEMA)
    except LLMError as exc:
        trace.note(f"query construction failed: {exc}; degraded to no filter")
        return MetadataFilter()

    # A hallucinated topic would mask out the whole corpus, so drop unknowns.
    chosen = tuple(t for t in parsed.get("topics", []) if t in topics)
    dropped = [t for t in parsed.get("topics", []) if t not in topics]
    if dropped:
        trace.note(f"query construction proposed unknown topics: {', '.join(dropped)}")

    def _date(key: str) -> str | None:
        value = parsed.get(key)
        return value if isinstance(value, str) and ISO_DATE.match(value) else None

    filter_ = MetadataFilter(
        topics=chosen,
        authors=tuple(a for a in parsed.get("authors", []) if isinstance(a, str)),
        published_before=_date("published_before"),
        published_after=_date("published_after"),
    )
    trace.add_translation("filter", filter_.describe())
    return filter_
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest tests/test_query_construction.py -q`

- [ ] **Step 6: Commit**

```bash
git add rag/prompts.py rag/query_construction.py tests/test_query_construction.py
git commit -m "feat: infer a metadata filter from the question

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Logical and semantic routing

**Files:**
- Create: `rag/routing.py`
- Modify: `rag/prompts.py`
- Test: `tests/test_routing.py`

**Interfaces:**
- Produces:
  - `ROUTE_SCHEMA`, `ROUTE_TEMPLATE` in `rag/prompts.py`
  - `PROMPT_VARIANTS: dict[str, str]` in `rag/prompts.py` — three answer-prompt variants keyed by name, each with `{context}` and `{question}`
  - `logical_route(question: str, llm, topics: tuple[str, ...], trace) -> tuple[str, ...]` in `rag/routing.py`
  - `SemanticRouter(embedder, descriptions: dict[str, str])` with `.route(question: str, trace) -> str` in `rag/routing.py`

The two routers answer different questions and `RAG.pdf` treats them separately: logical routing picks *where to search* by reasoning over the available collections; semantic routing picks *which prompt to answer with* by embedding similarity, no LLM call involved.

- [ ] **Step 1: Write the failing test**

Create `tests/test_routing.py`:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_routing.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.routing'`.

- [ ] **Step 3: Add the schema, template and prompt variants to `rag/prompts.py`**

```python
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"topics": {"type": "array", "items": {"type": "string"}}},
    "required": ["topics"],
}

ROUTE_TEMPLATE = """Decide which collections of papers to search for this question.

Available collections:
{descriptions}

Choose every collection that might hold the answer, and no more. Choosing too
few loses the answer; choosing all of them is the same as not routing.

Reply with a JSON object: {{"topics": ["name", ...]}}.

Question: {question}

JSON:"""

TOPIC_DESCRIPTIONS = {
    "foundations": "transformer and language-model architecture papers",
    "retrieval-models": "dense, sparse and late-interaction retrieval models and embeddings",
    "rag-systems": "retrieval-augmented generation systems and their architectures",
    "prompting-reasoning": "prompting, chain-of-thought reasoning and query transformation",
    "evaluation-benchmarks": "benchmarks, datasets, metrics and evaluation studies",
}

PROMPT_VARIANTS = {
    "definition": ANSWER_TEMPLATE,
    "mechanism": """You are explaining how something works, using retrieved excerpts.

Rules:
- Use only the context below. Do not use prior knowledge.
- If the context does not answer the question, say so plainly and stop.
- If no documents were retrieved at all, reply that no documents were retrieved
  and that you therefore cannot answer. Do not repeat this instruction back.
- Describe the mechanism in order, step by step.
- Cite the excerpts you used with their bracketed numbers, like [1] or [2].

Context:
{context}

Question: {question}

Answer:""",
    "comparison": """You are comparing approaches, using retrieved excerpts.

Rules:
- Use only the context below. Do not use prior knowledge.
- If the context does not answer the question, say so plainly and stop.
- If no documents were retrieved at all, reply that no documents were retrieved
  and that you therefore cannot answer. Do not repeat this instruction back.
- State what each approach does, then what distinguishes them.
- Cite the excerpts you used with their bracketed numbers, like [1] or [2].

Context:
{context}

Question: {question}

Answer:""",
}

PROMPT_DESCRIPTIONS = {
    "definition": "explaining what a term, concept or system is",
    "mechanism": "describing how a method works, step by step",
    "comparison": "contrasting two or more approaches against each other",
}
```

`PROMPT_VARIANTS["definition"]` reuses `ANSWER_TEMPLATE` so the default path is unchanged and there is one fewer copy of those rules to drift.

- [ ] **Step 4: Implement `rag/routing.py`**

```python
"""PDF Stage 2: decide where to search, and how to answer.

Two different mechanisms, deliberately kept apart:

- **Logical routing** asks the model which collections could hold the answer.
  It needs reasoning about what each collection contains, which is what an
  LLM is for, and it uses structured output so the answer is a field rather
  than prose to be parsed.
- **Semantic routing** picks which answer prompt to use by embedding the
  question and comparing it to embedded descriptions of each prompt. No LLM
  call — if it needed one it would be logical routing with extra steps.

Both degrade to their unrestricted default and record it: routing narrows
things, so a routing failure should widen the search back rather than
narrowing it wrongly.
"""

from __future__ import annotations

import numpy as np

from rag.llm import LLMError
from rag.prompts import ROUTE_SCHEMA, ROUTE_TEMPLATE, TOPIC_DESCRIPTIONS
from rag.similarity import cosine_similarity


def logical_route(
    question: str, llm, topics: tuple[str, ...], trace
) -> tuple[str, ...]:
    """Which collections to search. An empty tuple means all of them."""
    if llm is None:
        trace.note("logical routing needs an LLM; degraded to searching everything")
        return ()

    described = "\n".join(
        f"- {name}: {TOPIC_DESCRIPTIONS.get(name, name)}" for name in topics
    )
    prompt = ROUTE_TEMPLATE.format(question=question, descriptions=described)
    try:
        with trace.stage("route"):
            parsed = llm.structured(prompt, ROUTE_SCHEMA)
    except LLMError as exc:
        trace.note(f"logical routing failed: {exc}; degraded to searching everything")
        return ()

    chosen = tuple(t for t in parsed.get("topics", []) if t in topics)
    if not chosen:
        trace.note("logical routing chose nothing valid; searching everything")
        return ()
    trace.add_translation("route", f"search {', '.join(chosen)}")
    return chosen


class SemanticRouter:
    """Picks an answer prompt by cosine similarity, with no LLM call.

    Descriptions are embedded once at construction; each question costs one
    embedding and one matmul against a handful of vectors.
    """

    def __init__(self, embedder, descriptions: dict[str, str]) -> None:
        if not descriptions:
            raise ValueError("a semantic router needs at least one description")
        self.names = tuple(descriptions)
        self._embedder = embedder
        self._vectors = embedder.encode([descriptions[n] for n in self.names])

    def route(self, question: str, trace) -> str:
        """The name of the best-matching prompt."""
        query = self._embedder.encode([question])
        scores = cosine_similarity(query, self._vectors)[0]
        chosen = self.names[int(np.argmax(scores))]
        trace.add_translation(
            "route", f"prompt {chosen} (cosine {float(scores.max()):.3f})"
        )
        return chosen
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest tests/test_routing.py -q`

- [ ] **Step 6: Commit**

```bash
git add rag/routing.py rag/prompts.py tests/test_routing.py
git commit -m "feat: logical and semantic routing

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Wire routing and filtering into the pipeline

**Files:**
- Modify: `rag/pipeline.py`, `rag/generation.py`, `evaluation/benchmark.py`
- Test: `tests/test_pipeline.py`, `tests/test_benchmark.py`

**Interfaces:**
- Produces:
  - `ask(..., route: bool = False, construct: bool = False, semantic_prompt: bool = False)`
  - the benchmark's degradation check widened to any note containing `degraded`

The data flow the spec specifies is route, then construct, then translate, then search.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_ask_without_routing_or_construction_is_unchanged(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert trace.retrieved
    assert not any(s.kind in ("route", "filter") for s in trace.translation)


def test_ask_with_construction_records_a_filter(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "2024-01-01"}')
    trace = ask("before 2024?", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert any(s.kind == "filter" for s in trace.translation)


def test_construction_restricts_what_is_retrieved(tiny_corpus: Config):
    # tiny_corpus: alpha is 2023-05-01, beta is 2024-02-11.
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "2024-01-01"}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert trace.retrieved
    assert all(r.chunk.doc_id == "alpha" for r in trace.retrieved)


def test_a_filter_matching_nothing_leaves_a_note_and_no_results(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"published_before": "1900-01-01"}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, construct=True)
    assert trace.retrieved == []
    assert any("filter" in n.lower() for n in trace.notes)


def test_ask_with_routing_records_a_route(tiny_corpus: Config):
    # The tiny fixture corpus has no topics, so give it two: routing is
    # skipped entirely when there is nothing to choose between.
    store = build_index(tiny_corpus, FakeEmbedder())
    store.doc_meta["alpha"]["topic"] = "similarity"
    store.doc_meta["beta"]["topic"] = "fusion"
    llm = FakeLLM('{"topics": ["similarity"]}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, route=True)
    assert any(s.kind == "route" for s in trace.translation)


def test_routing_is_skipped_when_the_corpus_has_no_topics(tiny_corpus: Config):
    # Nothing to route between is not a failure, and must not cost an LLM call.
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM('{"topics": ["anything"]}')
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, route=True)
    assert not any(s.kind == "route" for s in trace.translation)
    assert llm.prompts == []


def test_semantic_prompt_selection_records_its_choice(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask(
        "q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, semantic_prompt=True
    )
    assert any(s.kind == "route" and "prompt" in s.text for s in trace.translation)
```

Append to `tests/test_benchmark.py`:

```python
def test_any_degraded_note_is_fatal():
    # Phase 4 adds routing and filtering, each with its own degradation
    # wording. Scanning only for "degraded to direct retrieval" would let a
    # silent routing failure be measured as "routing does not help".
    from evaluation.benchmark import check_not_degraded
    from rag.trace import Trace

    for note in [
        "hyde needs an LLM; degraded to direct retrieval",
        "logical routing failed: rate limited; degraded to searching everything",
        "query construction failed: bad JSON; degraded to no filter",
    ]:
        trace = Trace(question="q")
        trace.note(note)
        with pytest.raises(RuntimeError, match="degraded"):
            check_not_degraded(trace)


def test_a_non_degradation_note_is_not_fatal():
    from evaluation.benchmark import check_not_degraded
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.note("generation served from cache")
    check_not_degraded(trace)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_pipeline.py tests/test_benchmark.py -q`
Expected: `TypeError` for the unexpected `construct` keyword, and the widened-check test failing.

- [ ] **Step 3: Wire the stages into `ask`**

In `rag/pipeline.py`, add the parameters and insert the two stages before the strategy runs:

```python
def ask(
    question: str,
    store: VectorStore,
    embedder,
    llm,
    config: Config,
    k: int | None = None,
    strategy: str = "direct",
    strategy_options: dict | None = None,
    generate: bool = True,
    route: bool = False,
    construct: bool = False,
    semantic_prompt: bool = False,
) -> Trace:
```

After the trace is created and before the strategy runs:

```python
    topics = tuple(
        sorted({r.get("topic") for r in store.doc_meta.values() if r.get("topic")})
    )

    active = MetadataFilter()
    if route and topics:
        chosen = logical_route(question, llm, topics, trace)
        if chosen:
            active = MetadataFilter(topics=chosen)
    if construct:
        inferred = build_filter(question, llm, topics, trace)
        if not inferred.is_empty():
            # Routing narrows by topic; construction adds its own constraints.
            # Keep the routed topics unless construction named its own.
            active = MetadataFilter(
                topics=inferred.topics or active.topics,
                authors=inferred.authors,
                published_before=inferred.published_before,
                published_after=inferred.published_after,
            )

    mask = None
    if not active.is_empty():
        mask = compile_mask(active, store.chunks, store.doc_meta)
        if not mask.any():
            trace.note(
                f"filter matched no documents ({active.describe()}); "
                "returning no results"
            )
```

Pass `mask=mask` when constructing the `StrategyContext`.

For the answer prompt, when `semantic_prompt` is on:

```python
    prompt_name = None
    if semantic_prompt:
        prompt_name = SemanticRouter(embedder, PROMPT_DESCRIPTIONS).route(
            question, trace
        )
```

and pass `prompt_name` through to `generate_answer`.

Add the imports: `from rag.query_construction import MetadataFilter, build_filter, compile_mask`, `from rag.routing import SemanticRouter, logical_route`, `from rag.prompts import PROMPT_DESCRIPTIONS`.

- [ ] **Step 4: Let `generate_answer` use a prompt variant**

In `rag/generation.py`, add `prompt_name: str | None = None` and pass it to `build_answer_prompt`. In `rag/prompts.py`, extend `build_answer_prompt` with the same parameter, selecting from `PROMPT_VARIANTS` and falling back to `ANSWER_TEMPLATE` when the name is `None` or unknown.

- [ ] **Step 5: Widen the benchmark's degradation check**

In `evaluation/benchmark.py`, change the sentinel from the full phrase to the single word:

```python
DEGRADED = "degraded"
```

and update the docstring to say that every degradation path in `rag/` includes that word deliberately, so one check covers strategies, routing and query construction alike.

- [ ] **Step 6: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 7: Commit**

```bash
git add rag/ evaluation/benchmark.py tests/
git commit -m "feat: wire routing and query construction into the pipeline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: CLI, step-back via structured output, README and benchmark

**Files:**
- Modify: `rag/__main__.py`, `rag/strategies/step_back.py`, `rag/prompts.py`, `README.md`
- Test: `tests/test_cli.py`, `tests/test_strategies.py`

**Interfaces:**
- Produces: `--route`, `--construct`, `--semantic-prompt` flags on `rag ask`; step-back using structured output

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_route_flag_reaches_the_pipeline(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    seen = {}

    def spy(question, store, embedder, llm, config, **kwargs):
        seen.update(kwargs)
        from rag.trace import Trace

        return Trace(question=question)

    monkeypatch.setattr("rag.__main__.ask", spy)
    main(["ask", "q", "--route", "--construct"], **_factories())
    assert seen["route"] is True
    assert seen["construct"] is True


def test_routing_flags_are_rejected_without_an_llm(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "q", "--route", "--no-llm"], **_factories()) == 1
    assert "--no-llm" in capsys.readouterr().err


def test_semantic_prompt_is_allowed_without_an_llm(tiny_corpus, monkeypatch):
    # Semantic routing is embedding-only; it needs no API key.
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "q", "--semantic-prompt", "--no-llm"], **_factories()) == 0
```

Append to `tests/test_strategies.py`:

```python
def test_step_back_uses_structured_output(tiny_corpus: Config):
    # Phase 2's step-back parsed prose and took four attempts to get right.
    # The code recorded that structured output was the principled fix.
    class StructuredLLM:
        def generate(self, prompt):
            return '{"question": "What is vector similarity?"}'

        def structured(self, prompt, schema):
            from rag.llm import GeminiLLM

            return GeminiLLM.structured(self, prompt, schema)

    ctx = build_context(tiny_corpus, llm=StructuredLLM())
    get_strategy("step-back").run("how does cosine handle magnitude?", ctx)
    assert ctx.trace.queries[1] == "What is vector similarity?"


def test_step_back_degrades_when_structured_output_fails(tiny_corpus: Config):
    class BadLLM:
        def generate(self, prompt):
            return "not json"

        def structured(self, prompt, schema):
            from rag.llm import GeminiLLM

            return GeminiLLM.structured(self, prompt, schema)

    ctx = build_context(tiny_corpus, llm=BadLLM())
    result = get_strategy("step-back").run("q", ctx)
    assert result.retrieved
    assert any("degraded" in n for n in ctx.trace.notes)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_cli.py tests/test_strategies.py -q`
Expected: `unrecognized arguments: --route`, and step-back still parsing prose.

- [ ] **Step 3: Add the CLI flags**

On the `ask` subparser in `rag/__main__.py`:

```python
    ask_parser.add_argument(
        "--route", action="store_true",
        help="let the model choose which topical collections to search",
    )
    ask_parser.add_argument(
        "--construct", action="store_true",
        help="infer a metadata filter from the question",
    )
    ask_parser.add_argument(
        "--semantic-prompt", action="store_true",
        help="pick the answer prompt by embedding similarity (no LLM call)",
    )
```

In `_run`, reject the LLM-requiring flags alongside `--no-llm`, in the same style as the existing `--strategy` conflict, and pass all three through to `ask`. `--semantic-prompt` must remain allowed with `--no-llm`.

- [ ] **Step 4: Render routing and filtering in `format_trace`**

The existing translation block already prints every `TranslationStep`, so `route` and `filter` kinds appear with no change. Confirm that by eye when you run the demo in Step 7, and adjust only if the labels read badly.

- [ ] **Step 5: Convert step-back to structured output**

Add to `rag/prompts.py`:

```python
STEP_BACK_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

STEP_BACK_JSON_TEMPLATE = """Given a specific question, write one more general
question about the underlying concept or principle it depends on.

The general question should be broad enough that a document explaining the
background would answer it, while staying on the same subject. Do not answer
either question.

Reply with a JSON object: {{"question": "..."}}.

Specific question: {question}

JSON:"""
```

In `rag/strategies/step_back.py`, replace the `generate` call and `_pick_question` with a `structured` call against `STEP_BACK_SCHEMA`, taking `parsed["question"].strip()`. Degrade as before when it is empty or raises. **Delete `_pick_question` and `CHATTER`** — they exist only to parse prose, and leaving dead heuristics beside the structured path invites someone to reinstate them. Delete the tests that covered them, and say in your report which ones you removed.

- [ ] **Step 6: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 7: Run the spec's demo against the real corpus**

Do not set `PYTHONIOENCODING`.

```
python -m rag ask "What did the RAPTOR paper say about clustering, from anything published before 2024?" --construct --trace
python -m rag ask "How does ColBERT score a document?" --route --trace
python -m rag ask "How does ColBERT score a document?" --semantic-prompt --trace
python -m rag ask "q" --route --no-llm
```

Expected: the first shows an inferred filter in the trace — note RAPTOR is dated 2024-01-31, so a `before 2024` filter genuinely excludes it, and the honest outcome is that the answer says so. Report what actually happens rather than presenting it as a success. The second shows a routed collection; the third shows a chosen prompt; the fourth exits 1.

- [ ] **Step 8: Re-run the benchmark and update the README**

```
python -m evaluation.benchmark
```

Confirm the six existing strategy rows are unchanged — Phase 4 adds stages that are off by default, so if any score moved, something leaked into the default path and you must say so.

Then add a Routing and query construction section to the README covering the three features, the five topics, the demo above including its honest outcome, and one line noting that routing and construction are not in the benchmark table because they change *what is searched* rather than *how*, so comparing them against the same gold set would measure a different thing. Check Phase 4 in the roadmap.

- [ ] **Step 9: Commit**

```bash
git add rag/ tests/ README.md
git commit -m "feat: routing and construction flags; step-back via structured output

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 4 Definition of Done

- [ ] `python -m pytest` passes with no network access; `-m slow` passes
- [ ] The five topics partition all 38 documents, none holding more than half
- [ ] `--construct` infers a filter from "anything published before 2024" and the trace shows it
- [ ] `--route` picks collections and the trace shows which
- [ ] `--semantic-prompt` picks an answer prompt with no LLM call, and works with `--no-llm`
- [ ] Masking happens before top-k: a filtered search still returns k results when k documents survive
- [ ] Every routing and construction failure degrades and records a note containing `degraded`
- [ ] The benchmark's check catches all three degradation wordings
- [ ] Step-back uses structured output; `_pick_question` and `CHATTER` are gone
- [ ] The six benchmark strategy rows are unchanged
- [ ] Working tree clean

## What later phases need from this

- `VectorStore.doc_meta` is what Phase 5's multi-representation indexing will hang document summaries off.
- `GeminiLLM.structured` is the mechanism for any later feature that needs a field rather than prose.
- `StrategyContext.mask` lets Phase 6's ColBERT reranker operate on a filtered candidate set.
- The widened `degraded` check means a Phase 5 or 6 feature that silently falls back cannot be measured as "it does not help".
