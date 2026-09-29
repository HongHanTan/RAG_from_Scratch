# Phase 6b: Retrieval Inspector Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only local dashboard that shows what retrieval actually did — chunks and scores, query translations, stage timings, a 2D projection, and strategies side by side.

**Architecture:** A thin FastAPI layer over the existing package plus one static page of vanilla JavaScript. The server holds one `VectorStore` and one `Embedder` for its lifetime and never mutates anything. `Trace.to_dict()` already serialises everything four of the five panels need, so the API is mostly a wrapper rather than new logic.

**Tech Stack:** Python 3.14, FastAPI 0.139, uvicorn 0.42, NumPy, vanilla JavaScript. No React, no build step, no Streamlit, no CDN.

## Global Constraints

- **No RAG framework.** `langchain`, `llama_index`, `sentence_transformers`, `sklearn`, `bs4`, `requests`, `dotenv` are installed and MUST NOT be imported. `tests/test_no_frameworks.py` scans `rag/`, `scripts/`, `tests/`, `evaluation/` and `pyproject.toml`. FastAPI is a web framework, not a RAG framework, and the spec names it explicitly — it is not on the forbidden list.
- **No front-end framework and no build step.** One `.html`, one `.css`, one `.js`, all served from disk. No React, no Streamlit, no bundler, and **no CDN `<script src="http...">`** — a framework fetched at runtime is still a framework wrapper, and it would cut against the premise of the project. Task 3 has a test enforcing this.
- **Allowed third-party:** `numpy`, `torch`, `transformers`, `google.genai`, `pytest`, `fastapi`, `uvicorn`, `httpx` (test client only).
- **Read-only.** Every endpoint is a `GET`. The dashboard never writes an index, never writes the corpus, never mutates a store.
- **Test-driven.** Write the test, run it, watch it fail for the expected reason, then implement.
- **Every unit test runs offline.** `fastapi.testclient.TestClient` works without a network; dependencies are injected so no test loads a real model or calls the API, unless marked `@pytest.mark.slow` / `@pytest.mark.live`.
- **Platform is Windows.** `pathlib`, explicit `encoding="utf-8"`.
- **Never set `PYTHONIOENCODING`** when running anything.
- **Do not rebuild or overwrite anything in `data/`.** The three index files are the baseline every published number rests on.
- **Commit after every task.** Messages end with a blank line then exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## What the earlier phases provide

| Thing | Signature |
|---|---|
| `rag.pipeline.ask(...)` | `(question, store, embedder, llm, config, k=None, strategy="direct", strategy_options=None, generate=True, route=False, construct=False, semantic_prompt=False, rerank=False) -> Trace` |
| `rag.pipeline.load_index(config, mode="flat")` | `-> VectorStore`; `INDEX_MODES = ("flat", "multirep", "raptor")` |
| `rag.trace.Trace.to_dict()` | `{question, queries, strategy, translation[{kind,text}], retrieved[{chunk,score,rank,score_kind}], prompt, answer, timings[{name,ms,depth}], notes, total_ms}` |
| `rag.strategies.get_strategy(name)` | names: `direct`, `multi-query`, `rag-fusion`, `decomposition`, `step-back`, `hyde` |
| `rag.store.VectorStore` | `.vectors` (n, 384) L2-normalised, `.chunks`, `.meta`, `.doc_meta`, `.docstore`, `.search(qv, k, mask=None)`, `.dim`, `__len__` |
| `rag.embedding.Embedder` | `.encode(texts) -> (n, dim)`, `.encode_tokens(texts) -> list[(n_tok, dim)]`, `.dim`, `.tokenizer` |
| `rag.__main__.load_config(**overrides)` | `-> Config` reading `.env` |
| `tests.conftest` | `FakeEmbedder` (dim 8), `FakeLLM`, `tiny_corpus` |

Current suite: 668 passed, 15 deselected.

## Design decisions, stated up front

**It lives at `rag/dashboard/`, not a top-level `dashboard/`.** `tests/test_no_frameworks.py` scans a fixed list of directories, `["rag", "scripts", "tests", "evaluation"]`, and `pyproject.toml` packages `["rag*"]`. A new top-level package would be covered by neither — the project's central test would silently stop covering its newest code, which is the same "passes while checking nothing" failure the suite already guards against elsewhere. Putting it under `rag/` gets both for free. `rag/__init__.py` must not import it, so `import rag` never pulls FastAPI.

**Four panels come from one request, not four.** The chunk table, translation trace, stage timings and projection are all views of a *single* run. Giving each its own endpoint would re-run retrieval three extra times and show four panels describing four different runs. They share `GET /api/ask`. Strategy comparison is genuinely separate work and much slower, so it gets `GET /api/compare` and loads independently — which is what the spec's "each panel renders its own loading state" is protecting against.

**The projection is PCA via `np.linalg.svd`, and is labelled as such.** 384 dimensions squeezed into 2 discards almost everything; the panel says "a projection of 384 dimensions, not the space itself" so nobody reads cluster structure into it that is not there.

**It binds `127.0.0.1` by default.** It is a local inspector with no authentication that can spend Gemini quota, so it must not be reachable from the network unless the operator asks. `--host` exists for the operator who means it.

## File Structure

```
rag/dashboard/
  __init__.py        empty; keeps `import rag` free of FastAPI
  projection.py      project_2d()
  app.py             create_app(), the GET endpoints
  static/
    index.html       the five panels
    style.css
    app.js           fetch + render, one loading state per panel
rag/__main__.py      MODIFY: `dashboard` subcommand
pyproject.toml       MODIFY: [dashboard] optional dependency group
tests/               test_projection.py, test_dashboard_api.py, test_dashboard_static.py
README.md            MODIFY
```

---

### Task 1: The 2D projection

**Files:**
- Create: `rag/dashboard/__init__.py`, `rag/dashboard/projection.py`
- Test: `tests/test_projection.py`

**Interfaces:**
- Produces: `project_2d(vectors: np.ndarray, query_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `((n, 2) chunk coords, (2,) query coords)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_projection.py`:

```python
import numpy as np
import pytest

from rag.dashboard.projection import project_2d


def _unit(rows):
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _blobs(n=12, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=2.0, scale=0.2, size=(n, dim))
    b = rng.normal(loc=-2.0, scale=0.2, size=(n, dim))
    return _unit(np.vstack([a, b])), n


def test_returns_two_dimensional_coordinates():
    vectors, _ = _blobs()
    coords, query = project_2d(vectors, _unit([[1.0] * 8])[0])
    assert coords.shape == (len(vectors), 2)
    assert query.shape == (2,)


def test_the_query_is_projected_into_the_same_space():
    # The query marker is only meaningful if it went through the same
    # transform as the chunks; projecting it separately would place it
    # somewhere arbitrary.
    vectors, _ = _blobs()
    q = vectors[0].copy()
    coords, query = project_2d(vectors, q)
    assert np.allclose(coords[0], query, atol=1e-4)


def test_separable_clusters_stay_separated_in_2d():
    # A projection that collapses obviously distinct groups is not showing
    # anything; this is the property the panel depends on.
    vectors, n = _blobs()
    coords, _ = project_2d(vectors, vectors.mean(axis=0))
    first, second = coords[:n].mean(axis=0), coords[n:].mean(axis=0)
    spread = np.linalg.norm(coords - coords.mean(axis=0), axis=1).mean()
    assert np.linalg.norm(first - second) > spread


def test_it_is_deterministic():
    vectors, _ = _blobs()
    q = _unit([[1.0] * 8])[0]
    a, aq = project_2d(vectors, q)
    b, bq = project_2d(vectors, q)
    assert np.array_equal(a, b)
    assert np.array_equal(aq, bq)


def test_output_is_finite():
    vectors, _ = _blobs()
    coords, query = project_2d(vectors, _unit([[1.0] * 8])[0])
    assert np.isfinite(coords).all()
    assert np.isfinite(query).all()


def test_a_single_vector_still_projects():
    # Top-k of 1 is a legitimate request and must not raise.
    coords, query = project_2d(_unit([[1.0, 0.0, 0.0]]), _unit([[0.0, 1.0, 0.0]])[0])
    assert coords.shape == (1, 2)
    assert np.isfinite(coords).all()


def test_two_vectors_still_project():
    coords, _ = project_2d(_unit([[1.0, 0.0], [0.0, 1.0]]), _unit([[1.0, 1.0]])[0])
    assert coords.shape == (2, 2)
    assert np.isfinite(coords).all()


def test_no_vectors_returns_empty_coordinates():
    coords, query = project_2d(np.zeros((0, 4), dtype=np.float32),
                               _unit([[1.0, 0.0, 0.0, 0.0]])[0])
    assert coords.shape == (0, 2)
    assert query.shape == (2,)


def test_identical_vectors_do_not_produce_nan():
    # Zero variance means a degenerate SVD; the panel must still render.
    coords, query = project_2d(np.ones((5, 4), dtype=np.float32),
                               np.ones(4, dtype=np.float32))
    assert np.isfinite(coords).all()
    assert np.isfinite(query).all()


def test_a_mismatched_query_dimension_is_an_error():
    with pytest.raises(ValueError, match="dimension"):
        project_2d(_unit([[1.0, 0.0]]), np.ones(3, dtype=np.float32))
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_projection.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.dashboard'`.

- [ ] **Step 3: Implement**

Create `rag/dashboard/__init__.py` as an empty file. **It must stay empty**: importing `app.py` from here would make `import rag` pull in FastAPI, turning an optional dashboard dependency into a hard one for the whole package.

Create `rag/dashboard/projection.py`:

```python
"""Chunk embeddings reduced to two dimensions, for the dashboard scatter.

This is PCA done with `np.linalg.svd`: centre the points, take the two
directions of greatest variance, project onto them. Those two directions
are chosen to preserve as much spread as any plane can, which is a much
weaker promise than it looks -- 384 dimensions do not fit in 2, and most of
the geometry is discarded. The panel is labelled a projection of 384
dimensions rather than the space itself for exactly that reason.

The query is projected with the same transform as the chunks, not its own,
because a marker placed by a different transform would sit somewhere
arbitrary relative to the points it is meant to be compared against.
"""

from __future__ import annotations

import numpy as np


def project_2d(
    vectors: np.ndarray, query_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project chunk vectors and the query into a shared 2D plane."""
    vectors = np.asarray(vectors, dtype=np.float32)
    query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)

    if vectors.size and vectors.shape[1] != query_vector.shape[0]:
        raise ValueError(
            f"dimension mismatch: chunks {vectors.shape[1]} vs "
            f"query {query_vector.shape[0]}"
        )
    if vectors.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)

    # The query is stacked in so it is centred and rotated with everything
    # else, then split back out at the end.
    stacked = np.vstack([vectors, query_vector[None, :]])
    centred = stacked - stacked.mean(axis=0, keepdims=True)

    # full_matrices=False keeps this (n, min(n, dim)) rather than (dim, dim).
    _, _, vt = np.linalg.svd(centred, full_matrices=False)

    # Fewer than two components exist when there are fewer than three points
    # or the points are identical; pad so the caller always gets 2 columns
    # rather than a ragged array it has to special-case.
    axes = vt[:2]
    if axes.shape[0] < 2:
        axes = np.vstack([axes, np.zeros((2 - axes.shape[0], axes.shape[1]))])

    coords = (centred @ axes.T).astype(np.float32)
    return coords[:-1], coords[-1]
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_projection.py -q`
Expected: 10 passed.

Then the full suite: `python -m pytest -q` (668 + 10 = 678 passed, 15 deselected).

- [ ] **Step 5: Confirm the no-frameworks guard covers the new directory**

This is the reason the package sits under `rag/`. Prove it rather than assume:

```
python -c "
import sys; sys.path.insert(0, 'tests')
from test_no_frameworks import _python_files
paths = [str(p) for p in _python_files() if 'dashboard' in str(p)]
print('dashboard files scanned by the guard:', len(paths))
for p in paths: print('  ', p)
"
```

Expect at least the two new files. If it prints 0, stop — the guard is not covering the new code and the rest of this plan would be written outside it.

- [ ] **Step 6: Commit**

```bash
git add rag/dashboard tests/test_projection.py
git commit -m "feat: 2D SVD projection for the dashboard scatter

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The read-only API

**Files:**
- Create: `rag/dashboard/app.py`
- Test: `tests/test_dashboard_api.py`

**Interfaces:**
- Consumes: `project_2d`, `ask`, `load_index`, `Trace.to_dict`
- Produces: `create_app(store, embedder, llm, config) -> fastapi.FastAPI` with
  - `GET /api/meta` -> `{index_mode, chunks, dim, embedding_model, llm_model, strategies}`
  - `GET /api/ask?q=&strategy=&k=&rerank=` -> `trace.to_dict()` plus `{"projection": {"chunks": [[x,y],...], "query": [x,y]}}`
  - `GET /api/compare?q=&strategies=a,b` -> `{"results": [{strategy, recall_preview, trace}, ...]}`
  - `GET /` -> the static page

Dependencies are passed into `create_app` rather than built inside it, so every test runs offline against fakes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

from rag.dashboard.app import create_app
from rag.pipeline import build_index
from tests.conftest import FakeEmbedder, FakeLLM


@pytest.fixture
def client(tiny_corpus):
    store = build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    app = create_app(store, FakeEmbedder(), FakeLLM("an answer"), tiny_corpus)
    return TestClient(app)


# --- meta ---------------------------------------------------------------------

def test_meta_reports_the_index(client):
    body = client.get("/api/meta").json()
    assert body["chunks"] > 0
    assert body["dim"] == 8
    assert "direct" in body["strategies"]


# --- ask ----------------------------------------------------------------------

def test_ask_returns_a_trace(client):
    body = client.get("/api/ask", params={"q": "cosine"}).json()
    assert body["question"] == "cosine"
    assert body["retrieved"]


def test_ask_includes_the_score_kind(client):
    # The chunk table shows raw scores; cosine and maxsim are different
    # quantities and the panel must not present them as one.
    body = client.get("/api/ask", params={"q": "cosine"}).json()
    assert all("score_kind" in r for r in body["retrieved"])


def test_ask_includes_stage_timings(client):
    body = client.get("/api/ask", params={"q": "cosine"}).json()
    assert body["timings"]
    assert all({"name", "ms"} <= set(t) for t in body["timings"])


def test_ask_includes_a_projection_for_every_chunk(client):
    body = client.get("/api/ask", params={"q": "cosine"}).json()
    assert len(body["projection"]["chunks"]) == len(body["retrieved"])
    assert len(body["projection"]["query"]) == 2


def test_ask_honours_k(client):
    body = client.get("/api/ask", params={"q": "cosine", "k": 2}).json()
    assert len(body["retrieved"]) <= 2


def test_ask_honours_the_strategy(client):
    body = client.get("/api/ask", params={"q": "cosine", "strategy": "multi-query"}).json()
    assert body["strategy"] == "multi-query"


def test_ask_exposes_the_translation_trace(client):
    body = client.get("/api/ask", params={"q": "cosine", "strategy": "multi-query"}).json()
    assert "translation" in body


def test_an_unknown_strategy_is_a_400_not_a_500(client):
    response = client.get("/api/ask", params={"q": "cosine", "strategy": "nope"})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_an_empty_question_is_a_400(client):
    assert client.get("/api/ask", params={"q": "   "}).status_code == 400


# --- compare ------------------------------------------------------------------

def test_compare_returns_one_result_per_strategy(client):
    body = client.get(
        "/api/compare", params={"q": "cosine", "strategies": "direct,multi-query"}
    ).json()
    assert [r["strategy"] for r in body["results"]] == ["direct", "multi-query"]


def test_compare_rejects_an_unknown_strategy(client):
    response = client.get(
        "/api/compare", params={"q": "cosine", "strategies": "direct,nope"}
    )
    assert response.status_code == 400


def test_compare_defaults_to_every_strategy(client):
    body = client.get("/api/compare", params={"q": "cosine"}).json()
    assert len(body["results"]) >= 6


# --- read-only ----------------------------------------------------------------

@pytest.mark.parametrize("method", ["post", "put", "delete", "patch"])
def test_the_api_is_read_only(client, method):
    # The dashboard inspects; it must not offer a way to mutate an index.
    response = getattr(client, method)("/api/ask", params={"q": "x"})
    assert response.status_code == 405


def test_the_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_dashboard_api.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.dashboard.app'`.

- [ ] **Step 3: Implement `rag/dashboard/app.py`**

```python
"""A read-only HTTP layer over the package, for the retrieval inspector.

Every endpoint is a GET and nothing here writes an index, a corpus or a
cache entry deliberately: the dashboard exists to show what retrieval did,
and an inspector that can change the thing it inspects is a worse tool.

Dependencies are passed into `create_app` rather than constructed inside
it, so the tests drive the whole API with fakes and never load a model or
call an API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag.dashboard.projection import project_2d
from rag.pipeline import ask
from rag.strategies import STRATEGY_NAMES

STATIC = Path(__file__).parent / "static"


def create_app(store, embedder, llm, config) -> FastAPI:
    app = FastAPI(title="RAG retrieval inspector", docs_url=None, redoc_url=None)

    def _check(question: str, names: list[str]) -> None:
        if not question.strip():
            raise HTTPException(status_code=400, detail="question is empty")
        for name in names:
            if name not in STRATEGY_NAMES:
                raise HTTPException(
                    status_code=400, detail=f"unknown strategy: {name}"
                )

    def _run(question: str, strategy: str, k: int | None, rerank: bool) -> dict:
        trace = ask(
            question, store, embedder, llm, config,
            k=k, strategy=strategy, generate=False, rerank=rerank,
        )
        payload = trace.to_dict()

        # Project the chunks that were actually returned, against the same
        # query vector the search used, so the scatter matches the table.
        rows = [
            store.vectors[i]
            for item in trace.retrieved
            for i, chunk in enumerate(store.chunks)
            if chunk.chunk_id == item.chunk.chunk_id
        ]
        matrix = np.vstack(rows) if rows else np.zeros((0, store.dim), np.float32)
        coords, query_xy = project_2d(matrix, embedder.encode([question])[0])
        payload["projection"] = {
            "chunks": coords.tolist(),
            "query": query_xy.tolist(),
            "note": "a projection of "
                    f"{store.dim} dimensions, not the space itself",
        }
        return payload

    @app.get("/api/meta")
    def meta() -> dict:
        return {
            "index_mode": store.meta.get("index_mode", "flat"),
            "chunks": len(store),
            "dim": store.dim,
            "embedding_model": store.meta.get("embedding_model", ""),
            "llm_model": getattr(config, "llm_model", ""),
            "strategies": list(STRATEGY_NAMES),
        }

    @app.get("/api/ask")
    def ask_endpoint(
        q: str = Query(...),
        strategy: str = Query("direct"),
        k: int | None = Query(None),
        rerank: bool = Query(False),
    ) -> dict:
        _check(q, [strategy])
        return _run(q, strategy, k, rerank)

    @app.get("/api/compare")
    def compare(
        q: str = Query(...),
        strategies: str | None = Query(None),
        k: int | None = Query(None),
        rerank: bool = Query(False),
    ) -> dict:
        names = (
            [s.strip() for s in strategies.split(",") if s.strip()]
            if strategies else list(STRATEGY_NAMES)
        )
        _check(q, names)
        return {
            "results": [
                {"strategy": name, "trace": _run(q, name, k, rerank)}
                for name in names
            ]
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
```

`STRATEGY_NAMES` already exists and is exported — `rag/strategies/__init__.py`
builds it as `tuple(_REGISTRY)` and lists it in `__all__`. Import it; do not
hardcode the six names in the dashboard, or the two lists will drift the
first time a strategy is added.

`STATIC` must exist before `StaticFiles` mounts, so create
`rag/dashboard/static/` with a placeholder `index.html` in this task; Task 3
replaces its contents.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 5: Commit**

```bash
git add rag/dashboard tests/test_dashboard_api.py
git commit -m "feat: read-only dashboard API over the pipeline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The page

**Files:**
- Create: `rag/dashboard/static/index.html`, `style.css`, `app.js`
- Test: `tests/test_dashboard_static.py`

The five panels from the spec: chunk table, strategy comparison, translation trace, stage timings, vector projection.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard_static.py`:

```python
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "rag" / "dashboard" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["index.html", "style.css", "app.js"])
def test_the_file_exists(name):
    assert (STATIC / name).is_file()


@pytest.mark.parametrize("panel", [
    "chunk-table", "strategy-comparison", "translation-trace",
    "stage-timings", "vector-projection",
])
def test_every_spec_panel_is_present(panel):
    assert panel in _read("index.html"), f"missing panel: {panel}"


def test_no_script_is_fetched_from_the_network():
    # A framework loaded from a CDN is still a framework wrapper, and the
    # project's premise is that none of this is one. This also keeps the
    # dashboard working offline.
    html = _read("index.html")
    remote = re.findall(r'<(?:script|link)[^>]+(?:src|href)\s*=\s*["\']https?://[^"\']+', html)
    assert not remote, f"remote asset(s): {remote}"


def test_no_front_end_framework_is_referenced():
    combined = (_read("index.html") + _read("app.js")).lower()
    for framework in ("react", "vue", "angular", "svelte", "streamlit", "jquery"):
        assert framework not in combined, f"references {framework}"


def test_each_panel_has_its_own_loading_state():
    # The LLM-heavy strategies take seconds; one shared spinner would block
    # the whole page on the slowest panel.
    html = _read("index.html")
    assert html.count("data-loading") >= 5


def test_the_projection_is_labelled_as_a_projection():
    # 384 dimensions do not fit in 2. The label stops a reader treating the
    # scatter as the embedding space.
    html = _read("index.html").lower()
    assert "projection" in html
    assert "not the space itself" in html


def test_the_page_calls_the_two_endpoints():
    js = _read("app.js")
    assert "/api/ask" in js
    assert "/api/compare" in js


def test_comparison_is_fetched_separately_from_ask():
    # If compare were awaited inside the same fetch as ask, the fast panels
    # would wait on the slow one, which is what the spec rules out.
    js = _read("app.js")
    assert js.count("fetch(") >= 2
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_dashboard_static.py -q`
Expected: failures for the missing `style.css`/`app.js` and the absent panel ids.

- [ ] **Step 3: Write the page**

`rag/dashboard/static/index.html` — five `<section>` elements with the ids
the test requires (`chunk-table`, `strategy-comparison`, `translation-trace`,
`stage-timings`, `vector-projection`), each carrying its own `data-loading`
element, a question input, a strategy select, a `k` input and a rerank
checkbox. The projection section must carry the text "not the space itself".
Link `style.css` and `app.js` with relative `/static/...` paths — no remote
assets.

`rag/dashboard/static/app.js` — plain ES modules-free JavaScript:
- on submit, fire `fetch('/api/ask?...')` and `fetch('/api/compare?...')` as
  **two independent requests**, each clearing its own `data-loading` when it
  resolves, so the four fast panels paint while comparison is still running;
- render the chunk table with rank, `score_kind` + score, `doc_id`, chunk
  index and a text excerpt;
- render translation entries as `kind: text`;
- render timings as a simple bar per stage, using `ms`;
- draw the projection on a `<canvas>` with `2d` context: chunks as dots, the
  query as a distinct marker (different shape and colour), scaled to fit;
- show an error panel on a non-200 rather than failing silently.

`rag/dashboard/static/style.css` — a plain readable stylesheet. No framework.

Keep all three files free of any `http://` or `https://` asset reference.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 5: Commit**

```bash
git add rag/dashboard/static tests/test_dashboard_static.py
git commit -m "feat: the retrieval inspector page, vanilla JS and no build step

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Serve it, and document it

**Files:**
- Modify: `rag/__main__.py`, `pyproject.toml`, `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `python -m rag dashboard [--host 127.0.0.1] [--port 8000] [--index {flat,multirep,raptor}]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_dashboard_subcommand_exists(tiny_corpus, monkeypatch):
    # Parsed without starting a server: uvicorn.run blocks forever.
    import rag.__main__ as cli

    monkeypatch.setattr(cli, "load_config", lambda **kw: tiny_corpus)
    started = {}

    def fake_serve(app, host, port):
        started["host"] = host
        started["port"] = port

    monkeypatch.setattr(cli, "_serve", fake_serve)
    main(["index"], **_factories())
    assert main(["dashboard", "--port", "9123"], **_factories()) == 0
    assert started["port"] == 9123


def test_the_dashboard_binds_localhost_by_default(tiny_corpus, monkeypatch):
    # No authentication, and it can spend Gemini quota, so it must not be
    # reachable from the network unless the operator asks for that.
    import rag.__main__ as cli

    monkeypatch.setattr(cli, "load_config", lambda **kw: tiny_corpus)
    started = {}
    monkeypatch.setattr(cli, "_serve",
                        lambda app, host, port: started.update(host=host))
    main(["index"], **_factories())
    main(["dashboard"], **_factories())
    assert started["host"] == "127.0.0.1"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_cli.py -q`
Expected: `AttributeError: module 'rag.__main__' has no attribute '_serve'`, and argparse rejecting `dashboard`.

- [ ] **Step 3: Add the subcommand**

In `rag/__main__.py`:

```python
def _serve(app, host: str, port: int) -> None:
    """Run the server. Split out so the CLI test can stub it -- uvicorn.run
    blocks forever, which a test cannot call."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
```

Add the subparser (defaulting `--host` to `127.0.0.1`, `--port` to 8000,
`--index` to `flat`), and in `_run` build the store, embedder and LLM, call
`create_app(...)`, then `_serve(app, args.host, args.port)` and return 0.
Import `create_app` **inside** the branch, not at module top level, so
`python -m rag ask` does not require FastAPI to be installed.

- [ ] **Step 4: Declare the optional dependency**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
dashboard = ["fastapi>=0.115", "uvicorn>=0.30"]
```

`tests/test_no_frameworks.py` parses every optional-dependency group, so
confirm it still passes — neither name is on the forbidden list, but check
rather than assume.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 6: Run it for real**

Start it against the real index, in the background, and check it serves:

```
python -m rag dashboard --port 8765
```

Then in another shell:

```
curl -s http://127.0.0.1:8765/api/meta
curl -s "http://127.0.0.1:8765/api/ask?q=How+does+ColBERT+score+a+document%3F&k=5"
```

Report the `/api/meta` body and the first retrieved chunk's `chunk_id`,
`score_kind` and `score`. Confirm the projection has one coordinate pair per
retrieved chunk. **Stop the server when done** — do not leave it running.

- [ ] **Step 7: Write the README section**

Cover what the dashboard is for, the five panels, how to install
(`pip install -e ".[dashboard]"`) and run it, and that it is read-only and
binds localhost. State plainly that the projection is 384 dimensions squeezed
into 2 and should not be read as the embedding space. Tick **Phase 6** in the
roadmap — 6a and 6b together complete it.

- [ ] **Step 8: Commit**

```bash
git add rag/__main__.py pyproject.toml README.md tests/test_cli.py
git commit -m "feat: serve the retrieval inspector from the CLI

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 6b Definition of Done

- [ ] `python -m pytest` passes offline; no test starts a server or loads a model
- [ ] All five spec panels exist and each has its own loading state
- [ ] No React, no build step, no Streamlit, and no asset fetched over the network
- [ ] Every endpoint is a GET; mutating verbs return 405
- [ ] The projection is computed with `np.linalg.svd` and labelled as a projection
- [ ] The no-frameworks guard demonstrably scans `rag/dashboard/`
- [ ] The server binds `127.0.0.1` unless told otherwise
- [ ] `python -m rag ask` still works without FastAPI installed
- [ ] Phase 6 ticked in the roadmap
- [ ] Working tree clean
