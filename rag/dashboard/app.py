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

    # chunk_id -> row in `store.vectors`, built once for the server's lifetime.
    # The store never changes under us -- nothing here mutates it -- so the
    # alternative, scanning `store.chunks` for each retrieved item on every
    # request, would redo the same 5k-row walk k times per request to reach
    # the same answer.
    row_of = {chunk.chunk_id: i for i, chunk in enumerate(store.chunks)}

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
            store.vectors[row_of[item.chunk.chunk_id]]
            for item in trace.retrieved
            if item.chunk.chunk_id in row_of
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
