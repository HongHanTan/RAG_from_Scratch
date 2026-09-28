"""The vector store: a dense NumPy matrix and a parallel list of chunks.

Row i of `vectors` is the embedding of `chunks[i]`. That invariant is the whole
data structure — there is no index, no graph, no quantisation. Persistence is
.npz with the chunk metadata alongside as a JSON blob, loaded with
allow_pickle=False so the index file is never an arbitrary-code vector.

Vectors are L2-normalised once, at construction (which `load` goes through
too), rather than on every `search` call. `Embedder.encode` already returns
unit vectors, so re-normalising all n rows per query bought nothing but cost:
on 5,116 rows it was the majority of `search`'s time. A query is a single row,
so normalising it per call is free and still happens in `search`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rag.atomic import write_atomic
from rag.chunking import Chunk, RetrievedChunk
from rag.embedding import l2_normalize
from rag.similarity import top_k


def _with_npz_suffix(path: Path) -> Path:
    """np.savez_compressed appends .npz; make save and load agree about that."""
    return path if path.suffix == ".npz" else path.with_name(f"{path.name}.npz")


@dataclass
class VectorStore:
    vectors: np.ndarray       # (n_chunks, dim) float32, L2-normalised
    chunks: list[Chunk]
    meta: dict = field(default_factory=dict)
    doc_meta: dict = field(default_factory=dict)
    """Per-document metadata, keyed by doc_id.

    Retrieval needs this to filter by author, date or topic, and a `Chunk`
    knows only its `doc_id`. Persisting it in the index keeps `ask()` able to
    filter from a reloaded index without also loading the corpus.
    """

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if self.vectors.shape[0] != len(self.chunks):
            raise ValueError(
                f"{self.vectors.shape[0]} vectors but {len(self.chunks)} chunks; "
                "row i must be the embedding of chunks[i]"
            )
        self.vectors = l2_normalize(self.vectors)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(
        self, query_vectors: np.ndarray, k: int
    ) -> list[list[RetrievedChunk]]:
        """Nearest chunks for each query row, best first, ranked from one."""
        query_vectors = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        if len(self.chunks) == 0:
            return [[] for _ in range(query_vectors.shape[0])]
        # self.vectors is already unit-length (see __post_init__); only the
        # (small) query batch needs normalising here.
        scores = l2_normalize(query_vectors) @ self.vectors.T
        indices, values = top_k(scores, k)
        return [
            [
                RetrievedChunk(chunk=self.chunks[int(i)], score=float(s), rank=rank)
                for rank, (i, s) in enumerate(zip(row_i, row_s), start=1)
            ]
            for row_i, row_s in zip(indices, values)
        ]

    def save(self, path: Path, meta: dict | None = None) -> None:
        """Persist the index, recording how it was built.

        `meta` should carry the embedding model and chunking parameters, so a
        later load can refuse an index built under a different configuration.
        Without it, an index built at one chunk size and queried at another
        returns plausible nonsense: the dimensions still match, so nothing
        errors.
        """
        path = _with_npz_suffix(path)
        payload = json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False)
        meta_payload = json.dumps(meta or {}, sort_keys=True)

        def _write(target: Path) -> None:
            # Write through an open handle: np.savez_compressed appends .npz to
            # a *path* that lacks it, which would leave the temporary somewhere
            # the rename cannot find it.
            with open(target, "wb") as handle:
                np.savez_compressed(
                    handle,
                    vectors=self.vectors,
                    chunks=np.array(payload),
                    meta=np.array(meta_payload),
                    doc_meta=np.array(json.dumps(self.doc_meta, ensure_ascii=False)),
                )

        write_atomic(path, _write)

    @classmethod
    def load(cls, path: Path, expect_meta: dict | None = None) -> "VectorStore":
        """Load an index, optionally checking it was built as expected.

        An index written before provenance existed has no meta; that is
        tolerated rather than treated as a mismatch, so old indexes still load.
        """
        path = _with_npz_suffix(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"index not found: {path} — run `python -m rag index` first"
            )
        with np.load(path, allow_pickle=False) as data:
            vectors = data["vectors"].astype(np.float32)
            chunks = [Chunk(**record) for record in json.loads(str(data["chunks"]))]
            meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
            doc_meta = (
                json.loads(str(data["doc_meta"])) if "doc_meta" in data.files else {}
            )

        if expect_meta and meta:
            differing = {
                key: (meta[key], value)
                for key, value in expect_meta.items()
                if key in meta and meta[key] != value
            }
            if differing:
                detail = ", ".join(
                    f"{k}: index has {found!r}, config wants {wanted!r}"
                    for k, (found, wanted) in sorted(differing.items())
                )
                raise ValueError(
                    f"index at {path} was built with a different configuration "
                    f"({detail}) — rebuild it with `python -m rag index`"
                )

        store = cls(vectors=vectors, chunks=chunks)
        store.meta = meta
        store.doc_meta = doc_meta
        return store
