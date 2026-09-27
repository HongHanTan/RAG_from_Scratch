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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.embedding import l2_normalize
from rag.similarity import top_k


@dataclass
class VectorStore:
    vectors: np.ndarray       # (n_chunks, dim) float32, L2-normalised
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
        self.vectors = l2_normalize(self.vectors)

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
        # self.vectors is already unit-length (see __post_init__); only the
        # (small) query batch needs normalising here.
        scores = l2_normalize(query_vectors) @ self.vectors.T
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
