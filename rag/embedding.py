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
