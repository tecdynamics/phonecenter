"""Vector store — local brute-force cosine search over a per-client .npz index.

For a phone-support KB (hundreds–few thousand chunks) brute force is sub-ms and
needs no extra service. `VectorStore` is the seam: a Qdrant/pgvector backend can
replace `LocalVectorStore` without touching the retriever or orchestrator.

Index file (built by ingest.py) is a single .npz with:
  vectors : float32 (N, D), L2-normalized
  texts   : object  (N,)   passage text
  metas   : object  (N,)   JSON string per passage (source, fields, ...)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    text: str
    score: float
    meta: dict = field(default_factory=dict)


class VectorStore(Protocol):
    def search(self, query_vec: np.ndarray, k: int) -> list[Passage]: ...


class LocalVectorStore:
    def __init__(self, vectors: np.ndarray, texts: list[str], metas: list[dict]) -> None:
        self._vectors = vectors  # (N, D) L2-normalized float32
        self._texts = texts
        self._metas = metas
        logger.info("Vector store loaded: %d passages, dim=%d",
                    len(texts), vectors.shape[1] if vectors.size else 0)

    @classmethod
    def load(cls, path: str) -> "LocalVectorStore":
        data = np.load(path, allow_pickle=True)
        vectors = data["vectors"].astype(np.float32)
        texts = [str(t) for t in data["texts"].tolist()]
        metas = [json.loads(m) if m else {} for m in data["metas"].tolist()]
        return cls(vectors, texts, metas)

    def search(self, query_vec: np.ndarray, k: int) -> list[Passage]:
        if self._vectors.size == 0:
            return []
        # vectors and query are L2-normalized → dot product == cosine similarity.
        scores = self._vectors @ query_vec.astype(np.float32)
        k = min(k, len(scores))
        # top-k indices, highest score first
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Passage(self._texts[i], float(scores[i]), self._metas[i]) for i in idx]


def save_index(path: str, vectors: np.ndarray, texts: list[str], metas: list[dict]) -> None:
    """Write an index file consumable by LocalVectorStore.load()."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vectors=vectors.astype(np.float32),
        texts=np.array(texts, dtype=object),
        metas=np.array([json.dumps(m, ensure_ascii=False) for m in metas], dtype=object),
    )
    logger.info("Saved index %s (%d passages)", path, len(texts))
