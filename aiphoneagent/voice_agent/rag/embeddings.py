"""e5 embedding client — wraps the EXISTING multilingual-e5-large server (8081).

multilingual-e5 REQUIRES input prefixes: "query: " for search queries and
"passage: " for indexed documents. We add them here (toggle with add_prefixes).
Output is L2-normalized so cosine similarity == dot product downstream.

⚠️ CONFIRM-BEFORE-DEPLOY: the request/response shape below assumes an
OpenAI-compatible `POST /v1/embeddings` ({input:[...]} → {data:[{embedding}]}).
If the e5 server uses a different contract (e.g. TEI's `{inputs:[...]}` → `[[...]]`),
adjust `_request`/`_parse` here — nothing else depends on the wire format.
"""

from __future__ import annotations

import logging
from typing import Literal

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """The embedding endpoint was unreachable or returned an error."""


class EmbeddingClient:
    def __init__(
        self,
        endpoint: str,
        model: str = "multilingual-e5-large",
        path: str = "/v1/embeddings",
        api_key: str = "",
        add_prefixes: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._url = endpoint.rstrip("/") + "/" + path.lstrip("/")
        self._add_prefixes = add_prefixes
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _prep(self, texts: list[str], kind: Literal["query", "passage"]) -> list[str]:
        if not self._add_prefixes:
            return texts
        return [f"{kind}: {t}" for t in texts]

    async def embed(
        self, texts: list[str], kind: Literal["query", "passage"]
    ) -> np.ndarray:
        """Return an L2-normalized (N, D) float32 matrix of embeddings."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        payload = {"model": self._model, "input": self._prep(texts, kind)}
        try:
            resp = await self._client.post(self._url, json=payload)
            if resp.status_code >= 400:
                raise EmbeddingError(f"embeddings HTTP {resp.status_code}: {resp.text[:200]}")
            vecs = self._parse(resp.json())
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embeddings request failed: {exc}") from exc
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))

    async def embed_one(self, text: str, kind: Literal["query", "passage"]) -> np.ndarray:
        return (await self.embed([text], kind))[0]

    @staticmethod
    def _parse(data: dict) -> list[list[float]]:
        # OpenAI shape: {"data": [{"embedding": [...]}, ...]} (preserve order).
        items = data.get("data")
        if isinstance(items, list):
            return [d["embedding"] for d in items]
        # Fallbacks for common alt shapes: {"embeddings": [[...]]} or a bare list.
        if isinstance(data.get("embeddings"), list):
            return data["embeddings"]
        if isinstance(data, list):
            return data
        raise EmbeddingError(f"unrecognized embeddings response: {str(data)[:200]}")


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    if m.size == 0:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)
