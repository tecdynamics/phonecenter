"""Retriever — embed the query, search the store, format a context block.

Assembles retrieved passages into the text block the persona's grounding gate
(§7) answers from. Passes all passage fields through (spec §5.5: don't silently
drop data). Returns "" when nothing clears the score threshold, so the agent
falls back to "I don't have that" + escalation rather than inventing an answer.
"""

from __future__ import annotations

import logging

from .embeddings import EmbeddingClient, EmbeddingError
from .store import Passage, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingClient,
        store: VectorStore,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    async def retrieve(self, query: str) -> list[Passage]:
        try:
            qvec = await self._embeddings.embed_one(query, "query")
        except EmbeddingError:
            logger.exception("Embedding query failed; returning no context")
            return []
        hits = self._store.search(qvec, self._top_k)
        kept = [h for h in hits if h.score >= self._min_score]
        logger.info(
            "RAG: %d/%d passages above %.2f (top score %.3f)",
            len(kept), len(hits), self._min_score, hits[0].score if hits else 0.0,
        )
        return kept

    async def context_for(self, query: str) -> str:
        """Retrieve and format a context block (empty string if nothing relevant)."""
        passages = await self.retrieve(query)
        return format_context(passages)


def format_context(passages: list[Passage]) -> str:
    if not passages:
        return ""
    blocks = []
    for i, p in enumerate(passages, 1):
        src = p.meta.get("source")
        head = f"[{i}]" + (f" ({src})" if src else "")
        blocks.append(f"{head}\n{p.text.strip()}")
    return "\n\n".join(blocks)
