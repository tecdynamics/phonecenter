"""RAG (phase 3) — e5 embeddings + a local vector store for grounded answers.

Pipeline: embed the caller's query (e5 `query:` prefix) → cosine top-k against a
per-client index built by `ingest.py` (passages embedded with the `passage:`
prefix) → assemble a context block the persona's grounding gate (§7) answers
from. The store is behind `VectorStore` so it can be swapped for Qdrant/pgvector.
"""
