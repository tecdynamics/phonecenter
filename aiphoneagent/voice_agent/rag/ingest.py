"""Build a per-client RAG index from source documents.

    python -m voice_agent.rag.ingest --src data/zai --out indexes/zai.npz

Reads, from the --src directory (recursively):
  * .jsonl  — one JSON object per line: {"text": "...", "source": "...", ...}
              (best for FAQs / structured data — each line is one chunk, all
              extra keys are kept as metadata and passed to the model)
  * .md/.txt — split into paragraph chunks (~CHUNK_CHARS each)

Each chunk is embedded with the e5 `passage:` prefix and saved to the .npz that
LocalVectorStore.load() reads. Run again to rebuild after content changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from ..config import get_settings
from .embeddings import EmbeddingClient
from .store import save_index

logger = logging.getLogger("voice_agent.rag.ingest")

CHUNK_CHARS = 600   # target chunk size for prose; FAQs in .jsonl are 1 chunk each
EMBED_BATCH = 32


def _chunk_text(text: str, source: str) -> list[dict]:
    """Paragraph-pack a prose document into ~CHUNK_CHARS chunks."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 2 > CHUNK_CHARS:
            chunks.append({"text": buf, "source": source})
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append({"text": buf, "source": source})
    return chunks


def _load_records(src: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("text"):
                    obj.setdefault("source", path.name)
                    records.append(obj)
        elif path.suffix.lower() in (".md", ".txt"):
            records.extend(_chunk_text(path.read_text(encoding="utf-8"), path.name))
    return records


async def _run(src: Path, out: str) -> None:
    records = _load_records(src)
    if not records:
        raise SystemExit(f"No .jsonl/.md/.txt content found under {src}")
    logger.info("Embedding %d chunks from %s", len(records), src)

    settings = get_settings()
    embedder = EmbeddingClient(
        endpoint=settings.embedding_endpoint,
        model=settings.embedding_model,
        path=settings.embedding_path,
        add_prefixes=settings.embedding_add_prefixes,
    )
    try:
        vectors = []
        for i in range(0, len(records), EMBED_BATCH):
            batch = records[i : i + EMBED_BATCH]
            vecs = await embedder.embed([r["text"] for r in batch], "passage")
            vectors.append(vecs)
            logger.info("  embedded %d/%d", min(i + EMBED_BATCH, len(records)), len(records))
    finally:
        await embedder.aclose()

    import numpy as np

    matrix = np.vstack(vectors)
    texts = [r["text"] for r in records]
    metas = [{k: v for k, v in r.items() if k != "text"} for r in records]
    save_index(out, matrix, texts, metas)
    print(f"Indexed {len(texts)} chunks -> {out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build a RAG index for the voice agent.")
    ap.add_argument("--src", required=True, help="Directory of .jsonl/.md/.txt source docs")
    ap.add_argument("--out", required=True, help="Output index path, e.g. indexes/zai.npz")
    args = ap.parse_args()
    asyncio.run(_run(Path(args.src), args.out))


if __name__ == "__main__":
    main()
