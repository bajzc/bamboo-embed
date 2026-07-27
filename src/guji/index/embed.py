"""Embeddings via Ollama (§2.1, backend pivoted from MLX to Ollama).

Cross-platform: the same Ollama model produces identical vectors on the Linux
dev box and the macOS deploy, so the index never needs rebuilding across
platforms — which was the spec's whole reason for pinning the embedding model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx

from . import vector


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, query_instruct: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.query_instruct = query_instruct
        self._client = httpx.Client(timeout=timeout)

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self.base_url}/api/embed", json={"model": self.model, "input": inputs}
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        # Qwen3-Embedding query template: "Instruct: <task>\nQuery: <q>"
        if self.query_instruct:
            text = f"Instruct: {self.query_instruct}\nQuery: {text}"
        return self._embed([text])[0]

    def dim(self) -> int:
        return len(self.embed_documents(["dim probe"])[0])

    def close(self) -> None:
        self._client.close()


def make_embedder(cfg) -> OllamaEmbedder:
    e = cfg.embedding
    if e.backend != "ollama":
        raise ValueError(f"unsupported embedding backend: {e.backend}")
    return OllamaEmbedder(e.base_url, e.model, e.query_instruct)


def _load_passages(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_corpus(
    cfg,
    limit: int | None = None,
    book_ids: set[str] | None = None,
    batch: int = 32,
    on_start: Callable[[int], None] | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """Embed passages into LanceDB, resumably. Returns (already_done, newly_added).

    Resumability: rows already present in the table are skipped, so an interrupted
    run continues where it stopped — never re-embeds from scratch (§5.4).
    """
    passages = _load_passages(cfg.passages_path)
    if book_ids:
        passages = [p for p in passages if p["book_id"] in book_ids]

    db = vector.connect(cfg.lancedb_path)
    table = vector.open_table(db, cfg.embedding.dim)
    done = vector.existing_ids(table)

    todo = [p for p in passages if p["chunk_id"] not in done]
    if limit:
        todo = todo[:limit]
    if on_start:
        on_start(len(todo))

    embedder = make_embedder(cfg)
    added = 0
    try:
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            vecs = embedder.embed_documents([p["text_for_embed"] for p in chunk])
            rows = [
                {
                    "chunk_id": p["chunk_id"],
                    "vector": v,
                    "book_id": p["book_id"],
                    "title": p["title"],
                    "dynasty": p.get("dynasty", ""),
                    "category": p["category"],
                    "juan_idx": int(p["juan_idx"]),
                }
                for p, v in zip(chunk, vecs)
            ]
            vector.add_rows(table, rows)
            added += len(rows)
            if on_batch:
                on_batch(len(rows))
    finally:
        embedder.close()
    return len(done), added
