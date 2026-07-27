"""Cross-encoder rerank (§4.1).

DashScope's rerank endpoint is NOT OpenAI-compatible, so it needs its own adapter
(spec §2.1). `gte-rerank` was retired 2026-05-30; we use `qwen3-rerank`. Rerank is
allowed to be a cloud call during development — it touches no index assets.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from ..config import Config


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document, aligned to input order (higher = better)."""
        ...


class DashScopeReranker:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 60.0):
        self.base_url = base_url
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._client = httpx.Client(timeout=timeout)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        resp = self._client.post(
            self.base_url,
            headers=self._headers,
            json={
                "model": self.model,
                "input": {"query": query, "documents": documents},
                "parameters": {"return_documents": False},
            },
        )
        resp.raise_for_status()
        results = resp.json()["output"]["results"]
        scores = [0.0] * len(documents)
        for r in results:
            scores[r["index"]] = r["relevance_score"]
        return scores


def make_reranker(cfg: Config) -> Reranker | None:
    backend = cfg.rerank.backend
    if backend == "none":
        return None
    if backend == "dashscope":
        key = cfg.api_key(cfg.rerank.api_key_env)
        if not key:
            raise RuntimeError(
                f"rerank backend 'dashscope' needs {cfg.rerank.api_key_env} (set it in .env)"
            )
        return DashScopeReranker(cfg.rerank.base_url, cfg.rerank.model, key)
    raise ValueError(f"unsupported rerank backend: {backend}")
