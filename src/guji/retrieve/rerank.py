"""Cross-encoder rerank (§4.1).

DashScope's rerank endpoint is NOT OpenAI-compatible, so it needs its own adapter
(spec §2.1). `gte-rerank` was retired 2026-05-30; we use `qwen3-rerank`. Rerank is
allowed to be a cloud call during development — it touches no index assets.

`local` backend targets a separate `llama-server --reranking --pooling rank
--embedding` process (its own GGUF reranker model, own port — distinct from the
LLM's llama-server). Its `/v1/rerank` response shape (`results[].index` /
`.relevance_score`) happens to match DashScope's, so both adapters share scoring logic.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from .. import procman
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


class LlamaServerReranker:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        procman.ensure_reranker()
        resp = self._client.post(
            f"{self.base_url}/v1/rerank",
            json={"model": self.model, "query": query, "documents": documents},
        )
        resp.raise_for_status()
        results = resp.json()["results"]
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
    if backend == "local":
        return LlamaServerReranker(cfg.rerank.base_url, cfg.rerank.model)
    raise ValueError(f"unsupported rerank backend: {backend}")
