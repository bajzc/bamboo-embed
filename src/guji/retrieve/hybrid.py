"""Hybrid retrieval chain (§4.1):

    query ─┬─ dense  (HyDE pseudo-doc or query vector) → LanceDB ─┐
           └─ sparse (raw query, char-bigram)          → FTS5   ─┴─ RRF
        → metadata filter → rerank (qwen3-rerank) → score threshold → related-book dedup
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..index import fts, vector
from ..index.embed import make_embedder
from . import dedup, hyde, rerank


def rrf_fuse(rankings: list[list[str]], k: int, rrf_k: int) -> list[tuple[str, float]]:
    """Fuse ranked id-lists. score = Σ 1/(rrf_k + rank), rank 0-based."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


@dataclass
class Hit:
    chunk_id: str
    rrf_score: float
    rerank_score: float | None
    dense_rank: int | None
    sparse_rank: int | None
    meta: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    hits: list[Hit]
    hyde_text: str | None = None
    dense_top: list[str] = field(default_factory=list)
    sparse_top: list[str] = field(default_factory=list)
    rejected_by_threshold: bool = False


def _match(meta: dict, book: str | None, dynasty: str | None, category: str | None) -> bool:
    if book and meta.get("title") != book and meta.get("book_id") != book:
        return False
    if dynasty and meta.get("dynasty") != dynasty:
        return False
    if category and meta.get("category") != category:
        return False
    return True


def _where(book: str | None, dynasty: str | None, category: str | None) -> str | None:
    def esc(v: str) -> str:
        return v.replace("'", "''")

    clauses = []
    if book:
        clauses.append(f"(title = '{esc(book)}' OR book_id = '{esc(book)}')")
    if dynasty:
        clauses.append(f"dynasty = '{esc(dynasty)}'")
    if category:
        clauses.append(f"category = '{esc(category)}'")
    return " AND ".join(clauses) if clauses else None


def search(
    cfg: Config,
    query: str,
    top_k: int | None = None,
    use_hyde: bool | None = None,
    use_rerank: bool = True,
    book: str | None = None,
    dynasty: str | None = None,
    category: str | None = None,
) -> SearchResult:
    top_k = top_k or cfg.retrieve.top_k
    use_hyde = cfg.hyde.enabled if use_hyde is None else use_hyde

    # --- dense path (HyDE pseudo-doc, else instructed query) ---
    embedder = make_embedder(cfg)
    hyde_text = None
    try:
        if use_hyde:
            hyde_text = hyde.generate(cfg, query)
        qvec = (
            embedder.embed_documents([hyde_text])[0]
            if hyde_text
            else embedder.embed_query(query)
        )
    finally:
        embedder.close()

    db = vector.connect(cfg.lancedb_path)
    table = vector.open_table(db, cfg.embedding.dim)
    where = _where(book, dynasty, category)
    dense = [h["chunk_id"] for h in vector.search(table, qvec, cfg.retrieve.dense_k, where)]

    # --- sparse path (raw query) ---
    sparse = [cid for cid, _ in fts.search(cfg.fts_db_path, query, cfg.retrieve.sparse_k)]

    # --- metadata filter (belt-and-suspenders: sparse path isn't prefiltered) ---
    meta = fts.fetch_meta(cfg.fts_db_path, list(dict.fromkeys(dense + sparse)))
    keep = lambda cid: cid in meta and _match(meta[cid], book, dynasty, category)
    dense = [c for c in dense if keep(c)]
    sparse = [c for c in sparse if keep(c)]

    # --- RRF fusion ---
    fused = rrf_fuse([dense, sparse], cfg.retrieve.fuse_k, cfg.retrieve.rrf_k)
    drank = {c: i for i, c in enumerate(dense)}
    srank = {c: i for i, c in enumerate(sparse)}
    hits = [
        Hit(cid, sc, None, drank.get(cid), srank.get(cid), meta.get(cid, {}))
        for cid, sc in fused
    ]

    # --- rerank + score threshold ---
    rejected = False
    reranker = rerank.make_reranker(cfg) if use_rerank else None
    if reranker and hits:
        scores = reranker.rerank(query, [h.meta.get("text_raw", "") for h in hits])
        for h, s in zip(hits, scores):
            h.rerank_score = s
        hits.sort(key=lambda h: h.rerank_score, reverse=True)
        if hits[0].rerank_score < cfg.rerank.threshold:
            rejected = True
            hits = []

    # --- related-book dedup ---
    if cfg.dedup.enabled and hits:
        related = dedup.load_related_map(cfg.manifest_path)
        hits = dedup.collapse(hits, related, cfg.dedup.jaccard)

    return SearchResult(
        hits=hits[:top_k],
        hyde_text=hyde_text,
        dense_top=dense[:10],
        sparse_top=sparse[:10],
        rejected_by_threshold=rejected,
    )
