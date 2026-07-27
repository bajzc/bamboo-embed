"""SQLite FTS5 sparse index with character-bigram tokenization (§4.1).

FTS5's unicode61 tokenizer collapses a run of CJK into a single token, so we
pre-tokenize into character bigrams (never jieba — its modern-Chinese dictionary
shreds 文言文). Both documents and queries are bigrammed identically. A ``meta``
sidecar table stores display fields so result rendering is a fast key lookup.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

META_COLS = ["chunk_id", "book_id", "title", "dynasty", "author", "category",
             "juan", "juan_idx", "text_raw", "prev_id", "next_id"]


def bigrams(text: str) -> str:
    """Space-joined character bigrams; falls back to the lone char if len < 2."""
    t = [c for c in text if not c.isspace()]
    if len(t) < 2:
        return "".join(t)
    return " ".join(t[i] + t[i + 1] for i in range(len(t) - 1))


def _match_query(query: str) -> str:
    toks = bigrams(query).split()
    # quote each bigram so FTS5 treats it literally (bigrams may contain punctuation)
    return " OR ".join('"' + tok.replace('"', '""') + '"' for tok in toks if tok)


def build(fts_db_path: Path, passages_path: Path, batch: int = 5000) -> int:
    fts_db_path.parent.mkdir(parents=True, exist_ok=True)
    if fts_db_path.exists():
        fts_db_path.unlink()
    conn = sqlite3.connect(fts_db_path)
    conn.execute("CREATE VIRTUAL TABLE fts USING fts5(chunk_id UNINDEXED, body, tokenize='unicode61')")
    conn.execute(
        f"CREATE TABLE meta ({', '.join(c + (' PRIMARY KEY' if c == 'chunk_id' else '') for c in META_COLS)})"
    )

    n = 0
    fts_rows: list[tuple] = []
    meta_rows: list[tuple] = []

    def flush():
        conn.executemany("INSERT INTO fts(chunk_id, body) VALUES (?,?)", fts_rows)
        conn.executemany(
            f"INSERT INTO meta({','.join(META_COLS)}) VALUES ({','.join('?' * len(META_COLS))})",
            meta_rows,
        )
        fts_rows.clear()
        meta_rows.clear()

    with passages_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            fts_rows.append((p["chunk_id"], bigrams(p["text_norm"])))
            meta_rows.append(tuple(p.get(c) for c in META_COLS))
            n += 1
            if len(fts_rows) >= batch:
                flush()
        flush()
    conn.commit()
    conn.close()
    return n


def search(fts_db_path: Path, query: str, k: int) -> list[tuple[str, float]]:
    """Return (chunk_id, bm25_score) best-first (lower bm25 = better -> negated)."""
    match = _match_query(query)
    if not match:
        return []
    conn = sqlite3.connect(fts_db_path)
    try:
        rows = conn.execute(
            "SELECT chunk_id, bm25(fts) AS s FROM fts WHERE fts MATCH ? ORDER BY s LIMIT ?",
            (match, k),
        ).fetchall()
    finally:
        conn.close()
    return [(cid, -s) for cid, s in rows]


def fetch_meta(fts_db_path: Path, chunk_ids: Iterable[str]) -> dict[str, dict]:
    ids = list(chunk_ids)
    if not ids:
        return {}
    conn = sqlite3.connect(fts_db_path)
    conn.row_factory = sqlite3.Row
    try:
        q = f"SELECT {','.join(META_COLS)} FROM meta WHERE chunk_id IN ({','.join('?' * len(ids))})"
        rows = conn.execute(q, ids).fetchall()
    finally:
        conn.close()
    return {r["chunk_id"]: dict(r) for r in rows}
