"""Three function-calling tools exposed to the generation LLM (§4).

    search_passages  — hybrid retrieval over narrative/poetry (dense+sparse+rerank).
    lookup_char      — exact dictionary lookup (字書訓詁), O(1) instead of vector search.
    get_context      — walk the prev_id/next_id chain around a chunk for surrounding text.

Each returns plain JSON-able dicts carrying citation metadata (title/juan/chunk_id),
since :mod:`guji.generate` validates the model's citations and quotes against exactly
these dicts.
"""

from __future__ import annotations

import sqlite3

from .config import Config
from .index import fts
from .normalize import to_norm
from .retrieve import hybrid

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_passages",
            "description": "混合检索叙事文本（史書/小說/子部/詩詞）。返回带完整出处的段落。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题（现代汉语或文言文均可）"},
                    "book": {"type": "string", "description": "限定书名或 book_id（可选）"},
                    "dynasty": {"type": "string", "description": "限定朝代（可选）"},
                    "category": {"type": "string", "description": "限定部类，如 史書/小說（可选）"},
                    "top_k": {"type": "integer", "description": "返回段落数，默认 8"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_char",
            "description": "精確查字書。返回 說文解字/康熙字典/大廣益會玉篇/一切經音義 等各家注解。",
            "parameters": {
                "type": "object",
                "properties": {
                    "char": {"type": "string", "description": "要查询的单字或词头"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "限定字书来源，如 [\"說文解字\", \"康熙字典\"]（可选）",
                    },
                },
                "required": ["char"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "取某段的前后文。檢索命中的往往是片段，需要上下文才能作答。",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "段落 chunk_id"},
                    "before": {"type": "integer", "description": "向前取几段，默认 1"},
                    "after": {"type": "integer", "description": "向后取几段，默认 1"},
                },
                "required": ["chunk_id"],
            },
        },
    },
]


def _hit_to_dict(h: hybrid.Hit) -> dict:
    m = h.meta
    return {
        "chunk_id": h.chunk_id,
        "title": m.get("title", ""),
        "dynasty": m.get("dynasty", ""),
        "author": m.get("author", ""),
        "category": m.get("category", ""),
        "juan": m.get("juan", ""),
        "text_raw": m.get("text_raw", ""),
        "rerank_score": h.rerank_score,
    }


def search_passages(
    cfg: Config,
    query: str,
    book: str | None = None,
    dynasty: str | None = None,
    category: str | None = None,
    top_k: int = 8,
) -> list[dict]:
    res = hybrid.search(cfg, query, top_k=top_k, book=book, dynasty=dynasty, category=category)
    return [_hit_to_dict(h) for h in res.hits]


def lookup_char(cfg: Config, char: str, sources: list[str] | None = None) -> list[dict]:
    norm = to_norm(char, str(cfg.pua_map_path))
    conn = sqlite3.connect(cfg.dict_db_path)
    conn.row_factory = sqlite3.Row
    try:
        q = "SELECT headword, source_book, body_raw, section FROM char_entry WHERE (headword = ? OR headword_norm = ?)"
        params: list = [char, norm]
        if sources:
            q += f" AND source_book IN ({','.join('?' * len(sources))})"
            params += sources
        rows = conn.execute(q, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_context(cfg: Config, chunk_id: str, before: int = 1, after: int = 1) -> list[dict]:
    anchor = fts.fetch_meta(cfg.fts_db_path, [chunk_id]).get(chunk_id)
    if not anchor:
        return []

    chain = [anchor]
    cur = anchor
    for _ in range(before):
        if not cur.get("prev_id"):
            break
        cur = fts.fetch_meta(cfg.fts_db_path, [cur["prev_id"]]).get(cur["prev_id"])
        if not cur:
            break
        chain.insert(0, cur)

    cur = anchor
    for _ in range(after):
        if not cur.get("next_id"):
            break
        cur = fts.fetch_meta(cfg.fts_db_path, [cur["next_id"]]).get(cur["next_id"])
        if not cur:
            break
        chain.append(cur)

    return [
        {
            "chunk_id": m["chunk_id"],
            "title": m.get("title", ""),
            "dynasty": m.get("dynasty", ""),
            "author": m.get("author", ""),
            "category": m.get("category", ""),
            "juan": m.get("juan", ""),
            "text_raw": m.get("text_raw", ""),
        }
        for m in chain
    ]


def call_tool(cfg: Config, name: str, arguments: dict) -> list[dict]:
    if name == "search_passages":
        return search_passages(cfg, **arguments)
    if name == "lookup_char":
        return lookup_char(cfg, **arguments)
    if name == "get_context":
        return get_context(cfg, **arguments)
    raise ValueError(f"unknown tool: {name}")
