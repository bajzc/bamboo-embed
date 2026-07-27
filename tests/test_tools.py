from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from guji import tools
from guji.index import fts
from guji.parse import dictionary


class _FakeConfig:
    """Minimal cfg stand-in exposing only the paths tools.py touches."""

    def __init__(self, dict_db_path=None, fts_db_path=None, pua_map_path=None):
        self.dict_db_path = dict_db_path
        self.fts_db_path = fts_db_path
        self.pua_map_path = pua_map_path or ""


def _make_dict_db(tmp_path):
    db = tmp_path / "dict.sqlite"
    conn = dictionary.create_db(db)
    conn.executemany(
        "INSERT INTO char_entry(headword,headword_norm,source_book,body_raw,body_norm,section)"
        " VALUES (?,?,?,?,?,?)",
        [
            ("敝", "敝", "說文解字", "帗也。", "帗也。", ""),
            ("敝", "敝", "康熙字典", "毘祭切，音幣。", "毘祭切，音币。", "【卯集下】【攴字部】"),
            ("國", "国", "說文解字", "邦也。", "邦也。", ""),
        ],
    )
    conn.commit()
    conn.close()
    return db


def _passage(cid, text, prev_id=None, next_id=None, **extra):
    base = {c: "" for c in fts.META_COLS}
    base.update(chunk_id=cid, text_raw=text, text_norm=text, juan_idx=1,
                prev_id=prev_id, next_id=next_id, **extra)
    return base


def _make_fts_db(tmp_path):
    jsonl = tmp_path / "passages.jsonl"
    db = tmp_path / "fts.sqlite"
    rows = [
        _passage("shiji/j001/p1", "黃帝者，少典之子。", next_id="shiji/j001/p2", title="史記", juan="卷一"),
        _passage("shiji/j001/p2", "軒轅之時，神農氏世衰。", prev_id="shiji/j001/p1",
                  next_id="shiji/j001/p3", title="史記", juan="卷一"),
        _passage("shiji/j001/p3", "黃帝崩，葬橋山。", prev_id="shiji/j001/p2", title="史記", juan="卷一"),
    ]
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fts.build(db, jsonl)
    return db


def test_lookup_char_exact_headword(tmp_path):
    cfg = _FakeConfig(dict_db_path=_make_dict_db(tmp_path))
    rows = tools.lookup_char(cfg, "敝")
    assert {r["source_book"] for r in rows} == {"說文解字", "康熙字典"}


def test_lookup_char_via_simplified_query(tmp_path):
    cfg = _FakeConfig(dict_db_path=_make_dict_db(tmp_path))
    rows = tools.lookup_char(cfg, "国")  # simplified; headword stored as traditional 國
    assert rows and rows[0]["headword"] == "國"


def test_lookup_char_sources_filter(tmp_path):
    cfg = _FakeConfig(dict_db_path=_make_dict_db(tmp_path))
    rows = tools.lookup_char(cfg, "敝", sources=["康熙字典"])
    assert len(rows) == 1
    assert rows[0]["source_book"] == "康熙字典"
    assert rows[0]["section"] == "【卯集下】【攴字部】"


def test_get_context_window_around_anchor(tmp_path):
    cfg = _FakeConfig(fts_db_path=_make_fts_db(tmp_path))
    ctx = tools.get_context(cfg, "shiji/j001/p2", before=1, after=1)
    ids = [c["chunk_id"] for c in ctx]
    assert ids == ["shiji/j001/p1", "shiji/j001/p2", "shiji/j001/p3"]


def test_get_context_clamps_at_chain_boundary(tmp_path):
    cfg = _FakeConfig(fts_db_path=_make_fts_db(tmp_path))
    ctx = tools.get_context(cfg, "shiji/j001/p1", before=5, after=5)
    ids = [c["chunk_id"] for c in ctx]
    assert ids == ["shiji/j001/p1", "shiji/j001/p2", "shiji/j001/p3"]  # no crash past the ends


def test_get_context_unknown_chunk_returns_empty(tmp_path):
    cfg = _FakeConfig(fts_db_path=_make_fts_db(tmp_path))
    assert tools.get_context(cfg, "nope/x/y") == []


def test_call_tool_dispatches_and_rejects_unknown(tmp_path, monkeypatch):
    cfg = _FakeConfig(dict_db_path=_make_dict_db(tmp_path))
    rows = tools.call_tool(cfg, "lookup_char", {"char": "敝"})
    assert len(rows) == 2

    with pytest.raises(ValueError):
        tools.call_tool(cfg, "not_a_tool", {})


def test_call_tool_forwards_search_passages_kwargs(monkeypatch):
    seen = {}

    def fake_search_passages(cfg, **kwargs):
        seen.update(kwargs)
        return [{"title": "論語"}]

    monkeypatch.setattr(tools, "search_passages", fake_search_passages)
    out = tools.call_tool(SimpleNamespace(), "search_passages", {"query": "仁", "book": "論語", "top_k": 3})
    assert out == [{"title": "論語"}]
    assert seen == {"query": "仁", "book": "論語", "top_k": 3}
