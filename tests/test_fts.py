from __future__ import annotations

import json

from guji.index import fts


def _passage(cid, text, **extra):
    base = {c: "" for c in fts.META_COLS}
    base.update(chunk_id=cid, text_raw=text, text_norm=text, juan_idx=1, **extra)
    base["text_norm"] = text
    return base


def _write_jsonl(path, passages):
    with path.open("w", encoding="utf-8") as f:
        for p in passages:
            row = dict(p)
            row.setdefault("text_norm", p["text_raw"])
            f.write(json.dumps(row) + "\n")


def test_bigrams():
    assert fts.bigrams("克己復禮") == "克己 己復 復禮"
    assert fts.bigrams("水") == "水"          # single char fallback
    assert fts.bigrams("a b") == "ab"          # whitespace stripped


def test_fts_finds_phrase(tmp_path):
    jsonl = tmp_path / "passages.jsonl"
    db = tmp_path / "fts.sqlite"
    _write_jsonl(
        jsonl,
        [
            _passage("lunyu/j012/p1", "顏淵問仁。子曰：克己復禮為仁。", title="論語", juan="顏淵"),
            _passage("shiji/j001/p1", "黃帝者，少典之子。", title="史記", juan="五帝本紀"),
        ],
    )
    n = fts.build(db, jsonl)
    assert n == 2

    hits = fts.search(db, "克己復禮", k=5)
    assert hits, "expected a match"
    assert hits[0][0] == "lunyu/j012/p1"


def test_fetch_meta(tmp_path):
    jsonl = tmp_path / "passages.jsonl"
    db = tmp_path / "fts.sqlite"
    _write_jsonl(jsonl, [_passage("a/b/c", "測試文本", title="測試", juan="卷一")])
    fts.build(db, jsonl)
    meta = fts.fetch_meta(db, ["a/b/c"])
    assert meta["a/b/c"]["title"] == "測試"
    assert meta["a/b/c"]["juan"] == "卷一"
