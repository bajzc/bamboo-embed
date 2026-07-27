from __future__ import annotations

from guji.retrieve.hybrid import _match, _where


def test_match_by_title_or_id():
    m = {"title": "史記", "book_id": "shiji", "dynasty": "漢", "category": "史書"}
    assert _match(m, "史記", None, None)
    assert _match(m, "shiji", None, None)         # id also matches
    assert not _match(m, "漢書", None, None)


def test_match_dynasty_and_category():
    m = {"title": "史記", "book_id": "shiji", "dynasty": "漢", "category": "史書"}
    assert _match(m, None, "漢", "史書")
    assert not _match(m, None, "唐", None)
    assert not _match(m, None, None, "小說")


def test_where_clause_and_escaping():
    assert _where(None, None, None) is None
    w = _where("史記", "漢", None)
    assert "title = '史記'" in w and "book_id = '史記'" in w and "dynasty = '漢'" in w
    assert " AND " in w
    # single quotes are escaped (appears in both title and book_id clauses)
    assert _where("a'b", None, None).count("''") == 2
