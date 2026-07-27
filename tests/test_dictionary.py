from __future__ import annotations

from guji.parse import dictionary, manifest


def _entries(corpus, title):
    m = manifest.build_manifest(corpus)
    book = next(b for b in m.books if b.title == title)
    return list(dictionary.parse_book(book, corpus, None))


def test_tab_continuations_grouped(corpus):
    entries = {e.headword: e for e in _entries(corpus, "說文解字")}
    # 㐁 gathers both 大徐本 and 小徐本 continuation lines into one body
    assert "大徐本" in entries["㐁"].body_raw
    assert "小徐本" in entries["㐁"].body_raw
    assert "敝" in entries


def test_section_extracted(corpus):
    entries = {e.headword: e for e in _entries(corpus, "康熙字典")}
    assert entries["敝"].section == "【卯集下】【攴字部】"


def test_kangxi_shards_merge_entries(corpus):
    entries = _entries(corpus, "康熙字典")
    heads = {e.headword for e in entries}
    assert {"敝", "龍"} <= heads          # 敝 from p1, 龍 from p2
    assert all(e.source_book == "康熙字典" for e in entries)


def test_sqlite_roundtrip(corpus, tmp_path):
    db = tmp_path / "dict.sqlite"
    conn = dictionary.create_db(db)
    n = dictionary.insert_entries(conn, iter(_entries(corpus, "說文解字")))
    got = conn.execute("SELECT COUNT(*) FROM char_entry").fetchone()[0]
    conn.close()
    assert n == got >= 2
