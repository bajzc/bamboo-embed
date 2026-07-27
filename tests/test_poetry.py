from __future__ import annotations

from guji.parse import manifest, poetry


def _poems(corpus, title):
    m = manifest.build_manifest(corpus)
    book = next(b for b in m.books if b.title == title)
    return list(poetry.parse_book(book, corpus, None, 250, 400, 1))


def test_one_poem_per_chunk_with_group(corpus):
    ps = _poems(corpus, "唐詩三百首")
    assert len(ps) == 2
    juans = [p.juan for p in ps]
    assert "五言古詩·感遇(四首之一)" in juans
    # per-poem author captured from '○ 詩題 / 作者'
    authors = {p.author for p in ps}
    assert "張九齡" in authors and "李白" in authors


def test_shijing_poems_not_split(corpus):
    ps = _poems(corpus, "詩經")
    assert len(ps) == 2
    assert ps[0].juan == "國風·周南·關雎"
    assert "關關雎鳩" in ps[0].text_raw


def test_prose_meng_xue_falls_back_to_narrative(corpus, make_book):
    # 千字文-style: no ○ markers -> narrative paragraph chunking
    make_book(
        corpus,
        "n.蒙學/千字文.txt",
        "# 千字文\n\n" + "".join("天地玄黃，宇宙洪荒。\n\n" for _ in range(40)),
    )
    ps = _poems(corpus, "千字文")
    assert ps  # produced chunks despite no ○ markers
    assert all(p.text_raw for p in ps)
