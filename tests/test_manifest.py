from __future__ import annotations

from guji.parse import manifest


def _by_title(m):
    return {b.title: b for b in m.books}


def test_level2_head_and_dynasty(corpus):
    m = manifest.build_manifest(corpus)
    books = _by_title(m)
    # 唐詩三百首 uses '##' as its head line
    t = books["唐詩三百首"]
    assert t.dynasty == "清"
    assert "蘅塘退士" in t.author
    assert t.form == "poetry"


def test_multi_bracket_head(corpus):
    m = manifest.build_manifest(corpus)
    swjz = _by_title(m)["說文解字"]
    assert swjz.dynasty == "漢"       # first 〔〕 group
    assert swjz.form == "dictionary"


def test_kangxi_shards_merge_into_one_book(corpus):
    m = manifest.build_manifest(corpus)
    kx = _by_title(m)["康熙字典"]
    assert len(kx.source_files) == 2
    assert "康熙字典_p1.txt" in kx.source_files[0]
    assert "康熙字典_p2.txt" in kx.source_files[1]


def test_section_head_not_mistaken_for_title(corpus):
    # 南齊書 has no '# ' head; first '## ' is 序 (a section), title must be filename
    m = manifest.build_manifest(corpus)
    assert "南齊書" in _by_title(m)


def test_stub_and_empty_flags_and_count(corpus):
    m = manifest.build_manifest(corpus)
    by = _by_title(m)
    # the pointer file 詩經_symbolic -> stub, excluded from parse
    stub = [b for b in m.books if b.is_stub]
    assert len(stub) == 1 and "內容見" in stub[0].notes
    # empty poetry file flagged
    assert any(b.is_empty for b in m.books)
    non_empty = sum(not b.is_empty for b in m.books)
    assert non_empty == 7  # mini-corpus: 8 books - 1 empty


def test_related_titles_linked(corpus):
    # title is filename-derived: 詩經 (real) vs 詩經_symbolic (stub); they are linked
    m = manifest.build_manifest(corpus)
    real = next(b for b in m.books if b.title == "詩經")
    stub = next(b for b in m.books if b.title == "詩經_symbolic")
    assert stub.book_id in real.related_to
    assert real.book_id in stub.related_to
