from __future__ import annotations

from guji.parse import manifest, narrative


def _book(corpus, title):
    m = manifest.build_manifest(corpus)
    return corpus, next(b for b in m.books if b.title == title)


def _passages(corpus, title):
    root, book = _book(corpus, title)
    return list(narrative.parse_book(book, root, None, 250, 400, 1))


def test_hard_boundary_between_juan(corpus):
    ps = _passages(corpus, "史記")
    juan_indices = {p.juan_idx for p in ps}
    assert juan_indices == {1, 2}
    # no chunk mixes 卷一 (甲) with 卷二 (乙)
    for p in ps:
        assert not ("甲" in p.text_raw and "乙" in p.text_raw)


def test_chunk_sizes_within_band(corpus):
    ps = [p for p in _passages(corpus, "史記") if p.juan_idx == 1]
    # every chunk except possibly the last should reach the lower bound
    for p in ps[:-1]:
        assert p.char_count >= 250


def test_paragraph_overlap(corpus):
    ps = [p for p in _passages(corpus, "史記") if p.juan_idx == 1]
    if len(ps) >= 2:
        # 1-paragraph overlap => consecutive chunks share a para index
        assert ps[0].para_end == ps[1].para_start


def test_prev_next_linkage(corpus):
    ps = _passages(corpus, "史記")
    assert ps[0].prev_id is None
    assert ps[-1].next_id is None
    for a, b in zip(ps, ps[1:]):
        assert a.next_id == b.chunk_id
        assert b.prev_id == a.chunk_id


def test_juan_and_embed_prefix(corpus):
    ps = _passages(corpus, "史記")
    p = ps[0]
    assert p.juan == "卷一‧五帝本紀第一"
    assert p.text_for_embed.startswith("《史記》卷一")
