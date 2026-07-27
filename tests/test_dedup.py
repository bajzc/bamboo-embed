from __future__ import annotations

from guji.retrieve import dedup
from guji.retrieve.hybrid import Hit

REL = {"shiji": {"shijisanjiazhu"}, "shijisanjiazhu": {"shiji"}}


def _hit(book, text, juan=""):
    return Hit(f"{book}/x", 0.1, None, 0, 0,
               {"book_id": book, "text_raw": text, "juan": juan})


def test_jaccard_bounds():
    a = dedup.char_bigrams("克己復禮為仁")
    assert dedup.jaccard(a, a) == 1.0
    assert dedup.jaccard(a, dedup.char_bigrams("風馬牛不相及")) < 0.2


def test_collapse_drops_related_near_dup():
    text = "黃帝者，少典之子，姓公孫，名曰軒轅。生而神靈，弱而能言。"
    hits = [_hit("shiji", text), _hit("shijisanjiazhu", text + "〔集解〕注文")]
    out = dedup.collapse(hits, REL, threshold=0.7)
    assert len(out) == 1
    assert out[0].meta["book_id"] == "shiji"  # higher-ranked kept


def test_collapse_keeps_unrelated_even_if_similar():
    text = "子曰：學而時習之，不亦說乎。"
    hits = [_hit("lunyu", text), _hit("mengzi", text)]  # not related
    out = dedup.collapse(hits, {}, threshold=0.7)
    assert len(out) == 2


def test_collapse_keeps_distinct_related():
    hits = [
        _hit("shiji", "黃帝者，少典之子，姓公孫。"),
        _hit("shijisanjiazhu", "夏禹，名曰文命，鯀之子。"),  # different passage, no juan
    ]
    out = dedup.collapse(hits, REL, threshold=0.7)
    assert len(out) == 2


def test_collapse_parallel_edition_same_juan():
    # 史記 vs 史記三家注: annotation-heavy text (low text similarity) but same 卷/篇
    hits = [
        _hit("shiji", "黃帝者，少典之子，姓公孫，名曰軒轅。", juan="卷一‧五帝本紀第一"),
        _hit("shijisanjiazhu", "黃帝者，【集解】徐廣曰：「號有熊。」【索隱】案：有土德之瑞……",
             juan="卷一‧五帝本紀第一"),
    ]
    out = dedup.collapse(hits, REL, threshold=0.7)
    assert len(out) == 1
    assert out[0].meta["book_id"] == "shiji"  # higher-ranked kept


def test_collapse_keeps_same_book_different_passages():
    # same book + same juan but distinct neighbour passages must both survive
    hits = [
        _hit("shiji", "黃帝者，少典之子。", juan="卷一‧五帝本紀第一"),
        _hit("shiji", "東至于海，登丸山，及岱宗。", juan="卷一‧五帝本紀第一"),
    ]
    out = dedup.collapse(hits, REL, threshold=0.7)
    assert len(out) == 2
