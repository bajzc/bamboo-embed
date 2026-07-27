from __future__ import annotations

from guji.generate import (
    REFUSAL_NO_RETRIEVAL,
    _Registry,
    _correction_message,
    extract_citations,
    extract_quotes,
    validate,
)

QO, QC = "『", "』"


def _registry(rows):
    r = _Registry()
    r.add(rows)
    return r


def test_extract_citations_with_and_without_juan():
    text = "此語出自《論語》顏淵篇第十二，另見《孟子》。"
    cites = extract_citations(text)
    assert ("論語", "顏淵篇第十二") in cites
    assert ("孟子", "") in cites


def test_extract_citations_stops_before_colon_and_quote_marks():
    # a colon (fullwidth or not) or a quote-open mark right after 《書名》 must not be
    # swallowed into the juan capture
    text = f"《說文解字》：{QO}敝，帗也。{QC}"
    assert extract_citations(text) == [("說文解字", "")]


def test_extract_citations_ignores_prose_glued_after_title():
    # "的解释", "中对" etc. are ordinary sentence continuations, not juan labels
    assert extract_citations("根据《說文解字》的解释如下") == [("說文解字", "")]
    assert extract_citations("《說文》中对這個字有記載") == [("說文", "")]


def test_extract_citations_keeps_real_juan_label():
    assert extract_citations("見《論語》顏淵篇第十二") == [("論語", "顏淵篇第十二")]
    assert extract_citations("見《史記》卷一") == [("史記", "卷一")]


def test_extract_quotes():
    text = f"子曰：{QO}克己復禮為仁{QC}，此為仁之要義。"
    assert extract_quotes(text, QO, QC) == ["克己復禮為仁"]


def test_registry_add_skips_falsy_fields():
    r = _registry([{"title": "", "text_raw": "x"}, {"title": "論語", "juan": "顏淵篇", "text_raw": "y"}])
    assert "論語" in r.titles
    assert "" not in r.titles
    assert r.juans_by_title["論語"] == {"顏淵篇"}
    assert r.text_pool == ["y"]  # the falsy-title row's text is not registered as citable


def test_registry_add_accepts_lookup_char_shape():
    # lookup_char rows key the book as "source_book", not "title" — must still register
    r = _registry([{"source_book": "說文解字", "body_raw": "帗也。一曰敗衣。"}])
    assert "說文解字" in r.titles
    assert r.text_pool == ["帗也。一曰敗衣。"]


def test_validate_accepts_dict_lookup_citation():
    r = _registry([{"source_book": "說文解字", "body_raw": "敝，帗也。一曰敗衣。"}])
    text = f"《說文解字》：{QO}敝，帗也。一曰敗衣。{QC}"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert bad_c == [] and bad_q == []


def test_validate_accepts_common_book_abbreviation():
    # 說文 is the standard scholarly abbreviation of 說文解字 — must not be rejected
    r = _registry([{"source_book": "說文解字", "body_raw": "敝，帗也。"}])
    text = f"《說文》：{QO}敝，帗也。{QC}"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert bad_c == [] and bad_q == []


def test_validate_does_not_require_juan_for_dict_only_book():
    # dict entries never carry a juan; a juan-shaped citation must not be punished
    # when we have no ground truth to check it against
    r = _registry([{"source_book": "說文解字", "body_raw": "敝，帗也。"}])
    text = f"《說文解字》卷大徐本：{QO}敝，帗也。{QC}"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert bad_c == [] and bad_q == []


def test_validate_accepts_simplified_citation_of_traditional_book():
    # corpus/registry titles are traditional; a model answering a simplified-script
    # question may cite in simplified — must still resolve to the same book
    r = _registry([{"source_book": "說文解字", "body_raw": "敝，帗也。"}])
    text = f"《说文解字》：{QO}敝，帗也。{QC}"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert bad_c == [] and bad_q == []


def test_validate_still_rejects_juan_mismatch_when_ground_truth_exists():
    r = _registry([{"title": "論語", "juan": "顏淵篇第十二", "text_raw": "顏淵問仁。"}])
    text = "《論語》學而篇第一如是說。"
    bad_c, _ = validate(text, r, QO, QC)
    assert any(reason == "juan_not_found" for _, _, reason in bad_c)


def test_validate_accepts_grounded_answer():
    r = _registry([{"title": "論語", "juan": "顏淵篇第十二", "text_raw": "顏淵問仁。子曰：「克己復禮為仁。」"}])
    text = f"孔子認為{QO}克己復禮為仁{QC}（《論語》顏淵篇第十二）。"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert bad_c == [] and bad_q == []


def test_validate_rejects_unknown_book():
    r = _registry([{"title": "論語", "juan": "顏淵篇", "text_raw": "顏淵問仁。"}])
    text = "《左傳》記載了這件事。"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert any(reason == "book_not_found" for _, _, reason in bad_c)


def test_validate_rejects_wrong_juan():
    r = _registry([{"title": "論語", "juan": "顏淵篇第十二", "text_raw": "顏淵問仁。"}])
    text = "《論語》學而篇第一如是說。"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert any(reason == "juan_not_found" for _, _, reason in bad_c)


def test_validate_requires_at_least_one_citation():
    r = _registry([{"title": "論語", "juan": "顏淵篇", "text_raw": "顏淵問仁。"}])
    text = "這是一個沒有任何引用的回答。"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert any(reason == "no_citation" for _, _, reason in bad_c)


def test_validate_refusal_text_short_circuits():
    r = _registry([])
    bad_c, bad_q = validate(REFUSAL_NO_RETRIEVAL, r, QO, QC)
    assert bad_c == [] and bad_q == []


def test_validate_rejects_paraphrased_quote():
    r = _registry([{"title": "論語", "juan": "顏淵篇", "text_raw": "顏淵問仁。子曰：克己復禮為仁。"}])
    text = f"（《論語》顏淵篇）{QO}克己復禮就是仁{QC}"  # paraphrased, not verbatim
    bad_c, bad_q = validate(text, r, QO, QC)
    assert bad_q  # the altered quote should not match


def test_validate_quote_tolerant_of_whitespace():
    r = _registry([{"title": "論語", "juan": "顏淵篇", "text_raw": "顏淵問仁。\n子曰：克己復禮為仁。"}])
    text = f"（《論語》顏淵篇）{QO}顏淵問仁。子曰：克己復禮為仁。{QC}"
    bad_c, bad_q = validate(text, r, QO, QC)
    assert bad_q == []  # newline vs none should not fail verbatim check


def test_correction_message_mentions_allowed_titles():
    r = _registry([{"title": "論語", "juan": "顏淵篇", "text_raw": "x"}])
    msg = _correction_message([("左傳", "", "book_not_found")], [], r)
    assert "左傳" in msg and "論語" in msg
