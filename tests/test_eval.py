from __future__ import annotations

from pathlib import Path

from guji.eval import (
    QuestionResult,
    _keyword_hit,
    _title_matches_gold,
    eval_dict_recall,
    eval_recall,
    load_questions,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"


def test_keyword_hit_requires_matching_title_and_keyword():
    meta = {"title": "論語", "text_raw": "顏淵問仁。子曰：克己復禮為仁。"}
    assert _keyword_hit(meta, ["論語"], ["克己復禮"])
    assert not _keyword_hit(meta, ["孟子"], ["克己復禮"])  # wrong book
    assert not _keyword_hit(meta, ["論語"], ["性善"])       # keyword absent


def test_title_matches_gold_exact_and_fuzzy():
    gold = {"说文解字"}  # simplified, as to_norm would produce
    assert _title_matches_gold("說文解字", gold)   # traditional -> normalizes to match
    assert _title_matches_gold("說文", gold)        # abbreviation, substring fallback
    assert not _title_matches_gold("康熙字典", gold)


def test_summarize_aggregates_by_category_and_overall():
    results = [
        QuestionResult("a", "字词训诂", "q1", ["x"], ["y"], recall_hit=True, citation_ok=True),
        QuestionResult("b", "字词训诂", "q2", ["x"], ["y"], recall_hit=False, citation_ok=False),
        QuestionResult("c", "人物事件", "q3", ["x"], ["y"], recall_hit=True, citation_ok=None),
    ]
    s = summarize(results)
    assert s["by_category"]["字词训诂"]["n"] == 2
    assert s["by_category"]["字词训诂"]["recall_at_k"] == 0.5
    assert s["by_category"]["字词训诂"]["citation_accuracy"] == 0.5
    # citation accuracy is None when no answers were evaluated for that category
    assert s["by_category"]["人物事件"]["citation_accuracy"] is None
    assert s["overall"]["n"] == 3
    assert abs(s["overall"]["recall_at_k"] - 2 / 3) < 1e-3  # summarize() rounds to 4dp
    assert s["overall"]["citation_n"] == 2  # only a,b had citation_ok set


def test_summarize_handles_zero_questions():
    s = summarize([])
    assert s["overall"]["n"] == 0
    assert s["overall"]["recall_at_k"] is None


def test_question_set_matches_spec_category_counts():
    qs = load_questions(QUESTIONS_PATH)
    assert len(qs) == 80
    counts: dict[str, int] = {}
    for q in qs:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
        assert q["gold_books"] and q["gold_keywords"]
        assert isinstance(q["gold_books"], list) and isinstance(q["gold_keywords"], list)
    assert counts == {"字词训诂": 20, "人物事件": 30, "典故出处": 20, "跨书比较": 10}


def test_question_ids_are_unique():
    qs = load_questions(QUESTIONS_PATH)
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))


class _FakeConfig:
    def __init__(self, dict_db_path):
        self.dict_db_path = dict_db_path
        self.pua_map_path = ""


def _make_dict_db(tmp_path):
    from guji.parse import dictionary

    db = tmp_path / "dict.sqlite"
    conn = dictionary.create_db(db)
    conn.execute(
        "INSERT INTO char_entry(headword,headword_norm,source_book,body_raw,body_norm,section)"
        " VALUES (?,?,?,?,?,?)",
        ("敝", "敝", "說文解字", "帗也。", "帗也。", ""),
    )
    conn.commit()
    conn.close()
    return db


def test_eval_dict_recall_hits_real_lookup(tmp_path):
    cfg = _FakeConfig(_make_dict_db(tmp_path))
    q = {"id": "zi01", "category": "字词训诂", "question": "?", "gold_books": ["說文解字"], "gold_keywords": ["敝"]}
    qr = eval_dict_recall(cfg, q)
    assert qr.recall_hit is True
    assert qr.recall_hit_rank == 0


def test_eval_dict_recall_misses_wrong_book(tmp_path):
    cfg = _FakeConfig(_make_dict_db(tmp_path))
    q = {"id": "zi02", "category": "字词训诂", "question": "?", "gold_books": ["康熙字典"], "gold_keywords": ["敝"]}
    qr = eval_dict_recall(cfg, q)
    assert qr.recall_hit is False
    assert qr.recall_hit_rank is None


def test_eval_recall_routes_dict_category_to_lookup_char(tmp_path, monkeypatch):
    # eval_recall must dispatch 字词训诂 to eval_dict_recall, never to hybrid.search
    cfg = _FakeConfig(_make_dict_db(tmp_path))
    q = {"id": "zi03", "category": "字词训诂", "question": "?", "gold_books": ["說文解字"], "gold_keywords": ["敝"]}

    def boom(*a, **kw):
        raise AssertionError("hybrid.search must not be called for 字词训诂")

    monkeypatch.setattr("guji.eval.hybrid.search", boom)
    qr = eval_recall(cfg, q, top_k=20)
    assert qr.recall_hit is True
