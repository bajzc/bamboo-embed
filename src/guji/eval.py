"""Phase 5 evaluation (spec §7): Recall@K (retrieval) + citation accuracy (generation).

Every question in ``eval/questions.jsonl`` carries a ``gold_books`` whitelist and
``gold_keywords`` — both verified against the actual corpus when the question set was
authored, since the spec requires questions to only reference books that exist in the
manifest (no 左傳/資治通鑑/紅樓夢 etc.).

Recall@K: did any of the top-K fused/reranked hits come from a gold book and contain
a gold keyword? This is a proxy for exact chunk-level gold labels, which would be
impractical to hand-annotate for 80 questions across a 27M-character corpus.

**字词训诂 is evaluated differently**: dictionaries are deliberately excluded from the
vector store (§4 — 字書 never enters LanceDB), so ``hybrid.search`` structurally cannot
retrieve them and would always score 0% regardless of system quality. For this category,
"recall" instead means: does ``lookup_char`` return an entry from a gold book for the
question's gold keyword (the character itself)? This costs nothing (local SQLite only,
no LLM/rerank) and measures the actually-relevant retrieval path.

Citation accuracy: does ``guji ask``'s final, validated answer cite at least one gold
book? (An answer that fails citation/quote validation, or the retrieval threshold gate,
counts as a miss — a refusal is never a "correct" citation.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .normalize import to_norm
from .retrieve import hybrid

DICT_CATEGORY = "字词训诂"


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    gold_books: list[str]
    gold_keywords: list[str]
    recall_hit: bool = False
    recall_hit_rank: int | None = None
    citation_ok: bool | None = None      # None => citation accuracy wasn't evaluated
    answer_text: str | None = None
    rejected: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "category": self.category, "question": self.question,
            "gold_books": self.gold_books, "gold_keywords": self.gold_keywords,
            "recall_hit": self.recall_hit, "recall_hit_rank": self.recall_hit_rank,
            "citation_ok": self.citation_ok, "answer_text": self.answer_text,
            "rejected": self.rejected,
        }


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    qs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return qs[:limit] if limit else qs


def _keyword_hit(meta: dict, gold_books: list[str], gold_keywords: list[str]) -> bool:
    if meta.get("title") not in gold_books:
        return False
    text = meta.get("text_raw", "")
    return any(kw in text for kw in gold_keywords)


def eval_dict_recall(cfg: Config, q: dict) -> QuestionResult:
    """字词训诂 recall: does lookup_char find the keyword in a gold dictionary?

    No LLM/rerank involved — dict.sqlite is queried directly, same as the
    lookup_char tool itself, so this is free and measures the real path.
    """
    from . import tools

    hit = False
    for kw in q["gold_keywords"]:
        rows = tools.lookup_char(cfg, kw)
        if any(r["source_book"] in q["gold_books"] for r in rows):
            hit = True
            break
    return QuestionResult(
        id=q["id"], category=q["category"], question=q["question"],
        gold_books=q["gold_books"], gold_keywords=q["gold_keywords"],
        recall_hit=hit, recall_hit_rank=0 if hit else None,
    )


def eval_recall(
    cfg: Config, q: dict, top_k: int, use_rerank: bool = True, use_hyde: bool | None = None,
) -> QuestionResult:
    if q["category"] == DICT_CATEGORY:
        return eval_dict_recall(cfg, q)
    res = hybrid.search(cfg, q["question"], top_k=top_k, use_rerank=use_rerank, use_hyde=use_hyde)
    hit, hit_rank = False, None
    for i, h in enumerate(res.hits):
        if _keyword_hit(h.meta, q["gold_books"], q["gold_keywords"]):
            hit, hit_rank = True, i
            break
    return QuestionResult(
        id=q["id"], category=q["category"], question=q["question"],
        gold_books=q["gold_books"], gold_keywords=q["gold_keywords"],
        recall_hit=hit, recall_hit_rank=hit_rank,
    )


def _title_matches_gold(title: str, gold_norm: set[str]) -> bool:
    nt = to_norm(title)
    return nt in gold_norm or any(nt in g or g in nt for g in gold_norm)


def eval_citation(
    cfg: Config, q: dict, qr: QuestionResult, use_rerank: bool = True, use_hyde: bool | None = None,
) -> None:
    from . import generate

    ans = generate.answer(cfg, q["question"], use_rerank=use_rerank, use_hyde=use_hyde)
    qr.answer_text = ans.text
    qr.rejected = ans.rejected_by_threshold or ans.citation_failure
    if qr.rejected:
        qr.citation_ok = False
        return
    gold_norm = {to_norm(b) for b in q["gold_books"]}
    cited_titles = {t for t, _ in ans.citations}
    qr.citation_ok = any(_title_matches_gold(t, gold_norm) for t in cited_titles)


def run(
    cfg: Config,
    questions: list[dict],
    top_k: int = 20,
    with_citation: bool = True,
    use_rerank: bool = True,
    use_hyde: bool | None = None,
) -> list[QuestionResult]:
    results = []
    for q in questions:
        qr = eval_recall(cfg, q, top_k=top_k, use_rerank=use_rerank, use_hyde=use_hyde)
        if with_citation:
            eval_citation(cfg, q, qr, use_rerank=use_rerank, use_hyde=use_hyde)
        results.append(qr)
    return results


def summarize(results: list[QuestionResult]) -> dict:
    by_cat: dict[str, dict] = {}
    for r in results:
        c = by_cat.setdefault(r.category, {"n": 0, "recall_hits": 0, "citation_n": 0, "citation_ok": 0})
        c["n"] += 1
        c["recall_hits"] += int(r.recall_hit)
        if r.citation_ok is not None:
            c["citation_n"] += 1
            c["citation_ok"] += int(r.citation_ok)

    def rate(hits: int, n: int) -> float | None:
        return round(hits / n, 4) if n else None

    for c in by_cat.values():
        c["recall_at_k"] = rate(c["recall_hits"], c["n"])
        c["citation_accuracy"] = rate(c["citation_ok"], c["citation_n"])

    overall_n = len(results)
    overall_recall = sum(r.recall_hit for r in results)
    overall_cn = sum(1 for r in results if r.citation_ok is not None)
    overall_cok = sum(1 for r in results if r.citation_ok)
    return {
        "by_category": by_cat,
        "overall": {
            "n": overall_n, "recall_hits": overall_recall,
            "recall_at_k": rate(overall_recall, overall_n),
            "citation_n": overall_cn, "citation_ok": overall_cok,
            "citation_accuracy": rate(overall_cok, overall_cn),
        },
    }


def write_report(cfg: Config, label: str, results: list[QuestionResult], summary: dict, config_snapshot: dict) -> Path:
    cfg.eval_report_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.eval_report_dir / f"{label}.json"
    payload = {
        "label": label, "config": config_snapshot, "summary": summary,
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
