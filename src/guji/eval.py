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
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .normalize import to_norm
from .retrieve import hybrid

DICT_CATEGORY = "字词训诂"

# Judge prompt + sampling params are fixed on purpose (§Step 0.3 of the local-model
# fidelity plan): changing them makes historical eval labels non-comparable. Do not
# edit without also invalidating every prior --judge run.
_JUDGE_PROMPT = """你是文言文問答質量的盲評裁判。以下是一道問題、若干檢索到的古籍原文段落、
以及某個模型給出的回答。請只評估一項：回答對原文的釋義是否準確、忠實於所引原文，
不要考慮文采、格式，也不要考慮引用是否完整（格式已由其他機制檢查）。

問題：
{question}

檢索到的原文段落：
{passages}

模型回答：
{answer}

請只輸出一個 1-5 的整數評分，不要輸出任何其他文字：
1 = 嚴重曲解或杜撰原文含義
2 = 有明顯偏差
3 = 大致準確但有瑕疵
4 = 準確，僅有極小瑕疵
5 = 完全準確且忠於原文
"""
_JUDGE_MODEL_PROFILE = "cloud"  # the judge is always qwen-max via the cloud provider, regardless of profile under test


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
    rejected_by_threshold: bool = False
    citation_failure: bool = False
    attempts: int = 0
    prompt_tokens_max: int = 0
    quote_total: int = 0
    quote_verbatim_ok: int = 0
    violation_reasons: list[str] = field(default_factory=list)  # from the first attempt only
    judge_score: int | None = None
    error: str | None = None  # set if recall or citation raised — question counted as a miss, run continues

    def to_dict(self) -> dict:
        return {
            "id": self.id, "category": self.category, "question": self.question,
            "gold_books": self.gold_books, "gold_keywords": self.gold_keywords,
            "recall_hit": self.recall_hit, "recall_hit_rank": self.recall_hit_rank,
            "citation_ok": self.citation_ok, "answer_text": self.answer_text,
            "rejected": self.rejected, "error": self.error,
            "rejected_by_threshold": self.rejected_by_threshold,
            "citation_failure": self.citation_failure,
            "attempts": self.attempts, "prompt_tokens_max": self.prompt_tokens_max,
            "quote_total": self.quote_total, "quote_verbatim_ok": self.quote_verbatim_ok,
            "violation_reasons": self.violation_reasons, "judge_score": self.judge_score,
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
) -> tuple[QuestionResult, hybrid.SearchResult | None]:
    """Returns ``(result, primary)`` — ``primary`` is the raw ``hybrid.search`` result
    (``None`` for 字词训诂, which never touches it), handed back so a subsequent
    ``eval_citation`` call can reuse it via ``generate.answer(..., precomputed=primary)``
    instead of re-running the identical dense+sparse+rerank search.
    """
    if q["category"] == DICT_CATEGORY:
        return eval_dict_recall(cfg, q), None
    res = hybrid.search(cfg, q["question"], top_k=top_k, use_rerank=use_rerank, use_hyde=use_hyde)
    hit, hit_rank = False, None
    for i, h in enumerate(res.hits):
        if _keyword_hit(h.meta, q["gold_books"], q["gold_keywords"]):
            hit, hit_rank = True, i
            break
    qr = QuestionResult(
        id=q["id"], category=q["category"], question=q["question"],
        gold_books=q["gold_books"], gold_keywords=q["gold_keywords"],
        recall_hit=hit, recall_hit_rank=hit_rank,
    )
    return qr, res


def _title_matches_gold(title: str, gold_norm: set[str]) -> bool:
    nt = to_norm(title)
    return nt in gold_norm or any(nt in g or g in nt for g in gold_norm)


def _violation_reasons(attempt_violations: list[tuple[list, list]]) -> list[str]:
    """Reason tags from the *first* attempt only — retries have no diagnostic value."""
    if not attempt_violations:
        return []
    bad_citations, bad_quotes = attempt_violations[0]
    return [reason for _, _, reason in bad_citations] + ["bad_quote"] * len(bad_quotes)


def judge_answer(cfg: Config, question: str, passages: list[str], answer_text: str) -> int | None:
    """Blind qwen-max quality judge: 1-5, how faithful is the paraphrase to the cited text.

    Always uses the cloud provider regardless of which profile is under test — the
    judge itself is not the thing being A/B'd. Returns None if the judge call fails
    or its output can't be parsed (never raises: a missing judge score must not sink
    an otherwise-valid eval run).
    """
    from openai import OpenAI

    llm = cfg.providers[_JUDGE_MODEL_PROFILE].llm
    client = OpenAI(base_url=llm.base_url, api_key=cfg.api_key(llm.api_key_env) or "not-needed")
    passages_text = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)) or "（無）"
    prompt = _JUDGE_PROMPT.format(question=question, passages=passages_text, answer=answer_text)
    try:
        resp = client.chat.completions.create(
            model=llm.model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001 — a judge failure shouldn't fail the whole eval run
        return None
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


def safe_recall(
    cfg: Config, q: dict, top_k: int, use_rerank: bool = True, use_hyde: bool | None = None,
) -> tuple[QuestionResult, hybrid.SearchResult | None]:
    """Same as ``eval_recall``, but a raised exception becomes a recorded miss instead
    of aborting the whole eval run — an 80-question run losing everything to one bad
    passage (e.g. a reranker batch-size limit) is worse than losing one data point.
    """
    try:
        return eval_recall(cfg, q, top_k=top_k, use_rerank=use_rerank, use_hyde=use_hyde)
    except Exception as e:  # noqa: BLE001 — see docstring
        qr = QuestionResult(
            id=q["id"], category=q["category"], question=q["question"],
            gold_books=q["gold_books"], gold_keywords=q["gold_keywords"], error=str(e),
        )
        return qr, None


def safe_citation(
    cfg: Config, q: dict, qr: QuestionResult, use_rerank: bool = True, use_hyde: bool | None = None,
    judge: bool = False, precomputed: hybrid.SearchResult | None = None,
) -> None:
    """Same as ``eval_citation``, but a raised exception is recorded on ``qr`` (as a
    miss) instead of aborting the whole eval run."""
    try:
        eval_citation(cfg, q, qr, use_rerank=use_rerank, use_hyde=use_hyde, judge=judge, precomputed=precomputed)
    except Exception as e:  # noqa: BLE001 — see docstring
        qr.error = str(e)
        qr.citation_ok = False
        qr.rejected = True


def eval_citation(
    cfg: Config, q: dict, qr: QuestionResult, use_rerank: bool = True, use_hyde: bool | None = None,
    judge: bool = False, precomputed: hybrid.SearchResult | None = None,
) -> None:
    from . import generate

    ans = generate.answer(
        cfg, q["question"], use_rerank=use_rerank, use_hyde=use_hyde, precomputed=precomputed,
    )
    qr.answer_text = ans.text
    qr.rejected = ans.rejected_by_threshold or ans.citation_failure
    qr.rejected_by_threshold = ans.rejected_by_threshold
    qr.citation_failure = ans.citation_failure
    qr.attempts = ans.attempts
    qr.prompt_tokens_max = ans.prompt_tokens_max
    qr.quote_total = ans.quote_stats.get("total", 0)
    qr.quote_verbatim_ok = ans.quote_stats.get("verbatim_ok", 0)
    qr.violation_reasons = _violation_reasons(ans.attempt_violations)
    if qr.rejected:
        qr.citation_ok = False
        return
    gold_norm = {to_norm(b) for b in q["gold_books"]}
    cited_titles = {t for t, _ in ans.citations}
    qr.citation_ok = any(_title_matches_gold(t, gold_norm) for t in cited_titles)
    if judge:
        passages = [r.get("text_raw", "") for r in ans.retrieved if r.get("text_raw")]
        qr.judge_score = judge_answer(cfg, q["question"], passages, ans.text)


def run(
    cfg: Config,
    questions: list[dict],
    top_k: int = 20,
    with_citation: bool = True,
    use_rerank: bool = True,
    use_hyde: bool | None = None,
    judge: bool = False,
) -> list[QuestionResult]:
    """Two phases, not interleaved per-question.

    ``procman`` treats the local reranker and LLM as mutually exclusive (16G-machine
    OOM mitigation, see ``LaunchCfg``) and swaps one out to load the other. Recall and
    citation used to run back-to-back per question, forcing a swap on *every* question
    (twice, in fact — citation's ``generate.answer`` re-ran the same search as recall,
    each needing the reranker before the LLM). Over 80 questions that's ~160 load/unload
    cycles, which is enough to reliably crash llama-server's Metal backend. Batching
    into "all recall, then all citation" (reusing recall's search result — see
    ``eval_recall``'s ``primary`` return) needs one swap for the whole run, not one per
    question.
    """
    results = []
    primaries: list[hybrid.SearchResult | None] = []
    for q in questions:
        qr, primary = safe_recall(cfg, q, top_k=top_k, use_rerank=use_rerank, use_hyde=use_hyde)
        results.append(qr)
        primaries.append(primary)
    if with_citation:
        for q, qr, primary in zip(questions, results, primaries):
            safe_citation(
                cfg, q, qr, use_rerank=use_rerank, use_hyde=use_hyde, judge=judge, precomputed=primary,
            )
    return results


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; no numpy dependency, fine at eval-set sizes (~80)."""
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))
    return s[idx]


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

    # everything below is generation-side diagnostics; only meaningful for questions
    # that actually went through eval_citation (attempts > 0 marks that they did).
    gen = [r for r in results if r.attempts > 0]
    n_gen = len(gen)
    first_pass_n = sum(1 for r in gen if r.attempts == 1 and not r.citation_failure and not r.rejected_by_threshold)
    quote_total = sum(r.quote_total for r in gen)
    quote_ok = sum(r.quote_verbatim_ok for r in gen)
    violation_counts: dict[str, int] = {}
    for r in gen:
        for reason in r.violation_reasons:
            violation_counts[reason] = violation_counts.get(reason, 0) + 1
    ptoks = [r.prompt_tokens_max for r in gen if r.prompt_tokens_max]
    judge_scores = [r.judge_score for r in gen if r.judge_score is not None]

    return {
        "by_category": by_cat,
        "overall": {
            "n": overall_n, "recall_hits": overall_recall,
            "recall_at_k": rate(overall_recall, overall_n),
            "citation_n": overall_cn, "citation_ok": overall_cok,
            "citation_accuracy": rate(overall_cok, overall_cn),
            "errors": sum(1 for r in results if r.error),
        },
        "generation": {
            "n": n_gen,
            "first_pass_rate": rate(first_pass_n, n_gen),
            "quote_verbatim_rate": rate(quote_ok, quote_total),
            "violation_counts": violation_counts,
            "avg_attempts": round(sum(r.attempts for r in gen) / n_gen, 3) if n_gen else None,
            "rejected_by_threshold_rate": rate(sum(r.rejected_by_threshold for r in gen), n_gen),
            "citation_failure_rate": rate(sum(r.citation_failure for r in gen), n_gen),
            "prompt_tokens_p50": _percentile(ptoks, 50),
            "prompt_tokens_p95": _percentile(ptoks, 95),
            "prompt_tokens_max": max(ptoks) if ptoks else None,
            "explain_score": round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None,
            "explain_score_n": len(judge_scores),
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
