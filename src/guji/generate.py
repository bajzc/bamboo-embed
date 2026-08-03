"""Citation-grounded generation with function calling (§4.2).

Three hard constraints enforced in code, not just prompted:

  1. Every ``《書名》卷/篇`` citation in the answer must reference a book (and,
     if given, a juan) that actually appears among this turn's retrieved
     passages. Unverifiable citation -> reject the answer, retry once.
  2. Every ``『…』``-wrapped span must be an exact (whitespace-normalized)
     substring of a retrieved passage's ``text_raw`` — verbatim, no
     paraphrase. Violation -> same reject-and-retry.
  3. If the primary search's rerank score is below threshold, generation is
     skipped entirely (``SearchResult.rejected_by_threshold``) — no LLM call.

The primary search doubles as the model's first "tool call": its hits are
seeded into the conversation as a real tool-result message before the model
gets to speak, so it can answer immediately or call the tools for more.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from . import procman
from .config import Config
from .normalize import to_norm
from .retrieve import hybrid
from .tools import TOOL_SCHEMAS, call_tool

# (event_kind, payload) — see generate.answer()'s docstring-adjacent comments below
# for the event kinds emitted. UI-agnostic on purpose: the CLI supplies the callback.
OnEvent = Callable[[str, dict], None]


def _noop_event(kind: str, payload: dict) -> None:
    pass

REFUSAL_NO_RETRIEVAL = "未檢索到相關記載"
REFUSAL_NO_CITATION = "根據目前檢索到的段落，無法給出可驗證引用來源的回答，請換一個問法或提供更多細節。"

_SYSTEM_PROMPT = """你是專精中國古代典籍的問答助手。你的核心原則是「可溯源」：
寧可回答「{no_hit}」，也絕不可以憑自身知識杜撰史實或出處。

規則（必須嚴格遵守，違反會被程式拒絕並要求重寫）：
1. 回答中每一個結論都必須附上結構化出處，格式為「《書名》卷X‧篇名」，且書名與卷/篇名
   必須是本輪已檢索到的段落中真實存在的（可通過 search_passages / get_context 取得）。
   不可引用未檢索到的書目或卷篇。
2. 凡是逐字引用古籍原文的地方，必須用「{qopen}」「{qclose}」包裹，且包裹的內容必須與
   檢索結果中的 text_raw 逐字一致，不可改寫、增刪、簡化。你自己的解釋說明不要放進
   引號內。
3. 若檢索結果不足以回答問題，直接回答「{no_hit}」，不要臆測。
4. 字詞訓詁類問題優先用 lookup_char 工具查字書，不要用 search_passages。
5. 檢索命中的段落可能是節選，若需要前後文以判斷語境，使用 get_context 工具。
6. 語言：你自己的解釋、分析、結論一律使用簡體中文書寫。但「{qopen}」「{qclose}」內的
   逐字引用、以及《書名》卷/篇名，必須保留檢索結果中的原始繁體用字，不可轉為簡體或
   改寫，即使你其餘的回答是簡體中文。
"""


def _system_prompt(cfg: Config) -> str:
    return _SYSTEM_PROMPT.format(
        no_hit=REFUSAL_NO_RETRIEVAL,
        qopen=cfg.generate.quote_open,
        qclose=cfg.generate.quote_close,
    )


@dataclass
class ToolCallLog:
    name: str
    arguments: dict
    result: list[dict]


@dataclass
class AnswerResult:
    text: str
    rejected_by_threshold: bool = False
    citation_failure: bool = False
    attempts: int = 1
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    hyde_text: str | None = None
    citations: list[tuple[str, str]] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # primary search hits, seeded into the turn
    messages: list[dict] = field(default_factory=list)  # full conversation sent to the composer LLM
    answer_attempts: list[str] = field(default_factory=list)  # raw text of every attempt, incl. rejected ones
    attempt_violations: list[tuple[list, list]] = field(default_factory=list)  # (bad_citations, bad_quotes) per attempt
    prompt_tokens_max: int = 0  # largest usage.prompt_tokens seen across all LLM calls this turn
    quote_stats: dict = field(default_factory=dict)  # {"total": n, "verbatim_ok": n} from the first attempt only


class _Registry:
    """Tracks every (title, juan, text_raw) seen this turn, across all tool calls."""

    def __init__(self) -> None:
        self.titles: set[str] = set()
        self.juans_by_title: dict[str, set[str]] = {}
        self.text_pool: list[str] = []

    def add(self, rows: list[dict]) -> None:
        for r in rows:
            # search_passages/get_context key the book as "title"; lookup_char (dict.sqlite)
            # keys it as "source_book" — both are citable book names.
            title = r.get("title") or r.get("source_book") or ""
            if not title:
                continue
            self.titles.add(title)
            juan = (r.get("juan") or "").strip()
            if juan:
                self.juans_by_title.setdefault(title, set()).add(juan)
            text = r.get("text_raw") or r.get("body_raw") or ""
            if text:
                self.text_pool.append(text)


# juan capture stops at whitespace, common CJK/ASCII punctuation, brackets, and quote
# marks — anything else (e.g. a bare "：" right after the title) would otherwise get
# swallowed into a bogus non-empty juan and trip a false juan_not_found rejection.
_CITATION_RE = re.compile(r"《([^》]+)》\s*([^\s，。；：！？、\)）「」『』《]*)")
# a genuine juan/篇 label contains one of these; ordinary prose glued on after 》
# (e.g. "的解释", "中对") does not, and must not be mistaken for a juan citation.
_JUAN_HINT_RE = re.compile(r"[卷篇‧·、0-9一二三四五六七八九十百千萬]")


def extract_citations(text: str) -> list[tuple[str, str]]:
    out = []
    for t, j in _CITATION_RE.findall(text):
        t, j = t.strip(), j.strip()
        if j and not _JUAN_HINT_RE.search(j):
            j = ""
        out.append((t, j))
    return out


def extract_quotes(text: str, qopen: str, qclose: str) -> list[str]:
    pattern = re.compile(re.escape(qopen) + r"(.*?)" + re.escape(qclose), re.S)
    return [m.strip() for m in pattern.findall(text)]


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


# juan/篇 labels aren't formatted consistently across the corpus — most books separate
# 卷/篇 with "‧", but some (e.g. 三國志) use full-width spaces and repeat the book title.
# Strip whitespace and common separator punctuation before comparing so the citation
# check only cares about the substantive 卷/篇 text, not which separator the model used.
# Models are also inconsistent about which separator glyph they use ("‧"/"・"/"·"/"•"/
# full-width space/none) and the citation-juan capture regex is greedy, so prose glued
# on with no punctuation (e.g. "卷三十五的記載，" or "第五中記載了") ends up inside the
# captured juan too. Chasing every separator variant is a losing game, so the primary
# check below matches on the "卷<number>" substring alone — that's the one part of a
# juan citation that's unambiguous and load-bearing; the 篇名 after it is treated as
# free text. _norm_juan() is kept only as a fallback for sources that cite by 篇/回 name
# without a 卷 number at all (where a normalized prefix match is the best we can do).
_JUAN_SEP_RE = re.compile(r"[\s‧・·、•]+")
_JUAN_NUM_RE = re.compile(r"卷[0-9一二三四五六七八九十百千萬]+")


def _norm_juan(s: str) -> str:
    return _JUAN_SEP_RE.sub("", s)


def _juan_num(s: str) -> str | None:
    m = _JUAN_NUM_RE.search(s)
    return m.group(0) if m else None


_URL_RE = re.compile(r"https?://[^\s\)\]）」』，。、]+")


def validate(text: str, registry: _Registry, qopen: str, qclose: str) -> tuple[list, list]:
    """Return (bad_citations, bad_quotes); both empty means the answer is grounded."""
    if REFUSAL_NO_RETRIEVAL in text:
        return [], []  # the model itself declined — nothing to ground

    citations = extract_citations(text)
    bad_citations: list[tuple[str, str, str]] = []
    if not citations:
        bad_citations.append(("", "", "no_citation"))
    # the corpus is plain classical text with no images or links; any URL (bare, or
    # inside markdown image syntax "![](url)") is fabricated by the model, never
    # grounded in a retrieved passage — same "can't be verified" failure as a bad
    # citation, so it goes through the same reject-and-retry path.
    for url in _URL_RE.findall(text):
        bad_citations.append(("", url, "hallucinated_link"))
    # compare book titles script-insensitively: the corpus is all-traditional, but a
    # model answering a simplified-script question may cite in simplified too.
    norm_titles = {to_norm(kt): kt for kt in registry.titles}
    for title, juan in citations:
        nt = to_norm(title)
        # exact/script-normalized match, else a common-abbreviation fallback
        # (《說文》 for 《說文解字》): substring either direction.
        canonical = norm_titles.get(nt) or next(
            (kt for nkt, kt in norm_titles.items() if nt in nkt or nkt in nt), None
        )
        if canonical is None:
            bad_citations.append((title, juan, "book_not_found"))
            continue
        if juan:
            known = registry.juans_by_title.get(canonical, set())
            # only flag a mismatch when we actually have juan ground-truth for this
            # book (dictionary sources never carry a juan, so there's nothing to
            # check them against — don't punish the model for that).
            cand_num = _juan_num(juan)
            nj = _norm_juan(juan)
            title_n = _norm_juan(canonical)
            ok = False
            for k in known:
                if not k:
                    continue
                # primary check: the "卷<number>" itself matches — the one part of the
                # citation that's unambiguous regardless of separator glyph or how much
                # trailing prose the greedy capture regex pulled in after it.
                if cand_num and cand_num == _juan_num(k):
                    ok = True
                    break
                # fallback for juan labels with no 卷 number (e.g. cited by 篇/回 name
                # only): normalized prefix match, tolerant of extra trailing junk on
                # either side and of books that repeat their own title inside juan.
                nk = _norm_juan(k)
                if nk.startswith(title_n):
                    nk = nk[len(title_n):]
                if nj and nk and (nk.startswith(nj) or nj.startswith(nk)):
                    ok = True
                    break
            if known and not ok:
                bad_citations.append((title, juan, "juan_not_found"))

    pool_norm = [_norm_ws(t) for t in registry.text_pool]
    bad_quotes = []
    for q in extract_quotes(text, qopen, qclose):
        qn = _norm_ws(q)
        if not qn or not any(qn in p for p in pool_norm):
            bad_quotes.append(q)

    return bad_citations, bad_quotes


def _correction_message(bad_citations: list, bad_quotes: list, registry: _Registry) -> str:
    lines = ["你上一輪的回答未通過校驗，請根據以下問題重新作答："]
    for title, juan, reason in bad_citations:
        if reason == "no_citation":
            lines.append("- 回答中沒有任何《書名》格式的結構化引用，必須至少引用一處。")
        elif reason == "book_not_found":
            lines.append(f"- 《{title}》不在本輪檢索結果中，禁止引用此書。")
        elif reason == "hallucinated_link":
            lines.append(f"- 回答中出現了連結「{juan}」，檢索到的古籍原文不含任何網址或圖片，禁止杜撰連結。")
        else:
            lines.append(f"- 《{title}》{juan} 的卷/篇名與檢索結果不符。")
    for q in bad_quotes:
        lines.append(f"- 引號內容「{q[:40]}」與檢索到的原文不完全一致，逐字引用必須完全相同。")
    titles = "、".join(sorted(registry.titles)) or "（無）"
    lines.append(f"本輪可引用的書目僅限：{titles}。請修正後重新輸出完整回答。")
    return "\n".join(lines)


def answer(
    cfg: Config,
    query: str,
    book: str | None = None,
    dynasty: str | None = None,
    category: str | None = None,
    use_hyde: bool | None = None,
    use_rerank: bool = True,
    validate_answer: bool | None = None,
    on_event: OnEvent | None = None,
    precomputed: hybrid.SearchResult | None = None,
) -> AnswerResult:
    """``precomputed`` lets a caller that already ran ``hybrid.search`` for this exact
    query (e.g. ``guji eval``, which needs the same primary search for Recall@K) skip
    doing it again here — same result, one retrieval call instead of two.
    """
    emit = on_event or _noop_event
    primary = precomputed if precomputed is not None else hybrid.search(
        cfg, query, book=book, dynasty=dynasty, category=category,
        use_hyde=use_hyde, use_rerank=use_rerank, on_event=on_event,
    )
    if primary.rejected_by_threshold or not primary.hits:
        return AnswerResult(text=REFUSAL_NO_RETRIEVAL, rejected_by_threshold=True,
                             hyde_text=primary.hyde_text)

    from openai import OpenAI

    llm = cfg.active_llm()
    client = OpenAI(base_url=llm.base_url, api_key=cfg.api_key(llm.api_key_env) or "not-needed")

    registry = _Registry()
    seed_rows = [
        {
            "chunk_id": h.chunk_id, "title": h.meta.get("title", ""),
            "dynasty": h.meta.get("dynasty", ""), "author": h.meta.get("author", ""),
            "category": h.meta.get("category", ""), "juan": h.meta.get("juan", ""),
            "text_raw": h.meta.get("text_raw", ""), "rerank_score": h.rerank_score,
        }
        for h in primary.hits
    ]
    registry.add(seed_rows)
    emit("retrieval", {"rows": seed_rows})

    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(cfg)},
        {"role": "user", "content": query},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "seed_search", "type": "function",
                "function": {"name": "search_passages", "arguments": json.dumps({"query": query})},
            }],
        },
        {"role": "tool", "tool_call_id": "seed_search", "content": json.dumps(seed_rows, ensure_ascii=False)},
    ]

    tool_calls_log: list[ToolCallLog] = []
    answer_attempts: list[str] = []
    attempt_violations: list[tuple[list, list]] = []

    do_validate = cfg.generate.validate_citations if validate_answer is None else validate_answer
    prompt_tokens_max = 0
    if not do_validate:
        # skip citation/quote/link validation and the reject-and-retry loop entirely —
        # whatever the model says on its first pass is returned as-is, ungrounded.
        emit("attempt_start", {"index": 1})
        final_text, ptoks = _converse(client, llm.model, messages, cfg, registry, tool_calls_log, on_event)
        # bad_citations/bad_quotes=None signals "validation skipped" to listeners,
        # distinct from an empty list (which would mean "validated and passed").
        emit("attempt_done", {"index": 1, "text": final_text, "bad_citations": None, "bad_quotes": None})
        return AnswerResult(
            text=final_text, attempts=1, tool_calls=tool_calls_log,
            hyde_text=primary.hyde_text,
            citations=extract_citations(final_text),
            quotes=extract_quotes(final_text, cfg.generate.quote_open, cfg.generate.quote_close),
            retrieved=seed_rows, messages=list(messages), answer_attempts=[final_text],
            prompt_tokens_max=ptoks,
        )

    max_retries = cfg.generate.retry_on_violation
    attempts = 0
    final_text = ""
    quote_stats: dict = {}

    while attempts <= max_retries:
        attempts += 1
        emit("attempt_start", {"index": attempts})
        final_text, ptoks = _converse(client, llm.model, messages, cfg, registry, tool_calls_log, on_event)
        prompt_tokens_max = max(prompt_tokens_max, ptoks)
        answer_attempts.append(final_text)
        bad_citations, bad_quotes = validate(final_text, registry, cfg.generate.quote_open, cfg.generate.quote_close)
        attempt_violations.append((bad_citations, bad_quotes))
        if attempts == 1:
            # diagnostic only, so always taken from the first attempt regardless of
            # whether it ultimately passes — retries would hide how much the model
            # drifted before correction.
            total_quotes = len(extract_quotes(final_text, cfg.generate.quote_open, cfg.generate.quote_close))
            quote_stats = {"total": total_quotes, "verbatim_ok": total_quotes - len(bad_quotes)}
        emit("attempt_done", {
            "index": attempts, "text": final_text,
            "bad_citations": bad_citations, "bad_quotes": bad_quotes,
        })
        if not bad_citations and not bad_quotes:
            return AnswerResult(
                text=final_text, attempts=attempts, tool_calls=tool_calls_log,
                hyde_text=primary.hyde_text,
                citations=extract_citations(final_text),
                quotes=extract_quotes(final_text, cfg.generate.quote_open, cfg.generate.quote_close),
                retrieved=seed_rows, messages=list(messages), answer_attempts=answer_attempts,
                attempt_violations=attempt_violations,
                prompt_tokens_max=prompt_tokens_max, quote_stats=quote_stats,
            )
        if attempts > max_retries:
            break
        emit("retry", {"index": attempts})
        messages.append({"role": "assistant", "content": final_text})
        messages.append({"role": "user", "content": _correction_message(bad_citations, bad_quotes, registry)})

    return AnswerResult(
        text=REFUSAL_NO_CITATION, citation_failure=True, attempts=attempts,
        tool_calls=tool_calls_log, hyde_text=primary.hyde_text,
        retrieved=seed_rows, messages=list(messages), answer_attempts=answer_attempts,
        attempt_violations=attempt_violations,
        prompt_tokens_max=prompt_tokens_max, quote_stats=quote_stats,
    )


def _converse(client, model: str, messages: list[dict], cfg: Config, registry: _Registry,
              tool_calls_log: list[ToolCallLog], on_event: OnEvent | None = None) -> tuple[str, int]:
    """Run tool-calling rounds until the model returns plain content.

    Streams each round so callers can render tokens as they're generated; tool-call
    id/name/arguments arrive fragmented across chunks and are reassembled by index
    before being dispatched (arguments are only valid JSON once the stream ends).

    Returns ``(content, prompt_tokens_max)`` — the latter is the largest
    ``usage.prompt_tokens`` seen across every round of this call, requested via
    ``stream_options={"include_usage": True}`` (the trailing usage-only chunk this
    produces has an empty ``choices`` list, hence the guard below).
    """
    emit = on_event or _noop_event
    prompt_tokens_max = 0
    for _ in range(cfg.generate.max_tool_rounds):
        procman.ensure_llm()
        stream = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS,
            tool_choice="auto", max_tokens=cfg.generate.max_tokens, stream=True,
            stream_options={"include_usage": True},
        )
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        for chunk in stream:
            if chunk.usage is not None:
                prompt_tokens_max = max(prompt_tokens_max, chunk.usage.prompt_tokens)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                emit("delta", {"text": delta.content})
            for tc in delta.tool_calls or []:
                acc = tool_calls_acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    acc["id"] = tc.id
                if tc.function and tc.function.name:
                    acc["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    acc["arguments"] += tc.function.arguments

        content = "".join(content_parts)
        if not tool_calls_acc:
            return content, prompt_tokens_max

        tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        messages.append({
            "role": "assistant", "content": content or "",
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            name, tc_id = tc["name"], tc["id"]
            args = json.loads(tc["arguments"] or "{}")
            emit("tool_call", {"name": name, "arguments": args})
            try:
                result = call_tool(cfg, name, args)
            except Exception as e:  # noqa: BLE001 — surface tool errors to the model, not a crash
                result = []
                messages.append({"role": "tool", "tool_call_id": tc_id,
                                  "content": json.dumps({"error": str(e)}, ensure_ascii=False)})
                tool_calls_log.append(ToolCallLog(name, args, result))
                emit("tool_result", {"name": name, "result": result, "error": str(e)})
                continue
            registry.add(result)
            tool_calls_log.append(ToolCallLog(name, args, result))
            messages.append({"role": "tool", "tool_call_id": tc_id,
                              "content": json.dumps(result, ensure_ascii=False)})
            emit("tool_result", {"name": name, "result": result})

    # ran out of rounds without a final answer
    return "", prompt_tokens_max
