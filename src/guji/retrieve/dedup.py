"""Collapse near-duplicate hits from related books (§3.3).

史記 and 史記三家注 (and 詩經 / 詩經_symbolic) share text; when both surface for the
same passage, keep only the higher-ranked one. Comparison is limited to books that
are ``related_to`` each other, so it is cheap and cannot merge genuinely distinct
passages that merely share stock phrases.
"""

from __future__ import annotations

import json
from pathlib import Path


def char_bigrams(s: str) -> set[str]:
    s = "".join(ch for ch in s if not ch.isspace())
    return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def load_related_map(manifest_path: Path) -> dict[str, set[str]]:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {b["book_id"]: set(b.get("related_to", [])) for b in m["books"]}


def _related(a: str, b: str, rel: dict[str, set[str]]) -> bool:
    return b in rel.get(a, set()) or a in rel.get(b, set())


def collapse(hits: list, related: dict[str, set[str]], threshold: float) -> list:
    """Drop later hits that duplicate an earlier kept hit. Two triggers:

    * **parallel edition** — related books covering the same 卷/篇 (e.g. 史記 and
      史記三家注 both 卷一‧五帝本紀). Text-similarity fails here because 三家注 is
      dominated by 集解/索隱/正義 commentary, so we key on (related + same juan).
    * **text near-dup** — same or related book with char-bigram Jaccard ≥ threshold.

    Same-book, same-juan but *different* passages (e.g. overlapping neighbour chunks)
    are kept: the juan trigger fires only across *different* related books.
    """
    kept: list = []
    kept_info: list[tuple[str, str, set[str]]] = []  # (book_id, juan, bigrams)
    for h in hits:
        book = h.meta.get("book_id", "")
        juan = (h.meta.get("juan") or "").strip()
        grams = char_bigrams(h.meta.get("text_raw", ""))
        dup = False
        for kbook, kjuan, kgrams in kept_info:
            if not (book == kbook or _related(book, kbook, related)):
                continue
            if book != kbook and juan and juan == kjuan:
                dup = True
                break
            if jaccard(grams, kgrams) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(h)
            kept_info.append((book, juan, grams))
    return kept
