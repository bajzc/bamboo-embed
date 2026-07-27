"""Narrative parser (form A, §3.4 / §5.4).

``## 卷/篇`` is a hard boundary — chunks never cross it. Within a juan, blank-
line-separated paragraphs are merged to 250–400 chars with a 1-paragraph
overlap. Output rows go into ``passages.jsonl``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..models import Book, Passage
from ..normalize import to_norm, to_raw
from .headline import find_head, read_content_lines

_HEAD2_RE = re.compile(r"^##[ \t]+(\S.*)$")


def _juans(content_lines: list[str], head_idx: int | None) -> list[tuple[str, list[str]]]:
    """Split post-head content into (juan_title, paragraphs)."""
    start = (head_idx + 1) if head_idx is not None else 0
    juans: list[tuple[str, list[str]]] = []
    cur_title = ""
    cur_paras: list[str] = []
    buf: list[str] = []

    def flush_para():
        if buf:
            cur_paras.append("".join(buf))
            buf.clear()

    def flush_juan():
        flush_para()
        if cur_paras:
            juans.append((cur_title, list(cur_paras)))
        cur_paras.clear()

    for line in content_lines[start:]:
        m = _HEAD2_RE.match(line)
        if m:
            flush_juan()
            cur_title = m.group(1).strip()
            continue
        if line.strip():
            buf.append(line.strip())
        else:
            flush_para()
    flush_juan()
    return juans


def merge_paragraphs(
    para_lens: list[int], lo: int, hi: int, overlap: int
) -> list[tuple[int, int]]:
    """Group paragraph indices into (start, end) inclusive chunks."""
    chunks: list[tuple[int, int]] = []
    n = len(para_lens)
    i = 0
    while i < n:
        size = 0
        j = i
        while j < n:
            if size >= lo and size + para_lens[j] > hi:
                break
            size += para_lens[j]
            j += 1
        if j == i:  # single oversized paragraph
            j = i + 1
        chunks.append((i, j - 1))
        if j >= n:
            break
        step = j - overlap
        i = step if step > i else i + 1
    return chunks


def parse_book(
    book: Book,
    raw_root: Path,
    pua_map_path: str | None,
    lo: int,
    hi: int,
    overlap: int,
) -> Iterator[Passage]:
    src = raw_root / book.source_files[0]
    content = read_content_lines(src)
    head = find_head(content, Path(src.stem).name)
    juans = _juans(content, head.head_line_index)

    emitted: list[Passage] = []
    for juan_idx, (juan_title, paras) in enumerate(juans, start=1):
        raw_paras = [to_raw(p, pua_map_path) for p in paras]
        para_lens = [len(p) for p in raw_paras]
        for seq, (a, b) in enumerate(merge_paragraphs(para_lens, lo, hi, overlap), start=1):
            raw = "\n".join(raw_paras[a : b + 1])
            norm = to_norm("\n".join(paras[a : b + 1]), pua_map_path)
            prefix = f"《{book.title}》{juan_title}：" if juan_title else f"《{book.title}》"
            emitted.append(
                Passage(
                    chunk_id=f"{book.book_id}/juan{juan_idx:03d}/p{seq:04d}",
                    book_id=book.book_id,
                    title=book.title,
                    dynasty=book.dynasty,
                    author=book.author,
                    category=book.category,
                    juan=juan_title,
                    juan_idx=juan_idx,
                    para_start=a + 1,
                    para_end=b + 1,
                    text_raw=raw,
                    text_norm=norm,
                    text_for_embed=prefix + norm,
                    char_count=len(raw),
                )
            )

    _link(emitted)
    yield from emitted


def _link(passages: list[Passage]) -> None:
    for idx, p in enumerate(passages):
        p.prev_id = passages[idx - 1].chunk_id if idx > 0 else None
        p.next_id = passages[idx + 1].chunk_id if idx < len(passages) - 1 else None
