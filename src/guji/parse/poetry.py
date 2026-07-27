"""Poetry parser (form C, §3.4 / §5.4).

Verse collections use ``## 体裁`` groups and ``○ 詩題 / 作者`` poem markers; one
poem = one chunk, never split. 蒙學 files that are prose/proverbs (no ``○``)
fall back to the narrative paragraph chunker so they still land in passages.jsonl.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..models import Book, Passage
from ..normalize import to_norm, to_raw
from . import narrative
from .headline import find_head, read_content_lines

_HEAD2_RE = re.compile(r"^##[ \t]+(\S.*)$")
_POEM_RE = re.compile(r"^○[ \t]*(.*)$")


def _split_poem_title(raw_title: str) -> tuple[str, str]:
    """'感遇 / 張九齡' -> ('感遇', '張九齡'); no slash -> (title, '')."""
    if "/" in raw_title:
        t, a = raw_title.split("/", 1)
        return t.strip(), a.strip()
    return raw_title.strip(), ""


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
    start = (head.head_line_index + 1) if head.head_line_index is not None else 0
    body = content[start:]

    if not any(_POEM_RE.match(ln) for ln in body):
        # no ○ markers -> prose 蒙學 (千字文, 三字經, 弟子規, ...): reuse narrative chunker
        yield from narrative.parse_book(book, raw_root, pua_map_path, lo, hi, overlap)
        return

    poems: list[tuple[str, str, str, list[str]]] = []  # (group, title, author, verses)
    group = ""
    cur_title: str | None = None
    cur_author = ""
    verses: list[str] = []

    def flush():
        if cur_title is not None and verses:
            poems.append((group, cur_title, cur_author, list(verses)))

    for ln in body:
        m2 = _HEAD2_RE.match(ln)
        if m2:
            flush()
            cur_title = None
            verses.clear()
            group = m2.group(1).strip()
            continue
        mp = _POEM_RE.match(ln)
        if mp:
            flush()
            title, author = _split_poem_title(mp.group(1))
            cur_title, cur_author = title, author
            verses.clear()
            continue
        if cur_title is not None and ln.strip():
            verses.append(ln.strip())
    flush()

    emitted: list[Passage] = []
    for idx, (grp, title, poem_author, vlines) in enumerate(poems, start=1):
        juan = f"{grp}·{title}" if grp else title
        raw = to_raw("\n".join(vlines), pua_map_path)
        norm = to_norm("\n".join(vlines), pua_map_path)
        emitted.append(
            Passage(
                chunk_id=f"{book.book_id}/juan{idx:03d}/p0001",
                book_id=book.book_id,
                title=book.title,
                dynasty=book.dynasty,
                author=poem_author or book.author,
                category=book.category,
                juan=juan,
                juan_idx=idx,
                para_start=1,
                para_end=1,
                text_raw=raw,
                text_norm=norm,
                text_for_embed=f"《{book.title}》{juan}：{norm}",
                char_count=len(raw),
            )
        )

    narrative._link(emitted)
    yield from emitted
