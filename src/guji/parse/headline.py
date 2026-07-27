"""Shared parsing of the 5-line comment block + book head line (§3.4).

The head line is tolerant of every observed shape:
  * ``# 書名〔朝代〕作者``            (most files)
  * ``## 書名 〔清〕...``            (唐詩三百首, 荀子, ... use level-2 as head)
  * multiple ``〔〕`` groups         (說文解字)
  * ``(前半部分)`` shard suffix      (康熙字典_p1/p2)
  * no head line at all             (千字文, 般若波羅蜜多心經) -> title from filename

Comment lines look like ``#中華經典古籍精校`` — a ``#`` immediately followed by a
non-space. Head lines always have whitespace after the hashes, which is how we
tell them apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_HEAD_RE = re.compile(r"^(#{1,2})[ \t]+(\S.*)$")
_BRACKET_RE = re.compile(r"〔([^〕]+)〕")
_SHARD_RE = re.compile(r"[（(](?:前半部分|後半部分|前半部|後半部)[)）]\s*$")
# role words that trail an author name and should be trimmed off
_ROLE_RE = re.compile(r"[ \t　]*(?:撰|注|編撰|編|著|譯|次韻|重修|集解|索隱|正義)$")


@dataclass
class HeadInfo:
    title: str
    dynasty: str = ""
    author: str = ""
    head_line_index: int | None = None       # index into content lines
    head_is_level2: bool = False
    field_extras: dict = field(default_factory=dict)


def clean_stem(stem: str) -> str:
    """Filename stem -> canonical book title fragment.

    Drops shard suffixes so 康熙字典_p1 / _p2 collapse to one book.
    """
    return re.sub(r"_p\d+$", "", stem)


def _strip_brackets(s: str) -> str:
    return _BRACKET_RE.sub("", s)


def _title_part(head_text: str) -> str:
    """The book-name portion of a head line, before any 〔〕/shard suffix."""
    t = _SHARD_RE.sub("", head_text)
    m = _BRACKET_RE.search(t)
    if m:
        t = t[: m.start()]
    return t.strip()


def parse_meta(head_text: str) -> tuple[str, str]:
    """Extract (dynasty, author) from a head-line-like string."""
    head_text = _SHARD_RE.sub("", head_text)  # drop '(前半部分)' etc. before parsing
    brackets = _BRACKET_RE.findall(head_text)
    dynasty = brackets[0].strip() if brackets else ""
    author = ""
    m = _BRACKET_RE.search(head_text)
    if m:
        rest = head_text[m.end():]
        nxt = _BRACKET_RE.search(rest)
        author = (rest[: nxt.start()] if nxt else rest).strip()
        author = _ROLE_RE.sub("", author).strip()
    return dynasty, author


def read_content_lines(path: Path) -> list[str]:
    """File lines with the leading comment block + following blanks removed."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    i = 0
    # skip the 5-line (or so) comment block: lines starting with '#' + non-space
    while i < len(lines) and re.match(r"^#\S", lines[i]):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return lines[i:]


def find_head(content_lines: list[str], stem: str) -> HeadInfo:
    """Locate the book head line and derive title/dynasty/author.

    Title authority is the *filename stem* (always clean); the head line only
    contributes dynasty/author. A level-2 line is accepted as the head only
    when its name matches the stem, so ``## 序`` (南齊書) is NOT mistaken for a
    title.
    """
    title = clean_stem(stem)
    head_idx: int | None = None
    head_text = ""
    is_l2 = False

    level1_idx = None
    level2_candidate = None
    for idx, line in enumerate(content_lines):
        m = _HEAD_RE.match(line)
        if not m:
            continue
        hashes, text = m.group(1), m.group(2).strip()
        if hashes == "#" and level1_idx is None:
            level1_idx = idx
            head_text = text
            break
        if hashes == "##" and level2_candidate is None:
            name = _title_part(text)
            if name and (name in title or title in name):
                level2_candidate = (idx, text)

    if level1_idx is not None:
        head_idx = level1_idx
    elif level2_candidate is not None:
        head_idx, head_text = level2_candidate
        is_l2 = True

    dynasty, author = ("", "")
    if head_text:
        dynasty, author = parse_meta(head_text)

    # Fallback: some files have no head line but carry a 〔朝代〕作者 line early on
    # (e.g. 千字文's "〔梁〕...周興嗣", 般若心經's "○ ... 〔唐〕玄奘 譯").
    if not dynasty:
        for line in content_lines[:8]:
            if _BRACKET_RE.search(line):
                dynasty, author = parse_meta(line)
                break

    return HeadInfo(
        title=title,
        dynasty=dynasty,
        author=author,
        head_line_index=head_idx,
        head_is_level2=is_l2,
    )
