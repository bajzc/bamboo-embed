"""Build ``data/manifest.json`` by scanning the raw corpus (§5.1).

Book identity rules:
  * one book per (category, cleaned-stem); ``康熙字典_p1`` + ``_p2`` merge into one.
  * title = filename stem (authoritative); dynasty/author parsed from head line.
  * empty files (8 詩詞 stubs) -> ``is_empty``; pointer files (詩經_symbolic) -> ``is_stub``.
  * related_to inferred from title containment (史記 / 史記三家注, 詩經 / 詩經_symbolic).
"""

from __future__ import annotations

import re
from pathlib import Path

from pypinyin import Style, lazy_pinyin

from ..models import Book, Manifest
from .headline import clean_stem, find_head, read_content_lines

SKIP_DIRS = {"9.工具腳本"}
POETRY_CATS = {"蒙學", "詩詞"}


def category_of(category_dir: str) -> str:
    """'j.史書' -> '史書', '0.詩詞' -> '詩詞'."""
    return category_dir.split(".", 1)[-1]


def form_for(category: str) -> str:
    if category == "字書訓詁":
        return "dictionary"
    if category in POETRY_CATS:
        return "poetry"
    return "narrative"


def slug(text: str) -> str:
    """Romanized, filesystem-safe book id (史記 -> 'shiji')."""
    parts = lazy_pinyin(text, style=Style.NORMAL, errors="default")
    s = re.sub(r"[^a-z0-9]", "", "".join(parts).lower())
    return s or "book"


def _iter_txt_files(raw_root: Path):
    for path in sorted(raw_root.rglob("*.txt")):
        rel = path.relative_to(raw_root)
        top = rel.parts[0]
        if top in SKIP_DIRS:
            continue
        yield path, rel, top


def _body_after_head(content_lines: list[str], head_idx: int | None) -> str:
    start = (head_idx + 1) if head_idx is not None else 0
    return "\n".join(content_lines[start:]).strip()


def build_manifest(raw_root: Path) -> Manifest:
    # group files by (category_dir, cleaned stem) so shards merge
    groups: dict[tuple[str, str], list[tuple[Path, Path]]] = {}
    for path, rel, top in _iter_txt_files(raw_root):
        key = (top, clean_stem(path.stem))
        groups.setdefault(key, []).append((path, rel))

    books: list[Book] = []
    for (category_dir, stem), members in sorted(groups.items()):
        members.sort(key=lambda m: m[1].as_posix())
        category = category_of(category_dir)
        source_files = [rel.as_posix() for _, rel in members]

        char_count = 0
        head = None
        body = ""
        for path, _ in members:
            char_count += len(path.read_text(encoding="utf-8"))
            if head is None:
                content = read_content_lines(path)
                head = find_head(content, stem)
                body = _body_after_head(content, head.head_line_index)

        assert head is not None
        is_empty = char_count == 0 or not body
        is_stub = bool(re.match(r"^內容見", body)) or ("內容見" in body and len(body) < 40)

        notes = ""
        if is_empty:
            notes = "empty file (no content)"
        elif is_stub:
            notes = f"pointer stub -> {body[:60]}"

        books.append(
            Book(
                book_id=slug(stem),
                title=head.title,
                dynasty=head.dynasty,
                author=head.author,
                category=category,
                category_dir=category_dir,
                source_files=source_files,
                char_count=char_count,
                form=form_for(category),
                is_empty=is_empty,
                is_complete=not (is_empty or is_stub),
                is_stub=is_stub,
                notes=notes,
            )
        )

    _link_related(books)
    # de-duplicate accidental id collisions deterministically
    _dedupe_ids(books)
    return Manifest(books=books)


def _link_related(books: list[Book]) -> None:
    for i, a in enumerate(books):
        for j, b in enumerate(books):
            if i == j:
                continue
            ta, tb = a.title, b.title
            if len(ta) >= 2 and len(tb) >= 2 and (ta in tb or tb in ta):
                if b.book_id not in a.related_to:
                    a.related_to.append(b.book_id)


def _dedupe_ids(books: list[Book]) -> None:
    counts: dict[str, int] = {}
    for b in books:
        base = b.book_id
        n = counts.get(base, 0) + 1
        counts[base] = n
        if n > 1:
            b.book_id = f"{base}_{n}"
