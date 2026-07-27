"""Dictionary parser (form B, §3.4 / §5.3).

Entries look like ``字头\t释义`` with continuation lines starting with ``\t``.
Each headword becomes one ``char_entry`` row in ``dict.sqlite`` — never the
vector store. 康熙字典 p1+p2 merge into a single ``source_book``.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from ..models import Book, CharEntry
from ..normalize import to_norm, to_raw
from .headline import read_content_lines

# leading 【..】【..】 run before the first space -> section (【卯集下】【攴字部】)
_SECTION_RE = re.compile(r"^((?:【[^】]+】)+)")
_MAX_HEADWORD = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS char_entry (
  id            INTEGER PRIMARY KEY,
  headword      TEXT NOT NULL,
  headword_norm TEXT,
  source_book   TEXT NOT NULL,
  body_raw      TEXT NOT NULL,
  body_norm     TEXT NOT NULL,
  section       TEXT
);
CREATE INDEX IF NOT EXISTS idx_headword ON char_entry(headword);
CREATE INDEX IF NOT EXISTS idx_headword_norm ON char_entry(headword_norm);
"""


def _extract_section(body: str) -> str:
    m = _SECTION_RE.match(body)
    return m.group(1) if m else ""


def _iter_entries(lines: list[str]) -> Iterator[tuple[str, str, str]]:
    """Yield (headword, section, body_raw) from post-comment content lines."""
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if not line.strip() or line[:1] in ("\t", " "):
            i += 1
            continue
        # headword row
        if "\t" in line:
            hw, first_body = line.split("\t", 1)
        else:
            hw, first_body = line, ""
        body_parts = [first_body] if first_body.strip() else []
        i += 1
        while i < n and lines[i][:1] in ("\t", " ") and lines[i].strip():
            body_parts.append(lines[i].lstrip("\t "))
            i += 1

        hw = hw.strip()
        if not hw or len(hw) > _MAX_HEADWORD:
            continue  # prose / preface line, not a dictionary entry
        body = "\n".join(p for p in body_parts if p.strip())
        if not body:
            continue
        yield hw, _extract_section(body), body


def parse_book(book: Book, raw_root: Path, pua_map_path: str | None) -> Iterator[CharEntry]:
    for rel in book.source_files:               # merges 康熙 p1 + p2 in order
        lines = read_content_lines(raw_root / rel)
        for hw, section, body in _iter_entries(lines):
            yield CharEntry(
                headword=to_raw(hw, pua_map_path),
                headword_norm=to_norm(hw, pua_map_path),
                source_book=book.title,
                body_raw=to_raw(body, pua_map_path),
                body_norm=to_norm(body, pua_map_path),
                section=to_raw(section, pua_map_path),
            )


def create_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def insert_entries(conn: sqlite3.Connection, entries: Iterator[CharEntry], batch: int = 2000) -> int:
    total = 0
    rows: list[tuple] = []
    for e in entries:
        rows.append((e.headword, e.headword_norm, e.source_book, e.body_raw, e.body_norm, e.section))
        if len(rows) >= batch:
            conn.executemany(
                "INSERT INTO char_entry(headword,headword_norm,source_book,body_raw,body_norm,section)"
                " VALUES (?,?,?,?,?,?)",
                rows,
            )
            total += len(rows)
            rows.clear()
    if rows:
        conn.executemany(
            "INSERT INTO char_entry(headword,headword_norm,source_book,body_raw,body_norm,section)"
            " VALUES (?,?,?,?,?,?)",
            rows,
        )
        total += len(rows)
    conn.commit()
    return total
