"""``guji stats``: distribution of chunks/entries + corpus char-budget check.

The corpus total (27,865,176, §3.1) is a ``wc -m`` count that *includes newline
characters*. Summing chunk text can never equal it, and the 1-paragraph overlap
double-counts shared paragraphs. So the budget is reconciled honestly:

    content (overlap-free passages + dict bodies)
  + newlines + comment/heading lines
  ≈ corpus total        (remaining gap should be < 2%)
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import Config
from .models import Manifest
from .normalize import is_pua, to_raw
from .parse import narrative
from .parse.headline import find_head, read_content_lines

CORPUS_TOTAL_CHARS = 27_865_176  # spec §3.1 (verified against the clone)
_COMMENT_RE = re.compile(r"^#\S")            # '#中華...' (also matches '##...')
_HEAD_RE = re.compile(r"^#{1,2}[ \t]+\S")


def _load_passages(path: Path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _unique_content_chars(cfg: Config, books) -> int:
    """Overlap-free content chars for narrative/poetry books (source-derived)."""
    total = 0
    for b in books:
        if b.form == "dictionary":
            continue
        src = cfg.raw_path / b.source_files[0]
        content = read_content_lines(src)
        head = find_head(content, Path(src.stem).name)
        for _, paras in narrative._juans(content, head.head_line_index):
            total += sum(len(to_raw(p, str(cfg.pua_map_path))) for p in paras)
    return total


def _structural_chars(cfg: Config) -> tuple[int, int]:
    """(newline chars, comment+heading line chars) across the raw corpus."""
    newlines = struct = 0
    for p in sorted(cfg.raw_path.rglob("*.txt")):
        if "9.工具腳本" in p.parts:
            continue
        t = p.read_text(encoding="utf-8")
        newlines += t.count("\n")
        for ln in t.split("\n"):
            if _COMMENT_RE.match(ln) or _HEAD_RE.match(ln):
                struct += len(ln)
    return newlines, struct


def compute(cfg: Config) -> dict:
    passages = _load_passages(cfg.passages_path)
    m = Manifest.model_validate_json(cfg.manifest_path.read_text(encoding="utf-8"))
    live = [b for b in m.books if not (b.is_empty or b.is_stub)]

    by_cat: dict[str, dict[str, int]] = {}
    passage_raw_chars = 0
    pua_residue = 0
    for p in passages:
        c = by_cat.setdefault(p["category"], {"chunks": 0, "chars": 0})
        c["chunks"] += 1
        c["chars"] += p["char_count"]
        passage_raw_chars += p["char_count"]
        pua_residue += sum(1 for ch in p["text_raw"] if is_pua(ord(ch)))

    dict_rows = dict_chars = 0
    if cfg.dict_db_path.is_file():
        conn = sqlite3.connect(cfg.dict_db_path)
        dict_rows = conn.execute("SELECT COUNT(*) FROM char_entry").fetchone()[0]
        dict_chars = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(headword)+LENGTH(body_raw)),0) FROM char_entry"
        ).fetchone()[0]
        conn.close()

    uniq_passage = _unique_content_chars(cfg, live)
    newlines, struct = _structural_chars(cfg)

    covered = uniq_passage + dict_chars
    accounted = covered + newlines + struct
    unexplained = CORPUS_TOTAL_CHARS - accounted
    return {
        "by_cat": by_cat,
        "passages_total": len(passages),
        "passage_raw_chars": passage_raw_chars,
        "pua_residue": pua_residue,
        "dict_rows": dict_rows,
        "dict_chars": dict_chars,
        "uniq_passage_chars": uniq_passage,
        "newlines": newlines,
        "struct_chars": struct,
        "covered_chars": covered,
        "accounted_chars": accounted,
        "corpus_total": CORPUS_TOTAL_CHARS,
        "coverage_pct": 100.0 * covered / CORPUS_TOTAL_CHARS,
        "accounted_pct": 100.0 * accounted / CORPUS_TOTAL_CHARS,
        "unexplained": unexplained,
        "unexplained_pct": 100.0 * unexplained / CORPUS_TOTAL_CHARS,
    }


def render(cfg: Config, console: Console | None = None) -> dict:
    console = console or Console()
    s = compute(cfg)

    table = Table(title="passages.jsonl — chunks & chars by category")
    table.add_column("category")
    table.add_column("chunks", justify="right")
    table.add_column("chars (w/ overlap)", justify="right")
    for cat, v in sorted(s["by_cat"].items(), key=lambda kv: -kv[1]["chars"]):
        table.add_row(cat, f"{v['chunks']:,}", f"{v['chars']:,}")
    table.add_row("[bold]TOTAL", f"[bold]{s['passages_total']:,}", f"[bold]{s['passage_raw_chars']:,}")
    console.print(table)
    console.print(f"dict.sqlite: [bold]{s['dict_rows']:,}[/] entries, {s['dict_chars']:,} body chars")

    console.rule("corpus char-budget reconciliation")
    console.print(f"content (overlap-free passages + dict): [bold]{s['covered_chars']:,}[/] "
                  f"= [bold]{s['coverage_pct']:.2f}%[/] of corpus")
    console.print(f"  + newlines {s['newlines']:,}  + comment/heading {s['struct_chars']:,}")
    console.print(f"  = accounted [bold]{s['accounted_chars']:,}[/] ({s['accounted_pct']:.2f}%)  "
                  f"unexplained [bold]{s['unexplained']:,}[/] ({s['unexplained_pct']:.2f}%)")
    residue_style = "green" if s["pua_residue"] == 0 else "red"
    console.print(f"PUA residue in text_raw: [{residue_style}]{s['pua_residue']}[/]")
    return s
