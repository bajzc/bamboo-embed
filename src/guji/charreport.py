"""``guji normalize --check``: enumerate PUA + extension-CJK characters (§3.5).

Writes ``char_report.json`` (stats + samples) and merges a ``pua_map.json``
skeleton (every found PUA codepoint -> "□", human-fillable) without clobbering
existing hand-added mappings.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .normalize import PLACEHOLDER, is_ext_cjk, is_pua


def scan(raw_root: Path) -> dict:
    pua = Counter()
    ext = Counter()
    files = 0
    for path in sorted(raw_root.rglob("*.txt")):
        if "9.工具腳本" in path.parts:
            continue
        files += 1
        for ch in path.read_text(encoding="utf-8"):
            cp = ord(ch)
            if is_pua(cp):
                pua[ch] += 1
            elif is_ext_cjk(cp):
                ext[ch] += 1

    def top(counter: Counter, k: int = 40):
        return [
            {"char": ch, "codepoint": f"U+{ord(ch):04X}", "count": c}
            for ch, c in counter.most_common(k)
        ]

    return {
        "files_scanned": files,
        "pua": {
            "distinct": len(pua),
            "total_occurrences": sum(pua.values()),
            "samples": top(pua),
        },
        "ext_cjk": {
            "distinct": len(ext),
            "total_occurrences": sum(ext.values()),
            "samples": top(ext),
        },
        "_pua_chars": sorted({ord(ch) for ch in pua}),
    }


def write_report(report: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    report_path.write_text(
        json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def merge_pua_map(report: dict, pua_map_path: Path) -> int:
    """Add newly-seen PUA codepoints as ``□`` placeholders; keep existing edits."""
    existing: dict[str, str] = {}
    if pua_map_path.is_file():
        existing = json.loads(pua_map_path.read_text(encoding="utf-8"))
    added = 0
    for cp in report["_pua_chars"]:
        key = f"{cp:04X}"
        if key not in existing:
            existing[key] = PLACEHOLDER
            added += 1
    pua_map_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: existing[k] for k in sorted(existing, key=lambda x: int(x, 16))}
    pua_map_path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return added
