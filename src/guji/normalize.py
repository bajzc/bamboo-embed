"""Text normalization: PUA mapping, OpenCC t2s, variant unification (§3.5).

Two text fields are produced downstream:
  * ``text_raw``  — traditional original, but with PUA placeholders resolved
                    (mapped char, or ``□`` when unknown) so nothing pollutes
                    tokenizers / display.
  * ``text_norm`` — ``text_raw`` further converted to simplified + variant-
                    normalized, used only for retrieval.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PLACEHOLDER = "□"

# Private Use Areas (§3.5): BMP PUA + the two supplementary-plane PUAs.
_PUA_RANGES: tuple[tuple[int, int], ...] = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


def is_pua(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _PUA_RANGES)


def is_ext_cjk(cp: int) -> bool:
    """CJK unified/compat ideographs living above the BMP (byte-fallback risk)."""
    return (
        0x20000 <= cp <= 0x2FFFF   # Ext-B..F
        or 0x30000 <= cp <= 0x3FFFF  # Ext-G..
        or 0x2F800 <= cp <= 0x2FA1F  # CJK Compat Ideographs Supplement
    )


@lru_cache(maxsize=None)
def _load_pua_map(pua_map_path: str | None) -> dict[int, str]:
    """Load ``{hex_codepoint: replacement}`` -> ``{int: replacement}``.

    Entries whose replacement is empty or the placeholder are treated as
    *unmapped* and fall through to ``□``.
    """
    if not pua_map_path:
        return {}
    p = Path(pua_map_path)
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for k, v in raw.items():
        if not v or v == PLACEHOLDER:
            continue
        out[int(k, 16)] = v
    return out


def map_pua(text: str, pua_map_path: str | None = None) -> str:
    """Replace PUA codepoints with their mapped char, else ``□``."""
    mapping = _load_pua_map(pua_map_path)
    if not text:
        return text
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if is_pua(cp):
            out.append(mapping.get(cp, PLACEHOLDER))
        else:
            out.append(ch)
    return "".join(out)


@lru_cache(maxsize=1)
def _t2s():
    from opencc import OpenCC

    return OpenCC("t2s")  # Traditional -> Simplified (also folds many variants)


def to_raw(text: str, pua_map_path: str | None = None) -> str:
    """Display/citation form: original traditional text, PUA resolved."""
    return map_pua(text, pua_map_path)


def to_norm(text: str, pua_map_path: str | None = None) -> str:
    """Retrieval form: PUA resolved, then simplified + variant normalized."""
    return _t2s().convert(map_pua(text, pua_map_path))
