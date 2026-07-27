from __future__ import annotations

import json

from guji.normalize import PLACEHOLDER, is_ext_cjk, is_pua, map_pua, to_norm


def test_pua_detection():
    assert is_pua(0xE000)
    assert is_pua(0xF8FF)
    assert is_pua(0xF0000)
    assert not is_pua(ord("字"))


def test_ext_cjk_detection():
    assert is_ext_cjk(0x20000)   # Ext-B
    assert not is_ext_cjk(ord("字"))


def test_unmapped_pua_becomes_placeholder():
    s = "A" + chr(0xE000) + "B"
    assert map_pua(s) == "A" + PLACEHOLDER + "B"


def test_mapped_pua_uses_table(tmp_path):
    m = tmp_path / "pua.json"
    m.write_text(json.dumps({"E000": "X"}), encoding="utf-8")
    assert map_pua("A" + chr(0xE000) + "B", str(m)) == "AXB"


def test_to_norm_simplifies():
    # 國 (traditional) -> 国 (simplified)
    assert to_norm("國") == "国"
