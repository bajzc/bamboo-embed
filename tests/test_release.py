from __future__ import annotations

from pathlib import Path

import pytest

from guji import release
from guji.config import Config

_PATHS = {
    "data_dir": "data",
    "manifest": "data/manifest.json",
    "passages": "data/passages.jsonl",
    "dict_db": "data/dict.sqlite",
    "pua_map": "data/pua_map.json",
    "char_report": "data/char_report.json",
    "lancedb": "data/lancedb",
    "fts_db": "data/fts.sqlite",
    "eval_questions": "eval/questions.jsonl",
    "eval_report_dir": "eval/report",
}


def _make_cfg(root: Path, dim: int = 8) -> Config:
    return Config.model_validate(
        {
            "profile": "local",
            "providers": {"local": {"llm": {"base_url": "http://x", "model": "m"}}},
            "corpus": {"repo_url": "http://x", "raw_dir": "data/raw/chtxt"},
            "paths": _PATHS,
            "chunk": {},
            "embedding": {"backend": "ollama", "model": "qwen3-embedding:0.6b", "dim": dim},
            "root": root,
        }
    )


def _write_fixture_artifacts(cfg: Config) -> None:
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text('{"books": []}', encoding="utf-8")
    cfg.passages_path.write_text('{"chunk_id": "a"}\n', encoding="utf-8")
    cfg.dict_db_path.write_bytes(b"sqlite-fake-bytes")
    cfg.char_report_path.write_text("{}", encoding="utf-8")
    cfg.pua_map_path.write_text("{}", encoding="utf-8")
    cfg.fts_db_path.write_bytes(b"fts-fake-bytes")
    cfg.lancedb_path.mkdir(parents=True, exist_ok=True)
    (cfg.lancedb_path / "passages.lance").mkdir()
    (cfg.lancedb_path / "passages.lance" / "data.bin").write_bytes(b"\x00\x01vector-fake")


def test_missing_artifacts_blocks_pack(tmp_path):
    cfg = _make_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        release.pack(cfg, tmp_path / "dist")


def test_pack_verify_extract_roundtrip(tmp_path):
    src_root = tmp_path / "producer"
    src_root.mkdir()
    cfg = _make_cfg(src_root)
    _write_fixture_artifacts(cfg)

    result = release.pack(cfg, tmp_path / "dist", label="test")
    assert result.archive_path.is_file()
    manifest = __import__("json").loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedding"] == {"backend": "ollama", "model": "qwen3-embedding:0.6b", "dim": 8}
    assert set(manifest["files"]) == {k for k, _ in release.ARTIFACTS}

    # verification passes on the untouched archive
    release.verify_archive(result.archive_path, manifest)

    # a consumer repo, same relative paths, no artifacts yet
    dst_root = tmp_path / "consumer"
    dst_root.mkdir()
    dst_cfg = _make_cfg(dst_root)
    keys = release.extract(dst_cfg, result.archive_path, manifest)
    assert set(keys) == {k for k, _ in release.ARTIFACTS}
    assert dst_cfg.passages_path.read_text(encoding="utf-8") == '{"chunk_id": "a"}\n'
    assert dst_cfg.dict_db_path.read_bytes() == b"sqlite-fake-bytes"
    assert (dst_cfg.lancedb_path / "passages.lance" / "data.bin").read_bytes() == b"\x00\x01vector-fake"


def test_extract_rejects_embedding_mismatch(tmp_path):
    src_root = tmp_path / "producer"
    src_root.mkdir()
    cfg = _make_cfg(src_root, dim=8)
    _write_fixture_artifacts(cfg)
    result = release.pack(cfg, tmp_path / "dist", label="test")
    manifest = __import__("json").loads(result.manifest_path.read_text(encoding="utf-8"))

    dst_root = tmp_path / "consumer"
    dst_root.mkdir()
    dst_cfg = _make_cfg(dst_root, dim=1024)  # different embedding dim
    with pytest.raises(ValueError, match="embedding mismatch"):
        release.extract(dst_cfg, result.archive_path, manifest)


def test_extract_refuses_to_overwrite_without_force(tmp_path):
    src_root = tmp_path / "producer"
    src_root.mkdir()
    cfg = _make_cfg(src_root)
    _write_fixture_artifacts(cfg)
    result = release.pack(cfg, tmp_path / "dist", label="test")
    manifest = __import__("json").loads(result.manifest_path.read_text(encoding="utf-8"))

    dst_root = tmp_path / "consumer"
    dst_root.mkdir()
    dst_cfg = _make_cfg(dst_root)
    release.extract(dst_cfg, result.archive_path, manifest)  # first extract succeeds

    with pytest.raises(FileExistsError):
        release.extract(dst_cfg, result.archive_path, manifest)
    release.extract(dst_cfg, result.archive_path, manifest, force=True)  # force overwrites fine


def test_verify_archive_detects_corruption(tmp_path):
    src_root = tmp_path / "producer"
    src_root.mkdir()
    cfg = _make_cfg(src_root)
    _write_fixture_artifacts(cfg)
    result = release.pack(cfg, tmp_path / "dist", label="test")
    manifest = __import__("json").loads(result.manifest_path.read_text(encoding="utf-8"))

    with result.archive_path.open("r+b") as f:
        f.seek(0)
        f.write(b"\x00")  # flip the first byte

    with pytest.raises(ValueError, match="checksum mismatch"):
        release.verify_archive(result.archive_path, manifest)
