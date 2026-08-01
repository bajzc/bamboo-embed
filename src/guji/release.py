"""Package the derived corpus/index artifacts into a single downloadable archive.

`guji parse` + `guji embed` + `guji index` take hours (full-corpus embed alone is
~1h on a discrete GPU, 2-5h on M3, see README). This module lets one user ship the
*results* — passages.jsonl, dict.sqlite, the LanceDB vector store, the FTS5 index,
etc. — so anyone else with a matching `config.yaml: embedding` can skip straight to
`guji search`/`guji ask` by extracting a release archive into `data/`.

Distribution is via a Hugging Face dataset repo (``guji release publish``/``fetch``);
``guji release pack`` itself has no network dependency and just builds the archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Config

SCHEMA_VERSION = 1

# (release-manifest key, PathsCfg field name)
ARTIFACTS: list[tuple[str, str]] = [
    ("manifest", "manifest"),
    ("passages", "passages"),
    ("dict_db", "dict_db"),
    ("char_report", "char_report"),
    ("pua_map", "pua_map"),
    ("lancedb", "lancedb"),
    ("fts_db", "fts_db"),
]


def _artifact_path(cfg: Config, field: str) -> Path:
    return cfg.p(getattr(cfg.paths, field))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(path)).encode())
                with f.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
    else:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def missing_artifacts(cfg: Config) -> list[str]:
    """Artifact keys whose file/directory doesn't exist yet."""
    return [key for key, field in ARTIFACTS if not _artifact_path(cfg, field).exists()]


def archive_filename(label: str) -> str:
    return f"guji-rag-data-{label}.tar.gz"


def manifest_filename(label: str) -> str:
    return f"guji-rag-data-{label}.manifest.json"


def manifest_path_for(archive_path: Path) -> Path:
    name = archive_path.name
    if not name.endswith(".tar.gz"):
        raise ValueError(f"not a release archive: {archive_path}")
    return archive_path.with_name(name[: -len(".tar.gz")] + ".manifest.json")


def newest_archive(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("guji-rag-data-*.tar.gz"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


@dataclass
class PackResult:
    archive_path: Path
    manifest_path: Path
    archive_sha256: str
    total_bytes: int


def pack(
    cfg: Config,
    out_dir: Path,
    label: str | None = None,
    on_artifact: Callable[[str], None] | None = None,
) -> PackResult:
    """Tar+gzip all release artifacts into ``out_dir``, with a checksum manifest.

    Archive members keep the paths from ``config.yaml: paths`` (e.g. ``data/passages.jsonl``)
    so extracting at another repo's root lands artifacts exactly where that repo's own
    config expects them, even if the two repos disagree on ``data_dir``.
    """
    missing = missing_artifacts(cfg)
    if missing:
        raise FileNotFoundError(
            "missing artifacts, build them first: " + ", ".join(missing)
            + " (see README 语料处理/建索引: guji fetch/manifest/normalize/parse/embed/index)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    label = label or datetime.now(timezone.utc).strftime("%Y%m%d")
    archive_path = out_dir / archive_filename(label)
    manifest_path = out_dir / manifest_filename(label)

    files: dict[str, dict] = {}
    total_bytes = 0
    with tarfile.open(archive_path, "w:gz") as tar:
        for key, field in ARTIFACTS:
            src = _artifact_path(cfg, field)
            arcname = getattr(cfg.paths, field)
            size = _size(src)
            digest = _sha256(src)
            files[key] = {"path": arcname, "bytes": size, "sha256": digest}
            total_bytes += size
            tar.add(src, arcname=arcname)
            if on_artifact:
                on_artifact(key)

    archive_sha256 = _sha256(archive_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(cfg.root),
        "embedding": {
            "backend": cfg.embedding.backend,
            "model": cfg.embedding.model,
            "dim": cfg.embedding.dim,
        },
        "chunk": cfg.chunk.model_dump(),
        "archive": {
            "filename": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha256,
        },
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return PackResult(archive_path, manifest_path, archive_sha256, total_bytes)


def verify_archive(archive_path: Path, manifest: dict) -> None:
    """Raise if the downloaded archive doesn't match the manifest's recorded checksum."""
    expected = manifest["archive"]["sha256"]
    digest = _sha256(archive_path)
    if digest != expected:
        raise ValueError(
            f"checksum mismatch for {archive_path.name}: expected {expected}, got {digest} "
            "(download likely corrupted or truncated — retry)"
        )


def extract(cfg: Config, archive_path: Path, manifest: dict, force: bool = False) -> list[str]:
    """Extract a checksum-verified release archive into this repo's ``data/`` dir.

    Refuses to run if the local embedding model/dim doesn't match what the release
    was built with (the vectors would be silently incompatible) or if artifacts are
    already present locally, unless ``force=True``.
    """
    existing = [key for key, field in ARTIFACTS if _artifact_path(cfg, field).exists()]
    if existing and not force:
        raise FileExistsError(
            "artifacts already present locally, pass --force to overwrite: " + ", ".join(existing)
        )

    rel_emb = manifest.get("embedding", {})
    if rel_emb.get("model") != cfg.embedding.model or rel_emb.get("dim") != cfg.embedding.dim:
        raise ValueError(
            f"embedding mismatch: release was built with {rel_emb.get('model')} "
            f"(dim={rel_emb.get('dim')}), local config.yaml has {cfg.embedding.model} "
            f"(dim={cfg.embedding.dim}) — the vector index would be incompatible"
        )

    root = cfg.root.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if target != root and os.path.commonpath([root, target]) != str(root):
                raise ValueError(f"refusing to extract unsafe path: {member.name}")
        tar.extractall(root)  # noqa: S202 — paths validated above
    return [key for key, _ in ARTIFACTS]


def publish_to_hub(repo_id: str, archive_path: Path, manifest_path: Path, private: bool = False) -> str:
    """Upload a packed release (archive + its .manifest.json) to a HF dataset repo.

    Requires the ``release`` extra (``huggingface_hub``) and prior auth
    (``huggingface-cli login`` or ``HF_TOKEN`` env var).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    for p in (archive_path, manifest_path):
        api.upload_file(
            path_or_fileobj=str(p), path_in_repo=p.name,
            repo_id=repo_id, repo_type="dataset",
        )
    return f"https://huggingface.co/datasets/{repo_id}"


def download_from_hub(repo_id: str, label: str, revision: str | None = None) -> tuple[Path, dict]:
    """Download a release's manifest + archive from a HF dataset repo.

    Returns the local (cached) archive path and the parsed manifest dict.
    """
    from huggingface_hub import hf_hub_download

    manifest_file = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=manifest_filename(label), revision=revision,
    )
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    archive_file = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=manifest["archive"]["filename"], revision=revision,
    )
    return Path(archive_file), manifest
