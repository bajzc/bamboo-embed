"""Download prebuilt GGUF models for the local llama-server profile (LLM + reranker).

Keeps everything on llama.cpp for these two roles so nothing needs Ollama or a mixed
setup — only `embedding` stays pinned to Ollama (see config.yaml, kept for cross-platform
vector reproducibility). Files land in `models/` for the `--model` flag documented in
the README's llama-server launch commands.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    filename: str
    description: str


# Verified-working GGUF sources (checked against each repo's file listing):
# - llm-large / llm-small: unsloth's official quants, same family already used in the
#   README's llama-server commands.
# - reranker: most community Qwen3-Reranker GGUF conversions are broken with llama.cpp
#   (missing the cls.output.weight tensor -> near-zero scores); Voodisss's repo used the
#   proper convert_hf_to_gguf.py path and is confirmed to work with `--reranking`.
CATALOG: dict[str, ModelSpec] = {
    "llm-large": ModelSpec(
        repo_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-IQ4_NL.gguf",
        description="35B-A3B MoE, ~18GB — full local `profile: local` LLM (HyDE + `guji ask`)",
    ),
    "llm-small": ModelSpec(
        repo_id="unsloth/Qwen3-8B-GGUF",
        filename="Qwen3-8B-IQ4_NL.gguf",
        description="8B dense, ~4.8GB — fits 16GB unified memory (e.g. M3 base)",
    ),
    "reranker": ModelSpec(
        repo_id="Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp",
        filename="Qwen3-Reranker-0.6B.Q8_0.gguf",
        description="0.6B cross-encoder — pairs with `rerank.backend: local` in config.yaml",
    ),
}


def fetch(key: str, dest_dir: Path) -> Path:
    """Download CATALOG[key] into dest_dir, returning the local file path.

    Uses huggingface_hub's cache (re-running is a no-op if already downloaded) and
    copies the result into dest_dir under its original filename so it matches the
    README's `--model models/<file>.gguf` commands.
    """
    from huggingface_hub import hf_hub_download

    spec = CATALOG[key]
    cached = hf_hub_download(repo_id=spec.repo_id, filename=spec.filename)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec.filename
    if not dest.exists():
        shutil.copyfile(cached, dest)
    return dest
