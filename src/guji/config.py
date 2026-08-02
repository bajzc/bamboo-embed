"""Configuration loader.

Everything model- and path-related comes from ``config.yaml`` (plus environment
variables for secrets). Nothing is hard-coded in business logic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


class CorpusCfg(BaseModel):
    repo_url: str
    raw_dir: str


class PathsCfg(BaseModel):
    data_dir: str
    manifest: str
    passages: str
    dict_db: str
    pua_map: str
    char_report: str
    lancedb: str
    fts_db: str
    eval_questions: str
    eval_report_dir: str


class ChunkCfg(BaseModel):
    narrative_min_chars: int = 250
    narrative_max_chars: int = 400
    overlap_paragraphs: int = 1


class LaunchCfg(BaseModel):
    """Spawn-on-demand for a local server process (§ OOM mitigation on 16G machines).

    Opt-in only: leave unset and guji behaves exactly as before (connect to whatever
    is already listening at base_url, never touching its lifecycle). Set it and guji
    starts this exact command when base_url isn't already serving, and stops the
    process it started once it's no longer needed — so the LLM/reranker/embedding
    servers don't have to sit resident between (or, for LLM/reranker, even within)
    `guji ask` invocations. See `guji.procman`.
    """

    command: list[str]
    health_path: str = "/health"
    ready_timeout: float = 300.0


class EmbeddingCfg(BaseModel):
    backend: str
    model: str
    dim: int
    base_url: str = "http://localhost:11434"
    query_instruct: str = ""
    launch: LaunchCfg | None = None


class RetrieveCfg(BaseModel):
    dense_k: int = 50
    sparse_k: int = 50
    rrf_k: int = 60
    fuse_k: int = 50
    top_k: int = 8


class LlmCfg(BaseModel):
    base_url: str
    model: str
    api_key_env: str | None = None
    launch: LaunchCfg | None = None


class ProviderCfg(BaseModel):
    llm: LlmCfg


class RerankCfg(BaseModel):
    backend: str = "none"
    model: str = ""
    base_url: str = ""
    api_key_env: str | None = None
    threshold: float = 0.35
    launch: LaunchCfg | None = None


class HydeCfg(BaseModel):
    enabled: bool = True
    max_tokens: int = 256


class DedupCfg(BaseModel):
    enabled: bool = True
    jaccard: float = 0.7


class GenerateCfg(BaseModel):
    max_tokens: int = 1024
    max_tool_rounds: int = 4          # LLM tool-call turns before forcing a final answer
    retry_on_violation: int = 1       # re-prompt attempts when citations/quotes fail validation
    quote_open: str = "『"            # verbatim-quote delimiters (§4.2 constraint 2)
    quote_close: str = "』"
    validate_citations: bool = True   # citation/quote/link validation + reject-and-retry (§4.2)


class Config(BaseModel):
    profile: str
    providers: dict[str, ProviderCfg]
    corpus: CorpusCfg
    paths: PathsCfg
    chunk: ChunkCfg
    embedding: EmbeddingCfg
    retrieve: RetrieveCfg = RetrieveCfg()
    rerank: RerankCfg = RerankCfg()
    hyde: HydeCfg = HydeCfg()
    dedup: DedupCfg = DedupCfg()
    generate: GenerateCfg = GenerateCfg()

    # Repo root, injected at load time so callers can resolve relative paths.
    root: Path

    def active_llm(self) -> LlmCfg:
        return self.providers[self.profile].llm

    @staticmethod
    def api_key(env_name: str | None) -> str | None:
        return os.environ.get(env_name) if env_name else None

    def p(self, relative: str) -> Path:
        """Resolve a repo-relative path to an absolute :class:`Path`."""
        return (self.root / relative).resolve()

    @property
    def raw_path(self) -> Path:
        return self.p(self.corpus.raw_dir)

    @property
    def manifest_path(self) -> Path:
        return self.p(self.paths.manifest)

    @property
    def passages_path(self) -> Path:
        return self.p(self.paths.passages)

    @property
    def dict_db_path(self) -> Path:
        return self.p(self.paths.dict_db)

    @property
    def pua_map_path(self) -> Path:
        return self.p(self.paths.pua_map)

    @property
    def char_report_path(self) -> Path:
        return self.p(self.paths.char_report)

    @property
    def lancedb_path(self) -> Path:
        return self.p(self.paths.lancedb)

    @property
    def fts_db_path(self) -> Path:
        return self.p(self.paths.fts_db)

    @property
    def eval_questions_path(self) -> Path:
        return self.p(self.paths.eval_questions)

    @property
    def eval_report_dir(self) -> Path:
        return self.p(self.paths.eval_report_dir)


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` looking for config.yaml."""
    cur = (start or Path.cwd()).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "config.yaml").is_file():
            return cand
    raise FileNotFoundError("config.yaml not found in cwd or any parent directory")


@lru_cache(maxsize=None)
def load_config(config_path: str | None = None) -> Config:
    path = Path(config_path) if config_path else (find_repo_root() / "config.yaml")
    root = path.resolve().parent
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # allow env override of the active profile only
    data["profile"] = os.environ.get("GUJI_PROFILE", data.get("profile", "local"))
    data["root"] = root
    return Config.model_validate(data)
