"""Spawn-on-demand lifecycle for local model-serving processes.

Problem: the LLM (`profile: local`), reranker (`rerank.backend: local`), and embedding
(Ollama) backends are each a separate long-running process the README has you start by
hand and leave resident. On a 16G unified-memory machine, having the LLM + reranker +
embedding model all loaded at once can OOM — Metal fails with
`kIOGPUCommandBufferCallbackErrorOutOfMemory` even with the "small" 8B LLM + 0.6B
reranker combo once ctx-size and KV cache are accounted for.

Two different lifecycles, because the three aren't symmetric (see `_Manager`'s
docstring for the full reasoning):

  - `ensure_llm()` / `ensure_reranker()` — exclusive: only one of these two heavy
    single-model `llama-server` processes is ever allowed to run at a time. Called
    right at the point of use (inside the reranker HTTP call, inside the HyDE/generation
    LLM call); each stops the other if it's running. `guji ask`'s tool-calling loop
    resends the full conversation on every request (llama-server holds no state across
    calls here), so swapping mid-conversation is safe — it costs a reload (tens of
    seconds), not correctness.
  - `ensure_embedding()` — a lightweight Ollama daemon, started once and left running
    for the rest of the command (Ollama manages its own model memory via its own idle
    timeout, so there's no reload cost to interleaving it with the exclusive slot).

Entirely opt-in via `config.yaml: providers.local.llm.launch` / `rerank.launch` /
`embedding.launch` (see `Config.LaunchCfg`). Without a `launch:` block for a role,
`ensure_*()` for that role is a no-op — guji just connects to base_url and assumes it's
already up, exactly as before this module existed. A server guji didn't start itself
(base_url already answering) is also never touched/stopped.
"""

from __future__ import annotations

import logging
import subprocess
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config, LaunchCfg

log = logging.getLogger("guji.procman")

Role = str  # "llm" | "reranker"


@dataclass
class _Started:
    role: Role
    base_url: str
    proc: subprocess.Popen


def _is_up(base_url: str, health_path: str, timeout: float = 2.0) -> bool:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}{health_path}", timeout=timeout)
        return resp.status_code < 500
    except httpx.HTTPError:
        return False


def _start(role: Role, base_url: str, launch: LaunchCfg, log_dir: Path) -> _Started:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{role}.log"
    log.info("starting %s: %s (log: %s)", role, " ".join(launch.command), log_path)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        launch.command, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + launch.ready_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{role} exited during startup (code {proc.returncode}); see {log_path}"
            )
        if _is_up(base_url, launch.health_path):
            log.info("%s ready at %s", role, base_url)
            return _Started(role, base_url, proc)
        time.sleep(0.5)

    proc.terminate()
    raise RuntimeError(f"{role} did not become ready within {launch.ready_timeout}s; see {log_path}")


def _stop(started: _Started) -> None:
    if started.proc.poll() is not None:
        return
    log.info("stopping %s (pid %d)", started.role, started.proc.pid)
    started.proc.terminate()
    try:
        started.proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        log.warning("%s did not exit in time, killing", started.role)
        started.proc.kill()
        started.proc.wait()


class _Manager:
    """Two lifecycles, because the roles aren't symmetric:

    - llm / reranker: each is a single-model `llama-server` heavy enough that only one
      may be resident at a time on a 16G machine — `ensure()` exclusively swaps them.
    - embedding: Ollama is a lightweight daemon that loads/unloads the (small) embedding
      model itself on its own idle timeout, and is needed interleaved with both of the
      above (every dense-path search embeds the query). Swapping it in the exclusive
      slot too would mean reloading the 8B/35B LLM after every single query embed —
      instead it's started once, alongside whichever exclusive role is active, and
      stopped only when the command ends.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.log_dir = cfg.root / ".guji" / "logs"
        self._active: _Started | None = None
        self._services: dict[Role, _Started] = {}

    def _launch_for(self, role: Role) -> tuple[LaunchCfg | None, str | None]:
        if role == "llm":
            llm = self.cfg.active_llm()
            return llm.launch, llm.base_url
        if role == "reranker":
            return self.cfg.rerank.launch, self.cfg.rerank.base_url
        if role == "embedding":
            return self.cfg.embedding.launch, self.cfg.embedding.base_url
        raise ValueError(role)

    def ensure(self, role: Role) -> None:
        launch, base_url = self._launch_for(role)
        if launch is None:
            return  # not opted into managed lifecycle for this role — leave it alone

        if self._active is not None and self._active.role == role:
            return  # already the exclusively-running managed process

        if self._active is not None:
            _stop(self._active)
            self._active = None

        if _is_up(base_url, launch.health_path):
            return  # something else (not us) is already serving this role — leave it

        self._active = _start(role, base_url, launch, self.log_dir)

    def ensure_service(self, role: Role) -> None:
        launch, base_url = self._launch_for(role)
        if launch is None:
            return

        if role in self._services or _is_up(base_url, launch.health_path):
            return  # already running (by us, or by something else) — leave it

        self._services[role] = _start(role, base_url, launch, self.log_dir)

    def close(self) -> None:
        if self._active is not None:
            _stop(self._active)
            self._active = None
        for started in self._services.values():
            _stop(started)
        self._services.clear()


_current: ContextVar[_Manager | None] = ContextVar("guji_procman_current", default=None)


@contextmanager
def managed_backends(cfg: Config):
    """Scope in which `ensure_llm()` / `ensure_reranker()` calls take effect.

    Wrap a whole CLI command in this; the actual start/stop happens lazily at each
    `ensure_*()` call site (inside the reranker/LLM HTTP calls), exclusively swapping
    the single managed `llama-server` process as needed. Whatever's left running when
    the command ends is stopped here.
    """
    mgr = _Manager(cfg)
    token = _current.set(mgr)
    try:
        yield
    finally:
        _current.reset(token)
        mgr.close()


def ensure_llm() -> None:
    mgr = _current.get()
    if mgr is not None:
        mgr.ensure("llm")


def ensure_reranker() -> None:
    mgr = _current.get()
    if mgr is not None:
        mgr.ensure("reranker")


def ensure_embedding() -> None:
    mgr = _current.get()
    if mgr is not None:
        mgr.ensure_service("embedding")
