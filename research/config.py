"""Loads models.yaml and .env from the repo root."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SANDBOX = os.path.join(tempfile.gettempdir(), "research_agent_sandbox")  # Claude's cwd: no CLAUDE.md, no project files
load_dotenv(ROOT / ".env")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> "ModelRef":
        provider, _, model = spec.partition("/")
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    min_interval: float = 1.0
    extra_body: dict | None = None

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.key_env) or None


@lru_cache
def load(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "models.yaml").read_text(encoding="utf-8"))


@lru_cache
def providers() -> dict[str, Provider]:
    return {name: Provider(name=name, **p) for name, p in load()["providers"].items()}


@lru_cache
def role(name: str) -> list[ModelRef]:
    return [ModelRef.parse(s) for s in load()["roles"].get(name, [])]


@lru_cache
def jury() -> tuple[list[ModelRef], list[ModelRef]]:
    cfg = load()
    return [ModelRef.parse(s) for s in cfg.get("jury", [])], [ModelRef.parse(s) for s in cfg.get("jury_fallback", [])]


def presets() -> dict[str, dict]:
    return load()["presets"]


def default_preset() -> str:
    return load().get("default_preset", "sonnet")


def language() -> str:
    return load().get("language", "Korean")


def nli_model() -> str | None:
    return (load().get("nli") or {}).get("model")


@dataclass(frozen=True)
class Researcher:
    key: str
    label: str
    search: str                    # papers | web | both
    prefer: tuple[str, ...]        # document kinds read first (see sources.LABELS)
    models: tuple[ModelRef, ...]


def parallel() -> dict:
    return load().get("parallel") or {}


def researchers() -> list[Researcher]:
    """Researchers for parallel mode, plus a Claude one if claude_researcher is on."""
    cfg = parallel()
    out = [Researcher(key, r.get("label", key), r.get("search", "both"), tuple(r.get("prefer", [])),
                      tuple(ModelRef.parse(m) for m in r.get("models", [])))
           for key, r in (cfg.get("researchers") or {}).items()]
    if cfg.get("claude_researcher"):
        out.append(Researcher("claude", f"Claude {cfg.get('claude_model', 'haiku')}", "both", (),
                              (ModelRef("claude", cfg.get("claude_model", "haiku")),)))
    return out


def compare_models() -> list[ModelRef]:
    return [ModelRef.parse(m) for m in parallel().get("compare", [])]
