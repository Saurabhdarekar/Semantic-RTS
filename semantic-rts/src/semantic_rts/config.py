"""Config loading: YAML → Pydantic models, with env-var substitution and deep-merge."""

from __future__ import annotations

import copy
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    provider: str = "gemini"
    chat_model: str = "gemini-2.5-flash-lite-preview-06-17"
    embedding_model: str = "text-embedding-005"
    max_retries: int = 3
    rate_limit_rpm: int = 14
    rate_limit_rpd: int = 1400


class VectorStoreConfig(BaseModel):
    type: str = "faiss"
    index_kind: str = "IndexFlatIP"
    embedding_dim: int = 768


class TierKeywords(BaseModel):
    tier_1: list[str] = Field(default_factory=lambda: [
        "security", "auth", "password", "crypto", "privacy", "payment",
        "authentication", "authorization",
    ])
    tier_2: list[str] = Field(default_factory=lambda: [
        "persistence", "database", "transaction", "concurrency",
        "jdbc", "hibernate", "sql",
    ])
    tier_3: list[str] = Field(default_factory=lambda: [
        "api", "controller", "service", "endpoint", "handler",
    ])
    tier_4: list[str] = Field(default_factory=lambda: [
        "util", "helper", "format", "parse", "convert",
    ])
    tier_5: list[str] = Field(default_factory=lambda: [
        "getter", "setter", "tostring", "equals", "hashcode", "constructor",
    ])

    def for_tier(self, tier: int) -> list[str]:
        return getattr(self, f"tier_{tier}", [])


class KBConfig(BaseModel):
    summary_max_tokens: int = 200
    source_max_tokens: int = 1500
    tier_keywords: TierKeywords = Field(default_factory=TierKeywords)


class RetrievalConfig(BaseModel):
    top_k: int = 30
    similarity_threshold: float = 0.55
    token_overlap_weight: float = 0.12
    package_proximity_weight: float = 0.08
    bm25_weight: float = 0.20
    negative_pass_enabled: bool = True
    negative_pass_penalty: float = 0.25
    adaptive_threshold_enabled: bool = True
    adaptive_threshold_min: float = 0.30


class SafetyBridgeConfig(BaseModel):
    always_include_tier_1: bool = True
    tier_2_threshold: float = 0.40


class PrecisionFilterConfig(BaseModel):
    tier_5_min: float = 0.65
    tier_4_min: float = 0.50


class SelectorConfig(BaseModel):
    safety_bridge: SafetyBridgeConfig = Field(default_factory=SafetyBridgeConfig)
    precision_filter: PrecisionFilterConfig = Field(default_factory=PrecisionFilterConfig)
    max_selected: int = 100
    topology_multiplier: float = 1.20
    topology_trigger_change_types: list[str] = Field(
        default_factory=lambda: ["api_change", "refactoring"]
    )
    topology_trigger_packages: list[str] = Field(
        default_factory=lambda: ["util", "common", "base", "core", "shared", "helper"]
    )
    sensitivity_multiplier_enabled: bool = True
    cross_encoder_enabled: bool = False
    cross_encoder_top_n: int = 40


class ImpactConfig(BaseModel):
    skip_intent_agent: bool = False
    diff_max_tokens: int = 8000
    micro_diff_bypass_enabled: bool = True
    micro_diff_max_lines: int = 5


class PathsConfig(BaseModel):
    defects4j_home: str = ""
    kb_dir: str = "data/kb"
    eval_dir: str = "data/eval"
    cache_dir: str = "data/cache"
    logs_dir: str = "data/logs"
    work_dir: str = "work/checkouts"   # where Defects4J bug checkouts live

    def kb_path(self, project: str) -> Path:
        return Path(self.kb_dir) / project

    def cache_path(self, kind: str) -> Path:
        return Path(self.cache_dir) / kind


class AblationFlags(BaseModel):
    safety_bridge_enabled: bool = True
    precision_filter_enabled: bool = True
    tiers_enabled: bool = True
    semantic_only: bool = False  # disable all deterministic bypasses; pure semantic search


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    kb: KBConfig = Field(default_factory=KBConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    impact: ImpactConfig = Field(default_factory=ImpactConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ablation: AblationFlags = Field(default_factory=AblationFlags)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _substitute_env(obj: Any) -> Any:
    """Replace ${VAR} with the current environment value (empty string if unset)."""
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {k: _substitute_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"


def load_config(
    config_path: str | Path | None = None,
    overlay_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load Config from YAML, optionally deep-merged with an overlay file.

    Args:
        config_path:  Path to main YAML (defaults to config/default.yaml).
        overlay_path: Optional ablation/override YAML merged on top.
        overrides:    Dict of dotted-key overrides, e.g. {"llm.chat_model": "gemini-2.5-pro"}.
    """
    base_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    raw: dict = _load_yaml(base_path) if base_path.exists() else {}

    if overlay_path:
        overlay = _load_yaml(Path(overlay_path))
        raw = _deep_merge(raw, overlay)

    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            node = raw
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value

    raw = _substitute_env(raw)

    # Extract ablation flags from _ablation key if present (used by ablation YAMLs)
    ablation_raw = raw.pop("_ablation", {})
    if ablation_raw:
        raw.setdefault("ablation", {}).update(ablation_raw)

    return Config.model_validate(raw)
