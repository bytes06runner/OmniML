from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

TrainingPath = Literal["sklearn", "pytorch"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_training_path() -> TrainingPath:
    raw = (os.environ.get("OMNIML_TRAINING_PATH") or "sklearn").strip().lower()
    if raw in {"pytorch", "torch", "path_a", "a"}:
        return "pytorch"
    return "sklearn"


@dataclass(frozen=True)
class OmniMLSettings:
    """Runtime flags for ablation experiments and optional pipeline behavior."""

    enable_hitl: bool = True
    enable_xai: bool = True
    monolithic_mode: bool = False
    training_path: TrainingPath = "sklearn"


def get_settings() -> OmniMLSettings:
    return OmniMLSettings(
        enable_hitl=_env_bool("OMNIML_ENABLE_HITL", True),
        enable_xai=_env_bool("OMNIML_ENABLE_XAI", True),
        monolithic_mode=_env_bool("OMNIML_MONOLITHIC", False),
        training_path=_env_training_path(),
    )


def resolve_training_path(state: Optional[Dict[str, Any]] = None) -> TrainingPath:
    """State training_config.training_path overrides OMNIML_TRAINING_PATH env."""
    if state:
        training_config = state.get("training_config") or {}
        raw = (training_config.get("training_path") or "").strip().lower()
        if raw in {"pytorch", "torch", "path_a", "a"}:
            return "pytorch"
        if raw in {"sklearn", "path_b", "b"}:
            return "sklearn"
    return get_settings().training_path
