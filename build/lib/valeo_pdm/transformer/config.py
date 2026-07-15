from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from valeo_pdm.paths import configs_dir

_CONFIG_CACHE: Optional[Dict] = None


def model_registry_path() -> Path:
    env = os.getenv("VALEO_PDM_MODEL_REGISTRY")
    if env:
        return Path(env).expanduser().resolve()
    return (configs_dir() / "model_registry.yaml").resolve()


def load_model_registry(force_reload: bool = False) -> Dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or force_reload:
        path = model_registry_path()
        with open(path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f) or {}
    return _CONFIG_CACHE


def get_model_config(equipment_code: str, meas_code: str) -> Optional[Dict[str, Any]]:
    registry = load_model_registry()
    models = registry.get("models", {})
    defaults = registry.get("defaults", {})

    config = models.get(equipment_code, {}).get(meas_code)
    if config:
        return {**defaults, **config}
    return defaults.copy() if defaults else None


def list_all_models() -> Dict[str, Dict[str, Dict]]:
    registry = load_model_registry()
    return registry.get("models", {})


def get_train_params(equipment_code: str, meas_code: str) -> Optional[Dict[str, Any]]:
    config = get_model_config(equipment_code, meas_code)
    if config:
        return config.get("train_params")
    return None

