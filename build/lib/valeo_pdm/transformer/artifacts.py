from __future__ import annotations

from pathlib import Path

from valeo_pdm.paths import checkpoints_dir


def checkpoint_dir_name(equipment_code: str, meas_code: str, model_type: str) -> str:
    return f"exp_{equipment_code}_{meas_code}_{model_type}"


def checkpoint_filename(model_type: str) -> str:
    return f"{model_type}_best.pt"


def resolve_checkpoint_path(equipment_code: str, meas_code: str, model_type: str) -> Path:
    exp = checkpoint_dir_name(equipment_code, meas_code, model_type)
    fname = checkpoint_filename(model_type)
    return checkpoints_dir() / exp / fname


def training_output_dir(equipment_code: str, meas_code: str, model_type: str) -> Path:
    exp = checkpoint_dir_name(equipment_code, meas_code, model_type)
    return checkpoints_dir() / exp
