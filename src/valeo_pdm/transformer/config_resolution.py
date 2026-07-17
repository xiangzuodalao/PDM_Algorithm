from __future__ import annotations

from copy import deepcopy
from typing import Any


ModelConfig = dict[str, Any]
ModelMap = dict[str, dict[str, ModelConfig]]
DEFAULTABLE_FIELDS = frozenset({"model_type", "freq", "days_back", "source"})


def _as_mapping(value: object, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field} 必须是映射")
    return value


def get_model_block(models: object, equipment_code: str, meas_code: str) -> ModelConfig | None:
    """从模型映射中读取原始场景块，不应用默认值。"""

    model_map = _as_mapping(models, field="models")
    equipment = model_map.get(equipment_code)
    if equipment is None:
        return None
    equipment_map = _as_mapping(equipment, field=f"models.{equipment_code}")
    block = equipment_map.get(meas_code)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise TypeError(f"models.{equipment_code}.{meas_code} 必须是映射")
    return deepcopy(block)


def apply_defaults(block: ModelConfig, defaults: object) -> ModelConfig:
    """以场景块覆盖允许的顶层默认值；嵌套字段不做深度合并。"""

    default_map = _as_mapping(defaults, field="defaults")
    allowed_defaults = {
        key: deepcopy(value) for key, value in default_map.items() if key in DEFAULTABLE_FIELDS
    }
    return {**allowed_defaults, **deepcopy(block)}


def resolve_model_config(
    *,
    defaults: object,
    yaml_models: object,
    db_models: object,
    equipment_code: str,
    meas_code: str,
) -> tuple[ModelConfig | None, str | None]:
    """按 DB 整块优先、YAML 兜底，再由 defaults 补缺的顺序解析配置。"""

    db_block = get_model_block(db_models, equipment_code, meas_code)
    if db_block is not None:
        return apply_defaults(db_block, defaults), "db"

    yaml_block = get_model_block(yaml_models, equipment_code, meas_code)
    if yaml_block is not None:
        return apply_defaults(yaml_block, defaults), "yaml"

    return None, None


def merge_effective_models(*, defaults: object, yaml_models: object, db_models: object) -> ModelMap:
    """合并所有场景；DB 同名场景整块覆盖 YAML，再补顶层默认值。"""

    yaml_map = _as_mapping(yaml_models, field="models")
    db_map = _as_mapping(db_models, field="db_models")
    equipment_codes = set(yaml_map) | set(db_map)
    merged: ModelMap = {}

    for equipment_code in sorted(equipment_codes):
        yaml_equipment = _as_mapping(yaml_map.get(equipment_code), field=f"models.{equipment_code}")
        db_equipment = _as_mapping(db_map.get(equipment_code), field=f"db_models.{equipment_code}")
        meas_codes = set(yaml_equipment) | set(db_equipment)
        merged[equipment_code] = {}

        for meas_code in sorted(meas_codes):
            block = db_equipment.get(meas_code, yaml_equipment.get(meas_code))
            if not isinstance(block, dict):
                raise TypeError(f"models.{equipment_code}.{meas_code} 必须是映射")
            merged[equipment_code][meas_code] = apply_defaults(block, defaults)

    return merged
