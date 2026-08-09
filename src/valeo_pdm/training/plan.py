from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from valeo_pdm.paths import resolve_data_path
from valeo_pdm.transformer.data import validate_window_params


PLAN_VERSION = 1
API_TRAIN_MODEL_TYPES = frozenset({"informer", "autoformer"})
ALLOWED_DATA_SOURCES = frozenset({"db", "postgres", "sqlserver", "csv"})
ALLOWED_EXECUTION_MODES = frozenset({"platform", "local_only"})
MODEL_INFO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

PARAM_CASTERS: dict[str, type] = {
    "seq_len": int,
    "label_len": int,
    "pred_len": int,
    "batch_size": int,
    "epochs": int,
    "lr": float,
    "d_model": int,
    "n_heads": int,
    "d_ff": int,
    "dropout": float,
    "e_layers": int,
    "d_layers": int,
    "attn_type": str,
    "distil": bool,
    "patience": int,
    "lr_factor": float,
    "lr_patience": int,
    "weight_decay": float,
    "grad_clip": float,
    "stride": int,
    "moving_avg": int,
}

COMMON_TRAIN_PARAM_NAMES = frozenset(
    {
        "seq_len",
        "label_len",
        "pred_len",
        "batch_size",
        "epochs",
        "lr",
        "d_model",
        "n_heads",
        "d_ff",
        "dropout",
        "e_layers",
        "d_layers",
        "patience",
        "lr_factor",
        "lr_patience",
        "weight_decay",
        "grad_clip",
        "stride",
    }
)
MODEL_TRAIN_PARAM_NAMES = {
    "informer": COMMON_TRAIN_PARAM_NAMES | {"attn_type", "distil"},
    "autoformer": COMMON_TRAIN_PARAM_NAMES | {"moving_avg"},
}

INFORMER_DEFAULTS: dict[str, Any] = {
    "seq_len": 672,
    "label_len": 192,
    "pred_len": 288,
    "batch_size": 32,
    "epochs": 100,
    "lr": 5e-4,
    "d_model": 256,
    "n_heads": 8,
    "d_ff": 512,
    "dropout": 0.1,
    "e_layers": 2,
    "d_layers": 2,
    "attn_type": "prob",
    "distil": True,
    "patience": 10,
    "lr_factor": 0.5,
    "lr_patience": 5,
    "weight_decay": 0.0,
    "grad_clip": 0.5,
    "stride": 1,
}

AUTOFORMER_DEFAULTS: dict[str, Any] = {
    "seq_len": 672,
    "label_len": 288,
    "pred_len": 288,
    "batch_size": 128,
    "epochs": 120,
    "lr": 3e-4,
    "moving_avg": 25,
    "d_model": 256,
    "n_heads": 8,
    "d_ff": 512,
    "dropout": 0.1,
    "e_layers": 2,
    "d_layers": 2,
    "patience": 8,
    "lr_factor": 0.5,
    "lr_patience": 4,
    "weight_decay": 1e-4,
    "grad_clip": 0.5,
    "stride": 1,
}


class TrainingPlanError(ValueError):
    """可安全返回给 API 调用方的训练计划错误。"""

    def __init__(self, message: str, *, status_code: int = 400, code: str = "INVALID_PLAN"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class TrainingPlan:
    equipment_code: str
    meas_code: str
    model_info_id: str
    model_type: str
    source: str
    freq: str
    days_back: int
    data_path: str
    train_params: dict[str, Any]
    execution_mode: str

    @property
    def side_effects(self) -> dict[str, bool]:
        platform = self.execution_mode == "platform"
        return {
            "local_artifacts": True,
            "sql_status_updates": platform,
            "image_upload": platform,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": PLAN_VERSION,
            "equipment_code": self.equipment_code,
            "meas_code": self.meas_code,
            "model_info_id": self.model_info_id,
            "model_type": self.model_type,
            "source": self.source,
            "freq": self.freq,
            "days_back": self.days_back,
            "data_path": self.data_path,
            "train_params": self.train_params,
            "execution_mode": self.execution_mode,
            "side_effects": self.side_effects,
            "output_policy": {"exclusive_new_directory": True},
        }

    @property
    def plan_hash(self) -> str:
        canonical = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and not isinstance(raw, bool) and raw in {0, 1}:
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"true", "1", "yes", "y", "on"}:
            return True
        if value in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError


def _convert_param(name: str, raw: Any) -> Any:
    caster = PARAM_CASTERS.get(name)
    if caster is None:
        raise TrainingPlanError(f"不支持的训练参数: {name}", code="UNKNOWN_PARAMETER")
    try:
        if caster is bool:
            return _parse_bool(raw)
        if caster is int:
            if isinstance(raw, bool) or isinstance(raw, float):
                raise ValueError
            value = int(raw)
            if isinstance(raw, str) and str(value) != raw.strip():
                raise ValueError
            return value
        if caster is float:
            if isinstance(raw, bool):
                raise ValueError
            return float(raw)
        return str(raw).strip()
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingPlanError(f"训练参数[{name}]的值无效", code="INVALID_PARAMETER") from exc


def _normalize_config_params(params: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, value in params.items():
        normalized[name] = _convert_param(str(name), value)
    return normalized


def _normalize_overrides(param_items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_name, value in param_items:
        name = str(raw_name)
        if name in normalized:
            raise TrainingPlanError(f"训练参数重复: {name}", code="DUPLICATE_PARAMETER")
        normalized[name] = _convert_param(name, value)
    return normalized


def _require_range(params: Mapping[str, Any], name: str, minimum: float, maximum: float) -> None:
    value = params[name]
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise TrainingPlanError(
            f"训练参数[{name}]必须在 {minimum:g}..{maximum:g} 范围内",
            code="PARAMETER_OUT_OF_RANGE",
        )


def _validate_train_params(params: Mapping[str, Any]) -> None:
    for name in ("seq_len", "label_len", "pred_len", "stride"):
        _require_range(params, name, 1, 10000)
    _require_range(params, "epochs", 1, 200)
    _require_range(params, "batch_size", 1, 256)
    _require_range(params, "d_model", 8, 1024)
    _require_range(params, "n_heads", 1, 32)
    _require_range(params, "d_ff", 8, 4096)
    _require_range(params, "e_layers", 1, 12)
    _require_range(params, "d_layers", 1, 12)
    _require_range(params, "lr", 1e-12, 1)
    _require_range(params, "patience", 1, 200)
    _require_range(params, "lr_patience", 1, 200)
    _require_range(params, "weight_decay", 0, 1)
    _require_range(params, "grad_clip", 1e-12, 1000)
    if "moving_avg" in params:
        _require_range(params, "moving_avg", 1, 10000)
        if params["moving_avg"] % 2 == 0:
            raise TrainingPlanError("训练参数[moving_avg]必须为奇数", code="INVALID_PARAMETER")
    if not 0 <= params["dropout"] < 1:
        raise TrainingPlanError(
            "训练参数[dropout]必须在 0（含）到 1（不含）之间",
            code="PARAMETER_OUT_OF_RANGE",
        )
    if not 0 < params["lr_factor"] < 1:
        raise TrainingPlanError(
            "训练参数[lr_factor]必须在 0 到 1 之间",
            code="PARAMETER_OUT_OF_RANGE",
        )
    if params["d_model"] % params["n_heads"] != 0:
        raise TrainingPlanError(
            "训练参数[d_model]必须能被[n_heads]整除",
            code="INCOMPATIBLE_PARAMETERS",
        )
    if "attn_type" in params and params["attn_type"] not in {"prob", "full"}:
        raise TrainingPlanError("训练参数[attn_type]仅支持 prob 或 full", code="INVALID_PARAMETER")
    try:
        validate_window_params(
            params["seq_len"], params["label_len"], params["pred_len"], params["stride"]
        )
    except ValueError as exc:
        raise TrainingPlanError(str(exc), code="INVALID_WINDOW_PARAMETERS") from exc


def validate_model_info_id(model_info_id: str) -> str:
    if not MODEL_INFO_ID_PATTERN.fullmatch(model_info_id):
        raise TrainingPlanError(
            "ModelInfoID 必须为 1-128 位 ASCII 字母、数字、点、下划线或连字符，且首位为字母或数字",
            code="INVALID_MODEL_INFO_ID",
        )
    return model_info_id


def build_training_plan(
    *,
    equipment_code: str,
    meas_code: str,
    model_info_id: str,
    requested_model_type: str | None,
    param_items: Iterable[tuple[str, Any]],
    requested_source: str,
    execution_mode: str,
    model_config: Mapping[str, Any] | None,
) -> TrainingPlan:
    validate_model_info_id(model_info_id)
    if model_config is None:
        raise TrainingPlanError(
            f"未找到设备[{equipment_code}]参数[{meas_code}]的模型配置",
            status_code=404,
            code="SCENARIO_NOT_FOUND",
        )

    model_type = str(requested_model_type or model_config.get("model_type", "informer"))
    model_type = model_type.strip().lower()
    if model_type not in API_TRAIN_MODEL_TYPES:
        raise TrainingPlanError(f"不支持的模型类型: {model_type}", code="UNSUPPORTED_MODEL_TYPE")

    source = str(requested_source).strip().lower()
    if source not in ALLOWED_DATA_SOURCES:
        raise TrainingPlanError(f"不支持的数据源: {source}", code="UNSUPPORTED_DATA_SOURCE")
    if model_type == "autoformer" and source != "csv":
        raise TrainingPlanError("Autoformer 目前仅支持 CSV 数据源", code="UNSUPPORTED_DATA_SOURCE")

    mode = str(execution_mode).strip().lower()
    if mode not in ALLOWED_EXECUTION_MODES:
        raise TrainingPlanError(f"不支持的执行模式: {mode}", code="INVALID_EXECUTION_MODE")

    defaults = INFORMER_DEFAULTS if model_type == "informer" else AUTOFORMER_DEFAULTS
    configured = _normalize_config_params(model_config.get("train_params", {}) or {})
    overrides = _normalize_overrides(param_items)
    unsupported = (set(configured) | set(overrides)) - MODEL_TRAIN_PARAM_NAMES[model_type]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TrainingPlanError(
            f"{model_type} 不支持训练参数: {names}", code="UNSUPPORTED_MODEL_PARAMETER"
        )
    params = {**defaults, **configured, **overrides}
    _validate_train_params(params)

    freq = str(model_config.get("freq", "15min")).strip()
    if not freq:
        raise TrainingPlanError("训练频率不能为空", code="INVALID_FREQUENCY")
    try:
        days_back = int(model_config.get("days_back", 365))
    except (TypeError, ValueError) as exc:
        raise TrainingPlanError("days_back 无效", code="INVALID_DAYS_BACK") from exc
    if not 1 <= days_back <= 3650:
        raise TrainingPlanError("days_back 必须在 1..3650 范围内", code="INVALID_DAYS_BACK")

    data_path = str(model_config.get("data_path", "")).strip()
    if source == "csv" and not data_path:
        raise TrainingPlanError("CSV 数据源缺少 data_path", code="MISSING_DATA_PATH")
    if source == "csv":
        try:
            resolve_data_path(data_path)
        except ValueError as exc:
            raise TrainingPlanError(str(exc), code="UNSAFE_DATA_PATH") from exc

    return TrainingPlan(
        equipment_code=equipment_code,
        meas_code=meas_code,
        model_info_id=model_info_id,
        model_type=model_type,
        source=source,
        freq=freq,
        days_back=days_back,
        data_path=data_path,
        train_params=params,
        execution_mode=mode,
    )
