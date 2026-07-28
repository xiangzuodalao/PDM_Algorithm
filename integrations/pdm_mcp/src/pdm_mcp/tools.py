"""PdM MCP 工具的业务实现。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import PurePosixPath
from typing import Any, Literal

from pdm_mcp.client import PdmApiClient
from pdm_mcp.config import Settings
from pdm_mcp.errors import PdmToolError
from pdm_mcp.preview_store import PreviewStore
from pdm_mcp.schemas import DataSource, ExecutionMode, TrainParams

MODEL_INFO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PLAN_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TRAIN_MODEL_TYPES = frozenset({"informer", "autoformer"})
PREDICT_MODEL_TYPES = frozenset({"informer", "autoformer", "testmodel"})
STATUS_VALUES = frozenset({"reserved", "running", "succeeded", "failed", "interrupted"})
TRAIN_PARAM_NAMES = frozenset(TrainParams.model_fields)
DATA_QUALITY_NAMES = (
    "input_rows",
    "filtered_rows",
    "dropped_invalid_time_rows",
    "dropped_invalid_value_rows",
    "valid_rows_before_resample",
    "aggregated_rows",
    "rows_after_resample",
)


def _invalid_response(message: str = "PdM API 响应结构无效") -> PdmToolError:
    return PdmToolError("PDM_INVALID_RESPONSE", message)


def _require_nonempty(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise PdmToolError("PDM_INVALID_ARGUMENT", f"{name} 不能为空")
    if len(normalized) > 256 or "\x00" in normalized:
        raise PdmToolError("PDM_INVALID_ARGUMENT", f"{name} 无效")
    return normalized


def _validate_model_info_id(value: str) -> str:
    if not MODEL_INFO_ID_PATTERN.fullmatch(value):
        raise PdmToolError(
            "PDM_INVALID_ARGUMENT",
            "model_info_id 必须为 1-128 位字母、数字、点、下划线或连字符",
        )
    return value


def _validate_model_type(value: str | None, allowed: frozenset[str]) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise PdmToolError("PDM_INVALID_ARGUMENT", "model_type 不受支持")
    return normalized


def _safe_csv_path(value: str | None) -> str:
    if value is None:
        raise PdmToolError("PDM_UNSAFE_DATA_PATH", "CSV 数据源必须提供 data_path")
    raw = value.strip()
    if not raw or "\x00" in raw or "\\" in raw:
        raise PdmToolError("PDM_UNSAFE_DATA_PATH", "data_path 必须是 data/ 下的相对路径")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "data":
        raise PdmToolError("PDM_UNSAFE_DATA_PATH", "data_path 必须是 data/ 下的相对路径")
    return path.as_posix()


def _require_int(mapping: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _invalid_response()
    return value


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _sanitize_train_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(key not in TRAIN_PARAM_NAMES for key in value):
        raise _invalid_response()
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, bool, int)) or isinstance(item, float) and math.isfinite(item):
            sanitized[key] = item
        else:
            raise _invalid_response()
    return sanitized


def _training_outcome_unknown() -> PdmToolError:
    return PdmToolError(
        "PDM_TRAIN_OUTCOME_UNKNOWN",
        "训练请求已提交，但返回结果无法确认；请查询训练状态，禁止自动重试",
    )


def _sample_indices(length: int, maximum: int) -> list[int]:
    if length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [0]
    return [(index * (length - 1)) // (maximum - 1) for index in range(maximum)]


class PdmTools:
    def __init__(
        self,
        settings: Settings,
        client: PdmApiClient,
        preview_store: PreviewStore | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.preview_store = preview_store or PreviewStore(settings.preview_ttl_seconds)

    async def health_check(self) -> dict[str, Any]:
        payload = await self.client.request("GET", "/healthz")
        status = payload.get("status")
        healthy = status in {"ok", "healthy"}
        if not isinstance(status, str):
            raise _invalid_response()
        return {"healthy": healthy, "status": status}

    async def list_models(
        self,
        equipment_code: str | None = None,
        meas_code: str | None = None,
        model_type: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        model_type = _validate_model_type(model_type, PREDICT_MODEL_TYPES)
        source_normalized = source.strip().lower() if source is not None else None
        if source_normalized is not None and source_normalized not in {
            "db",
            "postgres",
            "sqlserver",
            "csv",
        }:
            raise PdmToolError("PDM_INVALID_ARGUMENT", "source 不受支持")
        equipment = equipment_code.strip() if equipment_code is not None else None
        measurement = meas_code.strip() if meas_code is not None else None
        payload = await self.client.request("GET", "/measPredict/models")
        models = payload.get("models")
        if not isinstance(models, list):
            raise _invalid_response()
        result: list[dict[str, Any]] = []
        for item in models:
            if not isinstance(item, dict):
                raise _invalid_response()
            normalized = {
                "equipment_code": item.get("equipment_code"),
                "meas_code": item.get("meas_code"),
                "model_type": item.get("model_type"),
                "freq": item.get("freq"),
                "source": item.get("source"),
            }
            if not all(isinstance(normalized[key], str) for key in ("equipment_code", "meas_code")):
                raise _invalid_response()
            if equipment is not None and normalized["equipment_code"] != equipment:
                continue
            if measurement is not None and normalized["meas_code"] != measurement:
                continue
            if model_type is not None and str(normalized["model_type"]).lower() != model_type:
                continue
            if (
                source_normalized is not None
                and str(normalized["source"]).lower() != source_normalized
            ):
                continue
            result.append(normalized)
        return {"count": len(result), "models": result}

    async def check_training_data(
        self,
        equipment_code: str,
        meas_code: str,
        source: DataSource,
        seq_len: int = 672,
        label_len: int = 192,
        pred_len: int = 288,
        stride: int = 1,
        freq: str = "15min",
        days_back: int = 365,
        data_path: str | None = None,
    ) -> dict[str, Any]:
        equipment = _require_nonempty("equipment_code", equipment_code)
        measurement = _require_nonempty("meas_code", meas_code)
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (seq_len, label_len, pred_len, stride, days_back)
        ):
            raise PdmToolError("PDM_INVALID_ARGUMENT", "窗口和 days_back 必须是整数")
        if not all(1 <= value <= 10000 for value in (seq_len, label_len, pred_len, stride)):
            raise PdmToolError("PDM_INVALID_ARGUMENT", "窗口参数必须在 1..10000 范围内")
        if not 1 <= days_back <= 3650:
            raise PdmToolError("PDM_INVALID_ARGUMENT", "days_back 必须在 1..3650 范围内")
        if label_len > seq_len:
            raise PdmToolError("PDM_INVALID_ARGUMENT", "label_len 不能大于 seq_len")
        normalized_freq = _require_nonempty("freq", freq)
        if source == "csv":
            safe_path: str | None = _safe_csv_path(data_path)
        else:
            if data_path:
                raise PdmToolError("PDM_INVALID_ARGUMENT", "data_path 仅适用于 CSV 数据源")
            safe_path = None
        request_body: dict[str, Any] = {
            "EquipmentCode": equipment,
            "MeasCode": measurement,
            "SeqLen": seq_len,
            "LabelLen": label_len,
            "PredLen": pred_len,
            "Stride": stride,
            "Freq": normalized_freq,
            "DaysBack": days_back,
            "Source": source,
        }
        if safe_path is not None:
            request_body["DataPath"] = safe_path
        payload = await self.client.request(
            "POST", "/measPredict/checkData", json_body=request_body
        )
        return self._normalize_data_check(payload)

    @staticmethod
    def _normalize_data_check(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise _invalid_response()
        usable_raw = payload.get("usable", data.get("Usable"))
        if not isinstance(usable_raw, bool):
            raise _invalid_response()
        result = {
            "usable": usable_raw,
            "raw_data_volume": _require_int(data, "RawDataVolume"),
            "total_data_volume": _require_int(data, "TotalDataVolume"),
            "effective_sample_count": _require_int(data, "EffectiveSampleCount"),
            "required_sample_count": _require_int(data, "RequiredSampleCount"),
            "missing_sample_count": _require_int(data, "MissingSampleCount"),
            "train_window_count": _require_int(data, "TrainWindowCount"),
            "validation_window_count": _require_int(data, "ValidationWindowCount"),
            "test_window_count": _require_int(data, "TestWindowCount"),
        }
        quality = data.get("DataQuality")
        if quality is not None:
            if not isinstance(quality, dict):
                raise _invalid_response()
            result["data_quality"] = {key: _require_int(quality, key) for key in DATA_QUALITY_NAMES}
        return result

    async def predict(
        self,
        equipment_code: str,
        meas_code: str,
        model_info_id: str | None = None,
        model_type: str | None = None,
        max_points: int = 100,
    ) -> dict[str, Any]:
        equipment = _require_nonempty("equipment_code", equipment_code)
        measurement = _require_nonempty("meas_code", meas_code)
        if not 1 <= max_points <= 100:
            raise PdmToolError("PDM_INVALID_ARGUMENT", "max_points 必须在 1..100 范围内")
        if model_info_id is not None:
            model_info_id = _validate_model_info_id(model_info_id)
        model_type = _validate_model_type(model_type, PREDICT_MODEL_TYPES)
        request_body: dict[str, Any] = {
            "EquipmentCode": equipment,
            "MeasCode": measurement,
        }
        if model_info_id is not None:
            request_body["ModelInfoID"] = model_info_id
        if model_type is not None:
            request_body["ModelType"] = model_type
        payload = await self.client.request("POST", "/measPredict/predict", json_body=request_body)
        response = payload.get("response")
        if payload.get("success") is not True or not isinstance(response, dict):
            raise _invalid_response()
        values = response.get("Values")
        if not isinstance(values, list):
            raise _invalid_response()
        normalized: list[dict[str, Any]] = []
        numeric_values: list[float] = []
        for item in values:
            if not isinstance(item, dict) or not isinstance(item.get("Value"), str):
                raise _invalid_response()
            number = _finite_float(item["Value"])
            if number is not None:
                numeric_values.append(number)
            normalized.append(
                {
                    "seq": item.get("Seq"),
                    "timestamp": item.get("XAxis"),
                    "value": item["Value"],
                    "unit": item.get("Unit") or "",
                }
            )
        indices = _sample_indices(len(normalized), max_points)
        statistics: dict[str, float | int | None] = {
            "numeric_point_count": len(numeric_values),
            "min": None,
            "max": None,
            "mean": None,
            "first": None,
            "last": None,
            "delta": None,
        }
        if numeric_values:
            statistics.update(
                {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                    "first": numeric_values[0],
                    "last": numeric_values[-1],
                    "delta": numeric_values[-1] - numeric_values[0],
                }
            )
        return {
            "success": True,
            "message": "预测成功",
            "history_sample_count": _require_int(response, "SampleCount"),
            "remark": response.get("Remark") or "",
            "total_points": len(normalized),
            "returned_points": len(indices),
            "truncated": len(normalized) > len(indices),
            "statistics": statistics,
            "values": [normalized[index] for index in indices],
        }

    async def prepare_training(
        self,
        equipment_code: str,
        meas_code: str,
        model_info_id: str,
        data_source: DataSource,
        train_params: TrainParams | dict[str, Any] | None = None,
        model_type: Literal["informer", "autoformer"] | None = None,
        execution_mode: ExecutionMode = "local_only",
    ) -> dict[str, Any]:
        equipment = _require_nonempty("equipment_code", equipment_code)
        measurement = _require_nonempty("meas_code", meas_code)
        model_id = _validate_model_info_id(model_info_id)
        normalized_model_type = _validate_model_type(model_type, TRAIN_MODEL_TYPES)
        try:
            params_model = (
                TrainParams() if train_params is None else TrainParams.model_validate(train_params)
            )
        except ValueError as exc:
            raise PdmToolError("PDM_INVALID_ARGUMENT", "train_params 无效") from exc
        overrides = params_model.model_dump(exclude_none=True)
        request_body: dict[str, Any] = {
            "EquipmentCode": equipment,
            "MeasCode": measurement,
            "ModelInfoID": model_id,
            "ParamArr": [
                {"FieldName": name, "CurValue": value} for name, value in overrides.items()
            ],
            "DataSource": data_source,
            "ExecutionMode": execution_mode,
        }
        if normalized_model_type is not None:
            request_body["ModelType"] = normalized_model_type
        preview = await self.client.request(
            "POST", "/measPredict/train/preview", json_body=request_body
        )
        plan_hash = preview.get("plan_hash")
        plan = preview.get("plan")
        if (
            preview.get("success") is not True
            or not isinstance(plan_hash, str)
            or not PLAN_HASH_PATTERN.fullmatch(plan_hash)
            or not isinstance(plan, dict)
        ):
            raise _invalid_response()
        self._verify_plan_identity(
            plan,
            plan_hash,
            equipment,
            measurement,
            model_id,
            data_source,
            normalized_model_type,
            execution_mode,
        )
        plan_for_output = self._sanitize_plan(plan)
        source = plan.get("source")
        data_path = plan.get("data_path")
        if source == "csv":
            plan_for_output["data_path"] = _safe_csv_path(
                data_path if isinstance(data_path, str) else None
            )
        else:
            plan_for_output["data_path"] = ""
        effective = plan_for_output["train_params"]
        data_check = await self.check_training_data(
            equipment,
            measurement,
            source=source,
            seq_len=_require_int(effective, "seq_len", 1),
            label_len=_require_int(effective, "label_len", 1),
            pred_len=_require_int(effective, "pred_len", 1),
            stride=_require_int(effective, "stride", 1),
            freq=_require_nonempty("plan.freq", str(plan.get("freq", ""))),
            days_back=_require_int(plan, "days_back", 1),
            data_path=plan_for_output["data_path"] if source == "csv" else None,
        )
        usable = data_check["usable"] and all(
            data_check[key] > 0
            for key in (
                "train_window_count",
                "validation_window_count",
                "test_window_count",
            )
        )
        preview_id: str | None = None
        expires_at: str | None = None
        if usable:
            request_body["ExpectedPlanHash"] = plan_hash
            preview_id, expires_at = await self.preview_store.create(
                {
                    "request": request_body,
                    "plan_hash": plan_hash,
                    "plan": plan_for_output,
                }
            )
        return {
            "success": True,
            "preview_id": preview_id,
            "expires_at": expires_at,
            "plan_hash": plan_hash,
            "plan": plan_for_output,
            "usable": usable,
            "data_check": data_check,
        }

    @staticmethod
    def _verify_plan_identity(
        plan: dict[str, Any],
        plan_hash: str,
        equipment_code: str,
        meas_code: str,
        model_info_id: str,
        data_source: str,
        requested_model_type: str | None,
        execution_mode: str,
    ) -> None:
        expected = {
            "equipment_code": equipment_code,
            "meas_code": meas_code,
            "model_info_id": model_info_id,
            "source": data_source,
            "execution_mode": execution_mode,
        }
        if any(plan.get(key) != value for key, value in expected.items()):
            raise _invalid_response("PdM API 返回了不匹配的训练计划")
        if plan.get("model_type") not in TRAIN_MODEL_TYPES:
            raise _invalid_response()
        if requested_model_type is not None and plan.get("model_type") != requested_model_type:
            raise _invalid_response("PdM API 返回了不匹配的训练计划")
        try:
            canonical = json.dumps(
                plan,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise _invalid_response() from exc
        if hashlib.sha256(canonical.encode()).hexdigest() != plan_hash:
            raise _invalid_response("训练计划与 plan_hash 不匹配")

    @staticmethod
    def _sanitize_plan(plan: dict[str, Any]) -> dict[str, Any]:
        side_effects = plan.get("side_effects")
        output_policy = plan.get("output_policy")
        if not isinstance(side_effects, dict) or not isinstance(output_policy, dict):
            raise _invalid_response()
        sanitized_side_effects: dict[str, bool] = {}
        for key in ("local_artifacts", "sql_status_updates", "image_upload"):
            value = side_effects.get(key)
            if not isinstance(value, bool):
                raise _invalid_response()
            sanitized_side_effects[key] = value
        exclusive = output_policy.get("exclusive_new_directory")
        if not isinstance(exclusive, bool):
            raise _invalid_response()
        version = plan.get("version")
        days_back = plan.get("days_back")
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise _invalid_response()
        if isinstance(days_back, bool) or not isinstance(days_back, int):
            raise _invalid_response()
        string_fields = (
            "equipment_code",
            "meas_code",
            "model_info_id",
            "model_type",
            "source",
            "freq",
            "data_path",
            "execution_mode",
        )
        if any(not isinstance(plan.get(key), str) for key in string_fields):
            raise _invalid_response()
        platform = plan["execution_mode"] == "platform"
        if (
            sanitized_side_effects["local_artifacts"] is not True
            or sanitized_side_effects["sql_status_updates"] is not platform
            or sanitized_side_effects["image_upload"] is not platform
            or exclusive is not True
        ):
            raise _invalid_response("PdM API 返回了不安全的副作用计划")
        return {
            "version": version,
            "equipment_code": plan["equipment_code"],
            "meas_code": plan["meas_code"],
            "model_info_id": plan["model_info_id"],
            "model_type": plan["model_type"],
            "source": plan["source"],
            "freq": plan["freq"],
            "days_back": days_back,
            "data_path": plan["data_path"],
            "train_params": _sanitize_train_params(plan.get("train_params")),
            "execution_mode": plan["execution_mode"],
            "side_effects": sanitized_side_effects,
            "output_policy": {"exclusive_new_directory": exclusive},
        }

    async def train_model(self, preview_id: str) -> dict[str, Any]:
        if not self.settings.enable_train:
            raise PdmToolError(
                "PDM_TRAIN_DISABLED",
                "训练工具未启用；管理员需设置 PDM_ENABLE_TRAIN=true",
            )
        normalized_id = _require_nonempty("preview_id", preview_id)
        try:
            # UUID 规范化也可拦截意外的超长/控制字符输入。
            canonical_id = str(uuid.UUID(normalized_id))
            if canonical_id != normalized_id.lower():
                raise ValueError
        except ValueError as exc:
            raise PdmToolError("PDM_INVALID_ARGUMENT", "preview_id 无效") from exc
        entry = await self.preview_store.consume(canonical_id)
        request_body = entry.get("request")
        if not isinstance(request_body, dict):
            raise _invalid_response()
        result = await self.client.request(
            "POST",
            "/measPredict/train",
            json_body=request_body,
            training_dispatch=True,
        )
        expected_model_id = request_body.get("ModelInfoID")
        if (
            result.get("success") is not True
            or result.get("status") != "succeeded"
            or result.get("model_info_id") != expected_model_id
        ):
            raise _training_outcome_unknown()
        return {
            "success": True,
            "message": "训练完成",
            "model_info_id": expected_model_id,
            "status": "succeeded",
            "best_val": _finite_float(result.get("best_val")),
            "test_loss": _finite_float(result.get("test_loss")),
            "plan_hash": entry.get("plan_hash"),
            "train_params": entry["plan"]["train_params"],
        }

    async def get_training_status(
        self,
        equipment_code: str,
        meas_code: str,
        model_info_id: str,
        model_type: Literal["informer", "autoformer"] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "EquipmentCode": _require_nonempty("equipment_code", equipment_code),
            "MeasCode": _require_nonempty("meas_code", meas_code),
            "ModelInfoID": _validate_model_info_id(model_info_id),
        }
        normalized_model_type = _validate_model_type(model_type, TRAIN_MODEL_TYPES)
        if normalized_model_type is not None:
            params["ModelType"] = normalized_model_type
        result = await self.client.request("GET", "/measPredict/train/status", params=params)
        status = result.get("status")
        if (
            result.get("success") is not True
            or status not in STATUS_VALUES
            or result.get("model_info_id") != params["ModelInfoID"]
        ):
            raise _invalid_response()
        error_code = result.get("error_code")
        if error_code is not None and (
            not isinstance(error_code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", error_code)
        ):
            error_code = "UNKNOWN_FAILURE"
        return {
            "success": True,
            "message": "获取训练状态成功",
            "model_info_id": params["ModelInfoID"],
            "status": status,
            "created_at": result.get("created_at"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "best_val": _finite_float(result.get("best_val")),
            "test_loss": _finite_float(result.get("test_loss")),
            "error_code": error_code,
        }
