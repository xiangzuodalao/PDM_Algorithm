from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from yaml.nodes import MappingNode, ScalarNode


DEFAULTABLE_FIELDS = frozenset({"model_type", "freq", "days_back", "source"})
CONFIG_FIELDS = DEFAULTABLE_FIELDS | {"data_path", "train_params"}
TRAIN_PARAM_FIELDS = frozenset(
    {
        "seq_len",
        "label_len",
        "pred_len",
        "stride",
        "batch_size",
        "epochs",
        "lr",
        "d_model",
        "n_heads",
        "d_ff",
        "dropout",
        "e_layers",
        "d_layers",
        "attn_type",
        "distil",
        "patience",
        "lr_factor",
        "lr_patience",
        "weight_decay",
        "grad_clip",
        "moving_avg",
    }
)
DB_FIXTURE_COLUMNS = frozenset({"equipment_code", "meas_code", "value", "unit", "collect_time"})
MVP_MODEL_ALLOWLIST = frozenset({"informer", "autoformer"})


class OnboardError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OnboardError("E_CANDIDATE_SCHEMA", f"{field} 必须是映射")
    return value


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OnboardError("E_CANDIDATE_SCHEMA", f"{field} 必须是非空字符串")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OnboardError("E_CANDIDATE_SCHEMA", f"{field} 必须是非布尔正整数")
    return value


def _load_structured_file(path: Path) -> Any:
    if not path.exists():
        raise OnboardError("E_DATA_FILE_MISSING", f"文件不存在: {path.name}")
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise OnboardError("E_CANDIDATE_SCHEMA", f"无法解析文件: {path.name}") from exc


def _assignment_value(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return node.value
    raise OnboardError("E_MODEL_UNSUPPORTED", f"未找到模型契约常量: {name}")


def _literal_collection(node: ast.expr) -> Any:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"frozenset", "set", "tuple", "list"} and len(node.args) == 1:
            return ast.literal_eval(node.args[0])
    return ast.literal_eval(node)


def _read_python_tree(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise OnboardError("E_MODEL_UNSUPPORTED", f"无法解析模型契约源码: {path.name}") from exc


def _trainer_target_imports(repo_root: Path, module_name: str, attribute: str) -> bool:
    environment = os.environ.copy()
    source_root = str(repo_root / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing_pythonpath else source_root + os.pathsep + existing_pythonpath
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    probe = (
        "from importlib import import_module; import sys; "
        "target = getattr(import_module(sys.argv[1]), sys.argv[2]); "
        "assert callable(target), 'trainer target is not callable'"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe, module_name, attribute],
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


@lru_cache(maxsize=4)
def discover_model_support(repo_root: Path) -> dict[str, Any]:
    registry_path = repo_root / "src" / "valeo_pdm" / "training" / "registry.py"
    router_path = repo_root / "src" / "valeo_pdm" / "api" / "router.py"
    predict_path = repo_root / "src" / "valeo_pdm" / "transformer" / "predict.py"
    registry_tree = _read_python_tree(registry_path)
    router_tree = _read_python_tree(router_path)
    predict_tree = _read_python_tree(predict_path)

    raw_imports = _literal_collection(_assignment_value(registry_tree, "TRAINER_IMPORTS"))
    trainer_imports = _mapping(raw_imports, "TRAINER_IMPORTS")
    trainer_models: set[str] = set()
    missing_trainers: dict[str, str] = {}
    for model_type, target in trainer_imports.items():
        if (
            not isinstance(model_type, str)
            or not isinstance(target, (tuple, list))
            or len(target) != 2
        ):
            continue
        module_name, attribute = target
        module_path = repo_root / "src" / Path(*str(module_name).split(".")).with_suffix(".py")
        if not module_path.exists():
            missing_trainers[model_type] = str(module_name)
            continue
        module_tree = _read_python_tree(module_path)
        exports = {
            node.name
            for node in module_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if str(attribute) in exports and _trainer_target_imports(
            repo_root, str(module_name), str(attribute)
        ):
            trainer_models.add(model_type)
        else:
            missing_trainers[model_type] = f"{module_name}:{attribute} import failed"

    predict_node = _assignment_value(router_tree, "MODEL_PREDICT_FUNCS")
    if not isinstance(predict_node, ast.Dict):
        raise OnboardError("E_MODEL_UNSUPPORTED", "MODEL_PREDICT_FUNCS 必须是字典")
    predictor_exports = {
        node.name
        for node in predict_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    predictor_models: set[str] = set()
    missing_predictors: dict[str, str] = {}
    for key, value in zip(predict_node.keys, predict_node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            continue
        if isinstance(value, ast.Name) and value.id in predictor_exports:
            predictor_models.add(key.value)
        else:
            missing_predictors[key.value] = getattr(value, "id", "invalid mapping")
    api_models = set(_literal_collection(_assignment_value(router_tree, "API_TRAIN_MODEL_TYPES")))
    contract_intersection = trainer_models & predictor_models & api_models
    supported = contract_intersection & MVP_MODEL_ALLOWLIST
    return {
        "supported": sorted(supported),
        "contract_intersection": sorted(contract_intersection),
        "trainer_models": sorted(trainer_models),
        "predictor_models": sorted(predictor_models),
        "api_models": sorted(api_models),
        "missing_trainers": missing_trainers,
        "missing_predictors": missing_predictors,
    }


def _project_functions(repo_root: Path) -> tuple[Callable[..., Any], ...]:
    source_root = str(repo_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from valeo_pdm.transformer.config_resolution import (
        apply_defaults,
        get_model_block,
        resolve_model_config,
    )
    from valeo_pdm.transformer.data import (
        clean_and_resample_timeseries,
        require_usable_window_splits,
        validate_window_params,
    )

    return (
        apply_defaults,
        get_model_block,
        resolve_model_config,
        clean_and_resample_timeseries,
        require_usable_window_splits,
        validate_window_params,
    )


def _resolve_local_path(
    repo_root: Path,
    raw_path: str | Path,
    *,
    allow_external_data: bool,
) -> Path:
    source = Path(raw_path).expanduser()
    resolved = source.resolve() if source.is_absolute() else (repo_root / source).resolve()
    if not resolved.is_relative_to(repo_root) and not allow_external_data:
        raise OnboardError(
            "E_EXTERNAL_DATA_PATH",
            "仓库外数据路径需要显式只读授权",
            path_name=resolved.name,
        )
    if not resolved.is_file():
        raise OnboardError("E_DATA_FILE_MISSING", f"数据文件不存在: {resolved.name}")
    return resolved


def _load_timeseries_rows(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "rows" in payload:
                payload = payload["rows"]
            if not isinstance(payload, list):
                raise ValueError("JSON 时序 fixture 必须是行数组或包含 rows")
            return pd.DataFrame(payload)
        return pd.read_csv(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise OnboardError("E_DATA_SCHEMA", f"无法读取时序 fixture: {path.name}") from exc


def _validate_db_fixture_columns(frame: pd.DataFrame) -> None:
    available = {str(column).lower() for column in frame.columns}
    if not DB_FIXTURE_COLUMNS.issubset(available):
        raise OnboardError(
            "E_DATA_SCHEMA",
            "DB 时序 fixture 必须包含 equipment_code/meas_code/value/unit/collect_time",
        )


def _validate_db_models_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"models"}:
        raise OnboardError(
            "E_CANDIDATE_SCHEMA",
            "DB 模型 fixture 顶层必须且只能包含 models",
        )
    models = _mapping(payload["models"], "db fixture models")
    for equipment_code, equipment in models.items():
        if not isinstance(equipment_code, str) or not isinstance(equipment, dict):
            raise OnboardError("E_CANDIDATE_SCHEMA", "DB 模型 fixture 的设备层级无效")
        for meas_code, config in equipment.items():
            if not isinstance(meas_code, str) or not isinstance(config, dict):
                raise OnboardError("E_CANDIDATE_SCHEMA", "DB 模型 fixture 的测点层级无效")
    return models


def _find_mapping_value(node: MappingNode, key_name: str) -> tuple[ScalarNode, Any] | None:
    for key_node, value_node in node.value:
        if isinstance(key_node, ScalarNode) and key_node.value == key_name:
            return key_node, value_node
    return None


def _indented_yaml(value: object, spaces: int, newline: str) -> str:
    dumped = yaml.safe_dump(
        value,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=100,
    )
    lines = dumped.splitlines()
    return newline.join((" " * spaces) + line for line in lines) + newline


def insert_scenario_text(
    original_text: str,
    equipment_code: str,
    meas_code: str,
    config: dict[str, Any],
) -> str:
    try:
        root = yaml.compose(original_text)
    except yaml.YAMLError as exc:
        raise OnboardError("E_TEMP_VALIDATION", "registry YAML 无法解析") from exc
    if not isinstance(root, MappingNode):
        raise OnboardError("E_TEMP_VALIDATION", "registry 根节点必须是映射")
    models_entry = _find_mapping_value(root, "models")
    if models_entry is None or not isinstance(models_entry[1], MappingNode):
        raise OnboardError("E_TEMP_VALIDATION", "registry 缺少 models 映射")
    models_node = models_entry[1]
    equipment_entry = _find_mapping_value(models_node, equipment_code)
    lines = original_text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in original_text else "\n"

    if equipment_entry is not None:
        equipment_node = equipment_entry[1]
        if not isinstance(equipment_node, MappingNode):
            raise OnboardError("E_TEMP_VALIDATION", "已有设备配置必须是映射")
        start_line = equipment_entry[0].start_mark.line + 1
        boundary = len(lines)
        for index in range(start_line, len(lines)):
            content = lines[index].strip()
            if not content:
                continue
            indent = len(lines[index]) - len(lines[index].lstrip(" "))
            if indent <= 2:
                boundary = index
                break
        block = _indented_yaml({meas_code: config}, 4, newline)
    else:
        root_keys = [
            key_node.start_mark.line
            for key_node, _value_node in root.value
            if key_node.start_mark.line > models_entry[0].start_mark.line
        ]
        boundary = min(root_keys) if root_keys else len(lines)
        block = _indented_yaml({equipment_code: {meas_code: config}}, 2, newline)

    if boundary > 0 and lines[boundary - 1].strip():
        block = newline + block
    if boundary < len(lines) and lines[boundary].strip():
        block += newline
    return "".join(lines[:boundary]) + block + "".join(lines[boundary:])


def _validate_registry_file(
    path: Path,
    *,
    equipment_code: str,
    meas_code: str,
    candidate_config: dict[str, Any],
    db_models: dict[str, Any],
    expected_effective: dict[str, Any],
    repo_root: Path,
) -> None:
    try:
        registry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise OnboardError("E_TEMP_VALIDATION", "临时 registry 无法解析") from exc
    registry_map = _mapping(registry, "registry")
    defaults = _mapping(registry_map.get("defaults", {}), "defaults")
    yaml_models = _mapping(registry_map.get("models", {}), "models")
    _apply_defaults, get_model_block, resolve_model_config, *_rest = _project_functions(repo_root)
    written = get_model_block(yaml_models, equipment_code, meas_code)
    if written != candidate_config:
        raise OnboardError("E_TEMP_VALIDATION", "临时副本中的候选块与输入不一致")
    effective, source = resolve_model_config(
        defaults=defaults,
        yaml_models=yaml_models,
        db_models=db_models,
        equipment_code=equipment_code,
        meas_code=meas_code,
    )
    if source != "yaml" or effective != expected_effective:
        raise OnboardError("E_TEMP_VALIDATION", "临时副本的最终生效配置不一致")


def validate_temporary_copy(
    registry_path: Path,
    candidate_text: str,
    validator: Callable[[Path], None],
) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{registry_path.name}.",
            suffix=".tmp",
            dir=registry_path.parent,
            delete=False,
        ) as handle:
            handle.write(candidate_text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        validator(temp_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def atomic_replace_with_rollback(
    registry_path: Path,
    candidate_text: str,
    validator: Callable[[Path], None],
    *,
    expected_original: bytes | None = None,
) -> None:
    original = registry_path.read_bytes()
    if expected_original is not None and original != expected_original:
        raise OnboardError("E_REGISTRY_CHANGED", "registry 在预览后已变化，拒绝覆盖")

    lock_path = registry_path.with_name(f".{registry_path.name}.onboard.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise OnboardError("E_REGISTRY_CHANGED", "另一个场景写入正在进行") from exc

    def write_atomic(payload: bytes) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{registry_path.name}.",
                suffix=".tmp",
                dir=registry_path.parent,
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, registry_path)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    try:
        os.close(lock_fd)
        if registry_path.read_bytes() != original:
            raise OnboardError("E_REGISTRY_CHANGED", "registry 在写入前已变化，拒绝覆盖")

        candidate_bytes = candidate_text.encode("utf-8")
        temp_validation_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{registry_path.name}.",
                suffix=".tmp",
                dir=registry_path.parent,
                delete=False,
            ) as handle:
                handle.write(candidate_bytes)
                handle.flush()
                os.fsync(handle.fileno())
                temp_validation_path = Path(handle.name)
            validator(temp_validation_path)
            if registry_path.read_bytes() != original:
                raise OnboardError("E_REGISTRY_CHANGED", "registry 在写入前已变化，拒绝覆盖")
            os.replace(temp_validation_path, registry_path)
            temp_validation_path = None
        finally:
            if temp_validation_path is not None:
                temp_validation_path.unlink(missing_ok=True)

        try:
            validator(registry_path)
        except Exception as exc:
            write_atomic(original)
            raise OnboardError(
                "E_POST_WRITE_VALIDATION",
                "写入后验证失败，已恢复原 registry",
            ) from exc
    finally:
        lock_path.unlink(missing_ok=True)


def _unified_diff(registry_path: Path, original: str, candidate: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile=f"{registry_path.name} (current)",
            tofile=f"{registry_path.name} (candidate)",
            n=0,
        )
    )


def onboard_scenario(
    *,
    repo_root: str | Path,
    candidate_path: str | Path,
    db_models_path: str | Path | None = None,
    timeseries_fixture: str | Path | None = None,
    registry_path: str | Path | None = None,
    allow_external_data: bool = False,
    apply: bool = False,
    confirmed: bool = False,
    expected_diff_hash: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    registry = (
        (root / "configs" / "model_registry.yaml").resolve()
        if registry_path is None
        else Path(registry_path).resolve()
    )
    if not registry.is_relative_to(root):
        raise OnboardError("E_CANDIDATE_SCHEMA", "registry 必须位于目标仓库内")
    if not registry.is_file():
        raise OnboardError("E_DATA_FILE_MISSING", "model_registry.yaml 不存在")

    candidate_payload = _mapping(_load_structured_file(Path(candidate_path).resolve()), "candidate")
    if set(candidate_payload) - {"equipment_code", "meas_code", "config"}:
        raise OnboardError("E_CANDIDATE_SCHEMA", "candidate 包含未授权字段")
    equipment_code = _required_string(candidate_payload.get("equipment_code"), "equipment_code")
    meas_code = _required_string(candidate_payload.get("meas_code"), "meas_code")
    candidate_config = deepcopy(_mapping(candidate_payload.get("config"), "config"))
    if set(candidate_config) - CONFIG_FIELDS:
        raise OnboardError("E_CANDIDATE_SCHEMA", "config 包含未授权字段")
    train_params = deepcopy(_mapping(candidate_config.get("train_params"), "config.train_params"))
    if set(train_params) - TRAIN_PARAM_FIELDS:
        raise OnboardError("E_CANDIDATE_SCHEMA", "config.train_params 包含未授权字段")
    candidate_config["train_params"] = train_params
    for field in ("seq_len", "label_len", "pred_len"):
        if field not in train_params:
            raise OnboardError("E_CANDIDATE_SCHEMA", f"config.train_params.{field} 必须显式提供")

    try:
        registry_payload = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise OnboardError("E_TEMP_VALIDATION", "model_registry.yaml 无法解析") from exc
    registry_map = _mapping(registry_payload, "registry")
    defaults = _mapping(registry_map.get("defaults", {}), "defaults")
    yaml_models = _mapping(registry_map.get("models", {}), "models")

    db_status = "unknown"
    db_models: dict[str, Any] = {}
    if db_models_path is not None:
        db_payload = _mapping(_load_structured_file(Path(db_models_path).resolve()), "db fixture")
        db_models = _validate_db_models_fixture(db_payload)
        db_status = "verified"

    (
        apply_defaults,
        get_model_block,
        _resolve_model_config,
        clean_and_resample_timeseries,
        require_usable_window_splits,
        validate_window_params,
    ) = _project_functions(root)

    try:
        yaml_duplicate = get_model_block(yaml_models, equipment_code, meas_code)
    except TypeError as exc:
        raise OnboardError("E_TEMP_VALIDATION", "registry models 层级无效") from exc
    if yaml_duplicate is not None:
        raise OnboardError("E_DUPLICATE_YAML", "YAML 已存在同名设备+测点")
    if get_model_block(db_models, equipment_code, meas_code) is not None:
        raise OnboardError("E_DUPLICATE_DB", "DB fixture 已存在同名设备+测点")

    effective = apply_defaults(candidate_config, defaults)
    raw_model_type = _required_string(effective.get("model_type"), "model_type")
    raw_source = _required_string(effective.get("source"), "source")
    raw_freq = _required_string(effective.get("freq"), "freq")
    model_type = raw_model_type.lower()
    source = raw_source.lower()
    freq = raw_freq

    normalized_strings = {
        "model_type": model_type,
        "source": source,
        "freq": freq,
    }
    for field, normalized in normalized_strings.items():
        if field in candidate_config:
            candidate_config[field] = normalized
        elif effective.get(field) != normalized:
            raise OnboardError(
                "E_CANDIDATE_SCHEMA",
                f"registry defaults.{field} 必须使用规范化值",
            )
    effective = apply_defaults(candidate_config, defaults)
    _positive_int(effective.get("days_back"), "days_back")
    if source not in {"csv", "db", "postgres", "sqlserver"}:
        raise OnboardError("E_SOURCE_UNSUPPORTED", f"不支持的数据源: {source}")

    model_support = deepcopy(discover_model_support(root))
    if model_type not in model_support["supported"]:
        raise OnboardError(
            "E_MODEL_UNSUPPORTED",
            f"模型不在 Trainer/预测/API 支持交集: {model_type}",
            supported=model_support["supported"],
        )
    if model_type == "autoformer" and source != "csv":
        raise OnboardError("E_SOURCE_UNSUPPORTED", "Autoformer 仅支持 CSV 数据源")

    stride = train_params.get("stride", 1)
    try:
        validate_window_params(
            train_params["seq_len"],
            train_params["label_len"],
            train_params["pred_len"],
            stride,
        )
    except ValueError as exc:
        raise OnboardError("E_WINDOW_INVALID", str(exc)) from exc

    if source == "csv":
        if "data_path" not in candidate_config:
            raise OnboardError("E_DATA_PATH_REQUIRED", "CSV 场景必须显式提供 data_path")
        candidate_config["data_path"] = _required_string(
            candidate_config.get("data_path"), "data_path"
        )
        effective = apply_defaults(candidate_config, defaults)
        data_path = _resolve_local_path(
            root,
            candidate_config["data_path"],
            allow_external_data=allow_external_data,
        )
    else:
        if timeseries_fixture is None:
            raise OnboardError("E_DATA_FILE_MISSING", "DB 场景必须提供本地时序 fixture")
        data_path = _resolve_local_path(
            root,
            timeseries_fixture,
            allow_external_data=allow_external_data,
        )

    raw_rows = _load_timeseries_rows(data_path)
    if source != "csv":
        _validate_db_fixture_columns(raw_rows)
    try:
        series_df, quality = clean_and_resample_timeseries(
            raw_rows,
            freq,
            equipment_code=equipment_code,
            meas_code=meas_code,
        )
    except (TypeError, ValueError) as exc:
        raise OnboardError("E_DATA_SCHEMA", str(exc)) from exc
    if source != "csv" and quality["filtered_rows"] == 0:
        raise OnboardError("E_DATA_SCHEMA", "DB 时序 fixture 不含目标设备+测点行")
    try:
        window_counts = require_usable_window_splits(
            len(series_df),
            train_params["seq_len"],
            train_params["label_len"],
            train_params["pred_len"],
            stride,
        )
    except ValueError as exc:
        raise OnboardError("E_WINDOW_INSUFFICIENT", str(exc)) from exc

    try:
        original_bytes = registry.read_bytes()
        original_text = original_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OnboardError("E_TEMP_VALIDATION", "model_registry.yaml 无法读取") from exc
    candidate_text = insert_scenario_text(
        original_text,
        equipment_code,
        meas_code,
        candidate_config,
    )

    def validator(path: Path) -> None:
        _validate_registry_file(
            path,
            equipment_code=equipment_code,
            meas_code=meas_code,
            candidate_config=candidate_config,
            db_models=db_models,
            expected_effective=effective,
            repo_root=root,
        )

    validate_temporary_copy(registry, candidate_text, validator)
    diff = _unified_diff(registry, original_text, candidate_text)
    confirmation_payload = json.dumps(
        {
            "diff": diff,
            "effective_config": effective,
            "registry_sha256": hashlib.sha256(original_bytes).hexdigest(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    diff_hash = hashlib.sha256(confirmation_payload.encode("utf-8")).hexdigest()

    can_apply = db_status == "verified"
    warnings: list[str] = []
    if not can_apply:
        warnings.append("DB fixture 未提供，无法证明目标键在 DB 中不存在")
    if quality["dropped_invalid_time_rows"] or quality["dropped_invalid_value_rows"]:
        warnings.append("时序 fixture 含不可解析行，已丢弃且未插值")
    if apply:
        if not confirmed:
            raise OnboardError("E_CONFIRM_REQUIRED", "apply 必须附带 --confirmed")
        if not can_apply:
            raise OnboardError("E_DB_STATUS_UNKNOWN", "DB 状态未知时禁止写入 YAML")
        if expected_diff_hash != diff_hash:
            raise OnboardError("E_DIFF_MISMATCH", "待写入 diff 与用户确认版本不一致")
        atomic_replace_with_rollback(
            registry,
            candidate_text,
            validator,
            expected_original=original_bytes,
        )

    return {
        "status": "applied" if apply else "preview",
        "can_apply": can_apply,
        "db_status": db_status,
        "equipment_code": equipment_code,
        "meas_code": meas_code,
        "effective_config": effective,
        "model_support": model_support,
        "data_quality": quality,
        "window_counts": window_counts,
        "warnings": warnings,
        "diff": diff,
        "diff_hash": diff_hash,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="预览并安全新增 PDM 设备/测点场景")
    parser.add_argument("--repo-root", default=".", help="仓库根目录")
    parser.add_argument("--candidate", required=True, help="候选 YAML/JSON")
    parser.add_argument("--db-models", help="本地 DB 模型配置 fixture")
    parser.add_argument("--timeseries-fixture", help="DB 查询结果的本地 CSV/JSON fixture")
    parser.add_argument("--registry", help="registry 路径，默认 configs/model_registry.yaml")
    parser.add_argument("--allow-external-data", action="store_true", help="允许只读仓库外数据文件")
    parser.add_argument("--apply", action="store_true", help="验证后原子写入 registry")
    parser.add_argument("--confirmed", action="store_true", help="声明用户已确认展示的 diff")
    parser.add_argument("--expected-diff-hash", help="预览结果中已确认的 diff_hash")
    parser.add_argument("--format", choices=("json", "text"), default="text")
    return parser


def _print_result(result: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"状态: {result['status']} | 可写入: {result['can_apply']}")
    print("最终生效配置:")
    print(yaml.safe_dump(result["effective_config"], allow_unicode=True, sort_keys=False).rstrip())
    print(f"模型支持交集: {', '.join(result['model_support']['supported'])}")
    print(
        "清洗: "
        f"input={result['data_quality']['input_rows']} "
        f"filtered={result['data_quality']['filtered_rows']} "
        f"bad_time={result['data_quality']['dropped_invalid_time_rows']} "
        f"bad_value={result['data_quality']['dropped_invalid_value_rows']} "
        f"resampled={result['data_quality']['rows_after_resample']}"
    )
    print(
        "窗口: "
        f"total={result['window_counts']['windows']} "
        f"train={result['window_counts']['train_windows']} "
        f"val={result['window_counts']['val_windows']} "
        f"test={result['window_counts']['test_windows']}"
    )
    for warning in result["warnings"]:
        print(f"警告: {warning}")
    print(f"diff_hash: {result['diff_hash']}")
    print(result["diff"])


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = onboard_scenario(
            repo_root=args.repo_root,
            candidate_path=args.candidate,
            db_models_path=args.db_models,
            timeseries_fixture=args.timeseries_fixture,
            registry_path=args.registry,
            allow_external_data=args.allow_external_data,
            apply=args.apply,
            confirmed=args.confirmed,
            expected_diff_hash=args.expected_diff_hash,
        )
    except OnboardError as exc:
        error = {"status": "error", "code": exc.code, "message": exc.message, **exc.details}
        if args.format == "json":
            print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        else:
            print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 2
    _print_result(result, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
