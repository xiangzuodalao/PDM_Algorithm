from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from valeo_pdm.paths import configs_dir
from valeo_pdm.transformer.config_resolution import (
    get_model_block,
    merge_effective_models,
    resolve_model_config,
)

_CONFIG_CACHE: Optional[Dict] = None


class DbModelConfigError(ValueError):
    """数据库存在同名模型行，但该行无法形成有效的整块配置。"""


def _build_db_model_config(
    model_type: object,
    freq: object,
    days_back: object,
    source: object,
    train_params_json: object,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for field, value in (
        ("model_type", model_type),
        ("freq", freq),
        ("source", source),
    ):
        if value is not None:
            config[field] = value

    if days_back is not None:
        try:
            config["days_back"] = int(days_back)
        except (TypeError, ValueError) as exc:
            raise DbModelConfigError("DB 模型配置的 days_back 无效") from exc

    if train_params_json not in (None, ""):
        try:
            train_params = (
                train_params_json
                if isinstance(train_params_json, dict)
                else json.loads(str(train_params_json))
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DbModelConfigError("DB 模型配置的 train_params 无效") from exc
        if not isinstance(train_params, dict):
            raise DbModelConfigError("DB 模型配置的 train_params 必须是映射")
        config["train_params"] = train_params

    return config


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


def get_train_params(equipment_code: str, meas_code: str) -> Optional[Dict[str, Any]]:
    config = get_model_config(equipment_code, meas_code)
    if config:
        return config.get("train_params")
    return None


def _get_sqlserver_conn_str() -> str | None:
    """
    解析 sqlserver_config.json，提取数据库连接字符串 (conn_str)。
    支持通过环境变量 VALEO_PDM_SQLSERVER_CONFIG 覆盖路径。
    """
    config_path = os.getenv("VALEO_PDM_SQLSERVER_CONFIG") or str(
        (configs_dir() / "sqlserver_config.json").resolve()
    )

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("conn_str")
    except Exception:
        print("读取 SQL Server 配置文件失败，已忽略本地连接配置。")
        return None


def _get_config_from_db(equipment_code: str, meas_code: str) -> dict | None:
    """1. 尝试从数据库读取单个配置"""
    conn_str = _get_sqlserver_conn_str()
    if not conn_str:
        return None

    try:
        import pyodbc

        with pyodbc.connect(conn_str, timeout=3) as conn:
            with conn.cursor() as cursor:
                query = """
                    SELECT ModelType as model_type, ModelFreq as freq, DaysBack as days_back, ModelSource as source, TrainParamsJson as train_params
                    FROM mom_bas_ai_model_config 
                    WHERE EquipmentCode = ? AND MeasCode = ?
                    and SeverType = 'transformer'
                """
                cursor.execute(query, (equipment_code, meas_code))
                row = cursor.fetchone()

                if row:
                    return _build_db_model_config(row[0], row[1], row[2], row[3], row[4])
    except DbModelConfigError:
        raise
    except Exception:
        print("数据库模型配置当前不可用，准备回退到 YAML。")

    return None


def _get_config_from_yaml(equipment_code: str, meas_code: str) -> dict | None:
    """2. 从 YAML 读取单个配置的兜底逻辑"""
    try:
        registry = load_model_registry(force_reload=True)
        return get_model_block(registry.get("models", {}), equipment_code, meas_code)
    except Exception:
        print("警告: 解析 YAML 配置文件失败。")
    return None


def get_model_config(equipment_code: str, meas_code: str) -> dict | None:
    """
    获取单个模型配置的对外统一入口：
    数据库整块优先、YAML 整块兜底，再由顶层 defaults 补缺。
    """
    registry = load_model_registry(force_reload=True)
    yaml_models = registry.get("models", {})
    defaults = registry.get("defaults", {})
    db_config = _get_config_from_db(equipment_code, meas_code)
    db_models = {equipment_code: {meas_code: db_config}} if db_config is not None else {}
    config, _source = resolve_model_config(
        defaults=defaults,
        yaml_models=yaml_models,
        db_models=db_models,
        equipment_code=equipment_code,
        meas_code=meas_code,
    )
    return config


def _list_models_from_db() -> dict:
    """3. 从数据库表中拉取全量配置列表"""
    models = {}
    conn_str = _get_sqlserver_conn_str()
    if not conn_str:
        return models

    try:
        import pyodbc

        with pyodbc.connect(conn_str, timeout=3) as conn:
            with conn.cursor() as cursor:
                query = """
                    SELECT EquipmentCode as equipment_code, MeasCode as meas_code, ModelType as model_type, ModelFreq as freq, DaysBack as days_back, ModelSource as source, TrainParamsJson as train_params
                    FROM mom_bas_ai_model_config
                    WHERE SeverType = 'transformer'
                """
                cursor.execute(query)
                rows = cursor.fetchall()

                for row in rows:
                    eq, meas = row[0], row[1]
                    if eq not in models:
                        models[eq] = {}

                    models[eq][meas] = _build_db_model_config(
                        row[2], row[3], row[4], row[5], row[6]
                    )
    except DbModelConfigError:
        raise
    except Exception:
        print("警告: 数据库模型配置列表当前不可用，将仅使用 YAML 配置。")

    return models


def _list_models_from_yaml() -> dict:
    """4. 从 YAML 文件拉取全量配置列表"""
    try:
        return load_model_registry(force_reload=True).get("models", {})
    except Exception:
        print("警告: 解析 YAML 配置文件失败。")
        return {}


def list_all_models() -> dict:
    """
    获取全量模型配置的对外统一入口：
    DB 同名配置整块覆盖 YAML，并为每个最终场景补齐顶层 defaults。
    """
    registry = load_model_registry(force_reload=True)
    yaml_models = registry.get("models", {})
    db_models = _list_models_from_db()
    return merge_effective_models(
        defaults=registry.get("defaults", {}),
        yaml_models=yaml_models,
        db_models=db_models,
    )
