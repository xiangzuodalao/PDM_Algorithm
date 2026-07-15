from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from valeo_pdm.paths import configs_dir
import json
import pyodbc  # 项目已在 pyproject.toml 中引入

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


# def get_model_config(equipment_code: str, meas_code: str) -> Optional[Dict[str, Any]]:
#     registry = load_model_registry()
#     models = registry.get("models", {})
#     defaults = registry.get("defaults", {})
#
#     config = models.get(equipment_code, {}).get(meas_code)
#     if config:
#         return {**defaults, **config}
#     return defaults.copy() if defaults else None


# def list_all_models() -> Dict[str, Dict[str, Dict]]:
#     registry = load_model_registry()
#     return registry.get("models", {})


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
    config_path = os.getenv("VALEO_PDM_SQLSERVER_CONFIG") or str((configs_dir() / "sqlserver_config.json").resolve())

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("conn_str")
    except Exception as e:
        print(f"读取 SQL Server 配置文件失败 ({config_path}): {e}")
        return None


def _get_config_from_db(equipment_code: str, meas_code: str) -> dict | None:
    """1. 尝试从数据库读取单个配置"""
    conn_str = _get_sqlserver_conn_str()
    if not conn_str:
        return None

    try:
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
                print(f"从数据库读取配置: {equipment_code} - {meas_code} ")
                print(f"从数据库读取配置: {row}")

                if row:
                    return {
                        "model_type": row[0],
                        "freq": row[1],
                        "days_back": int(row[2]) if row[2] else 365,
                        "source": row[3],
                        "train_params": json.loads(row[4]) if row[4] else {}
                    }
    except Exception as e:
        print(f"从数据库读取配置异常，准备回退到 YAML: {e}")

    return None


def _get_config_from_yaml(equipment_code: str, meas_code: str) -> dict | None:
    """2. 从 YAML 读取单个配置的兜底逻辑"""
    registry_path = configs_dir() / "model_registry.yaml"
    if not registry_path.exists():
        return None

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        models = data.get("models", {})
        if equipment_code in models and meas_code in models[equipment_code]:
            return models[equipment_code][meas_code]
    except Exception as e:
        print(f"警告: 解析 YAML 配置文件失败。({e})")
    return None


def get_model_config(equipment_code: str, meas_code: str) -> dict | None:
    """
    获取单个模型配置的对外统一入口：
    数据库优先，YAML 兜底。
    """
    print(f"[{equipment_code} - {meas_code}] 查询数据库动态配置。")
    db_config = _get_config_from_db(equipment_code, meas_code)
    if db_config:
        print(f"[{equipment_code} - {meas_code}] 已命中数据库动态配置。")
        return db_config

    yaml_config = _get_config_from_yaml(equipment_code, meas_code)
    if yaml_config:
        return yaml_config

    return None


def _list_models_from_db() -> dict:
    """3. 从数据库表中拉取全量配置列表"""
    models = {}
    conn_str = _get_sqlserver_conn_str()
    if not conn_str:
        return models

    try:
        with pyodbc.connect(conn_str, timeout=3) as conn:
            with conn.cursor() as cursor:
                query = """
                    SELECT EquipmentCode as equipment_code, MeasCode as meas_code, ModelType as model_type, ModelFreq as freq, DaysBack as days_back, ModelSource as source, TrainParamsJson as train_params
                    FROM mom_bas_ai_model_config 
                """
                cursor.execute(query)
                rows = cursor.fetchall()

                for row in rows:
                    eq, meas = row[0], row[1]
                    if eq not in models:
                        models[eq] = {}

                    models[eq][meas] = {
                        "model_type": row[2],
                        "freq": row[3],
                        "days_back": int(row[4]) if row[4] else 365,
                        "source": row[5],
                        "train_params": json.loads(row[6]) if row[6] else {}
                    }
    except Exception as e:
        print(f"警告: 无法从数据库读取模型配置列表，将仅使用 YAML 配置。({e})")

    return models


def _list_models_from_yaml() -> dict:
    """4. 从 YAML 文件拉取全量配置列表"""
    registry_path = configs_dir() / "model_registry.yaml"
    if not registry_path.exists():
        return {}

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("models", {})
    except Exception as e:
        print(f"警告: 解析 YAML 配置文件失败。({e})")
        return {}


def list_all_models() -> dict:
    """
    获取全量模型配置的对外统一入口：
    分别拉取 YAML 和数据库配置并进行深度合并（数据库配置覆盖 YAML 同名配置）。
    """
    yaml_models = _list_models_from_yaml()
    db_models = _list_models_from_db()

    merged_models = {}

    # 1. 先把 YAML 中的静态配置装入
    for eq, params in yaml_models.items():
        merged_models[eq] = {}
        for meas, config in params.items():
            merged_models[eq][meas] = config

    # 2. 用数据库中的动态配置进行追加或覆盖
    for eq, params in db_models.items():
        if eq not in merged_models:
            merged_models[eq] = {}
        for meas, config in params.items():
            merged_models[eq][meas] = config

    return merged_models
