from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration_db


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"缺少显式集成测试参数: {name}")
    return value


def test_new_scenario_is_absent_from_real_sqlserver_readonly() -> None:
    """显式授权后只执行 SELECT，确认真实库中不存在待新增设备+测点。"""

    pyodbc = pytest.importorskip("pyodbc")
    config_path = Path(_required_env("VALEO_PDM_READONLY_SQLSERVER_CONFIG")).resolve()
    equipment_code = _required_env("VALEO_PDM_READONLY_EQUIPMENT_CODE")
    meas_code = _required_env("VALEO_PDM_READONLY_MEAS_CODE")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    conn_str = payload.get("conn_str")
    if not isinstance(conn_str, str) or not conn_str:
        raise AssertionError("只读集成测试配置缺少 conn_str")

    query = """
        SELECT COUNT(1)
        FROM mom_bas_ai_model_config
        WHERE EquipmentCode = ? AND MeasCode = ? AND SeverType = 'transformer'
    """
    with pyodbc.connect(conn_str, timeout=3) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (equipment_code, meas_code))
            row = cursor.fetchone()

    assert row is not None and int(row[0]) == 0
