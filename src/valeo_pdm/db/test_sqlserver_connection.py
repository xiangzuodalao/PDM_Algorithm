#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from valeo_pdm.db.train_status import SqlServerStatusConfig


def load_config(path: str) -> SqlServerStatusConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SqlServerStatusConfig.from_dict(data)


def run_probe(cfg: SqlServerStatusConfig, model_info_id: str | None) -> int:
    try:
        import pyodbc  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 无法导入 pyodbc：{e}")
        return 1

    print("[INFO] 正在连接 SQL Server（连接信息已隐藏）")
    with pyodbc.connect(cfg.conn_str, autocommit=True) as conn:
        print(f"[INFO] 连接成功，驱动: {conn.getinfo(pyodbc.SQL_DRIVER_NAME)}")
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            one = cursor.fetchone()
            print(f"[INFO] SELECT 1 -> {one[0] if one else '无返回'}")

            if model_info_id:
                query = (
                    f"SELECT {cfg.status_column}, {cfg.desc_column}, {cfg.endtime_column} "
                    f"FROM {cfg.table} WHERE {cfg.id_column} = ?"
                )
                params: list[Any] = [model_info_id]
                if cfg.activate_column:
                    query += f" AND {cfg.activate_column} = ?"
                    params.append(cfg.activate_value)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                if not rows:
                    print(f"[WARN] 未找到 ModelInfoId={model_info_id} 的记录")
                else:
                    for row in rows:
                        print(f"[INFO] 记录: status={row[0]}, desc={row[1]}, end_time={row[2]}")
    print("[INFO] 测试完成")
    return 0


def main():
    parser = argparse.ArgumentParser(description="测试 SQL Server 连接以及训练状态表可访问性。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sqlserver_config.json",
        help="SQL Server 配置 JSON 路径（默认 configs/sqlserver_config.json）",
    )
    parser.add_argument("--model-info-id", type=str, help="可选，查询指定 ModelInfoId 的记录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sys.exit(run_probe(cfg, args.model_info_id))


if __name__ == "__main__":
    main()
