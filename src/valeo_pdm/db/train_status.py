from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import os
import requests
from valeo_pdm.paths import configs_dir
import urllib3

# 2. 禁用忽略 SSL 校验时产生的控制台警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import pyodbc  # type: ignore

    _pyodbc_import_error: Exception | None = None
except Exception as e:  # noqa: BLE001 - 捕获驱动缺失的 ImportError
    pyodbc = None
    _pyodbc_import_error = e


@dataclass
class SqlServerStatusConfig:
    conn_str: str
    table: str = "mom_bas_ai_model_train_record"
    id_column: str = "ModelInfoId"
    status_column: str = "TrainStatus"
    desc_column: str = "ResDescription"
    endtime_column: str = "TrainEndTime"
    activate_column: Optional[str] = "IsActivate"
    activate_value: Optional[int] = 1
    attachments_column: Optional[str] = None
    attachments_max_length: int = 2048
    status_running: int = 1
    status_success: int = 3
    status_failed: int = 2
    description_max_length: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SqlServerStatusConfig":
        return cls(
            conn_str=data["conn_str"],
            table=data.get("table", cls.table),
            id_column=data.get("id_column", cls.id_column),
            status_column=data.get("status_column", cls.status_column),
            desc_column=data.get("desc_column", cls.desc_column),
            endtime_column=data.get("endtime_column", cls.endtime_column),
            activate_column=data.get("activate_column", cls.activate_column),
            activate_value=data.get("activate_value", cls.activate_value),
            attachments_column=data.get("attachments_column", cls.attachments_column),
            attachments_max_length=int(
                data.get("attachments_max_length", cls.attachments_max_length)
            ),
            status_running=int(data.get("status_running", cls.status_running)),
            status_success=int(data.get("status_success", cls.status_success)),
            status_failed=int(data.get("status_failed", cls.status_failed)),
            description_max_length=int(
                data.get("description_max_length", cls.description_max_length)
            ),
        )


class SqlServerTrainStatusUpdater:
    """更新 SQL Server 训练状态（TrainStatus/TrainEndTime/ResDescription）。"""

    def __init__(self, cfg: SqlServerStatusConfig):
        if pyodbc is None:
            detail = (
                "缺少依赖 pyodbc 或 ODBC 驱动 (libodbc.so.2)。"
                " 请安装系统 unixODBC + Microsoft ODBC Driver 18，并执行: pip install pyodbc"
            )
            if _pyodbc_import_error:
                detail += f"；原始错误：{_pyodbc_import_error}"
            raise ModuleNotFoundError(detail)
        self.cfg = cfg

    @classmethod
    def from_json(cls, path: str) -> "SqlServerTrainStatusUpdater":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(SqlServerStatusConfig.from_dict(data))

    def _connect(self):
        return pyodbc.connect(self.cfg.conn_str, autocommit=True)

    def _update(
        self,
        model_info_id: str,
        status: int,
        description: str,
        end_time: datetime | None,
        attachments: str | None = None,
    ) -> None:
        desc = (description or "").strip()
        if desc and len(desc) > self.cfg.description_max_length:
            desc = desc[: self.cfg.description_max_length]

        set_clauses = [
            f"{self.cfg.status_column} = ?",
            f"{self.cfg.desc_column} = ?",
            f"{self.cfg.endtime_column} = ?",
        ]
        params: list[Any] = [status, desc, end_time]

        if attachments and self.cfg.attachments_column:
            att = attachments.strip()
            if (
                att
                and self.cfg.attachments_max_length
                and len(att) > self.cfg.attachments_max_length
            ):
                att = att[: self.cfg.attachments_max_length]
            set_clauses.append(f"{self.cfg.attachments_column} = ?")
            params.append(att)

        query = (
            f"UPDATE {self.cfg.table} WITH (ROWLOCK) SET "
            + ", ".join(set_clauses)
            + f" WHERE 1=1 and (IsRecent = '1' or IsRecent is null) and {self.cfg.id_column} = ?"
        )
        params.append(model_info_id)

        if self.cfg.activate_column:
            query += f" AND {self.cfg.activate_column} = ?"
            params.append(self.cfg.activate_value)

        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SET LOCK_TIMEOUT 5000;")

                cursor.execute(query, params)
                if cursor.rowcount == 0:
                    print(f"警告: 状态更新未影响任何行，可能未找到 ModelInfoId: {model_info_id}")

            conn.commit()
        except Exception as e:
            print(f"数据库状态更新异常(可能触发了锁超时): {e}")
        finally:
            conn.close()

    def mark_running(self, model_info_id: str, description: str = "训练中") -> None:
        self._update(model_info_id, self.cfg.status_running, description, None)

    def mark_success(
        self, model_info_id: str, description: str = "训练成功", attachments: str | None = None
    ) -> None:
        self._update(
            model_info_id, self.cfg.status_success, description, datetime.now(), attachments
        )
        self._update_model_info_status(model_info_id, "训练完成待发布")

    def mark_failed(self, model_info_id: str, description: str = "训练失败") -> None:
        self._update(model_info_id, self.cfg.status_failed, description, datetime.now())
        self._update_model_info_status(model_info_id, "训练失败")

    def _update_model_info_status(self, model_info_id: str, status_text: str) -> None:
        """
        更新 mom_bas_ai_model_info 表的模型主状态。
        """
        query = "UPDATE mom_bas_ai_model_info SET ModelInfoStatus = ? WHERE f_id = ?"

        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, (status_text, model_info_id))

                if cursor.rowcount == 0:
                    print(
                        f"警告: 未能在 mom_bas_ai_model_info 中找到对应记录，f_id: {model_info_id}"
                    )

            conn.commit()
        except Exception as e:
            print(f"!!! 更新 mom_bas_ai_model_info 状态失败: {e}")
            # 这里可以选择是否抛出异常，通常作为附属更新，打印错误即可，不阻断主流程
        finally:
            conn.close()


def _get_upload_url() -> str | None:
    """
    获取图片上传接口的 URL。
    优先读取环境变量，其次读取通用配置。
    """
    # 1. 优先尝试从环境变量获取（Docker 等容器化部署优先）
    env_url = os.getenv("VALEO_PDM_UPLOAD_URL")
    if env_url:
        return env_url

    # 2. 从全局通用配置文件中读取对应的 key
    app_config = _get_app_config()
    return app_config.get("upload_url")


def upload_forecast_image(file_path: str) -> str | None:
    """
    将训练预测图片上传至服务器，并返回指定格式的 JSON 字符串供写入数据库。
    """
    if not file_path or not os.path.exists(file_path):
        return None

    # 动态获取 URL
    upload_url = _get_upload_url()
    if not upload_url:
        print("未配置图片上传 URL (app_config.json 中缺失 upload_url)，跳过上传。")
        return None

    # 同样可以动态获取超时时间，默认给 10 秒
    app_config = _get_app_config()
    timeout = app_config.get("api_timeout", 10)

    try:
        # 构造 form-data
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/png")}

            # 加入 verify=False，强行跳过 SSL 证书合法性检查
            response = requests.post(upload_url, files=files, timeout=timeout, verify=False)

            response.raise_for_status()

            resp_data = response.json()
            # 校验接口返回状态
            if resp_data.get("code") == 200 and "data" in resp_data:
                data_obj = resp_data["data"]

                # 按要求的格式组装 JSON 数组
                attachment_json = [
                    {
                        "name": data_obj.get("name"),
                        "fileId": data_obj.get("fileId"),
                        "url": data_obj.get("url"),
                        "thumbUrl": data_obj.get("thumbUrl"),
                    }
                ]
                # 将字典转为 JSON 字符串
                return json.dumps(attachment_json, ensure_ascii=False)
            else:
                print(f"图片上传失败，接口返回: {resp_data}")
    except Exception as e:
        print(f"上传预测结果图片时发生错误: {e}")

    return None


def _get_app_config() -> dict:
    """
    统一读取系统的全局通用配置文件 (app_config.json)。
    """
    config_path = configs_dir() / "app_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            print("读取全局上传配置失败，已隐藏本地配置细节。")

    return {}
