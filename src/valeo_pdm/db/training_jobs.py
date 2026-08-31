from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from valeo_pdm.paths import configs_dir
from valeo_pdm.training.plan import TrainingPlan


TrainingJobStatus = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]
TRAINING_JOB_TABLE = "dbo.training_jobs"


class TrainingJobStoreError(RuntimeError):
    pass


class TrainingJobConflict(TrainingJobStoreError):
    pass


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    equipment_code: str
    meas_code: str
    model_info_id: str
    model_type: str
    plan_json: str
    plan_hash: str
    status: TrainingJobStatus
    error_code: str | None
    best_val: float | None
    test_loss: float | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


def sqlserver_config_path() -> Path:
    configured = os.getenv("VALEO_PDM_SQLSERVER_CONFIG")
    return Path(configured) if configured else configs_dir() / "sqlserver_config.json"


def _load_pyodbc():
    try:
        import pyodbc  # type: ignore
    except Exception as exc:
        raise TrainingJobStoreError("SQL Server ODBC 驱动不可用") from exc
    return pyodbc


class SqlServerTrainingJobRepository:
    """仅管理异步训练任务状态，不承载平台业务状态。"""

    _COLUMNS = (
        "job_id",
        "equipment_code",
        "meas_code",
        "model_info_id",
        "model_type",
        "plan_json",
        "plan_hash",
        "status",
        "error_code",
        "best_val",
        "test_loss",
        "created_at",
        "started_at",
        "finished_at",
    )
    _SELECT_COLUMNS = ", ".join(_COLUMNS)
    _OUTPUT_COLUMNS = ", ".join(f"INSERTED.{name}" for name in _COLUMNS)

    def __init__(self, conn_str: str):
        self.conn_str = conn_str

    @classmethod
    def from_json(cls, path: str | os.PathLike[str] | None = None):
        config_path = Path(path) if path is not None else sqlserver_config_path()
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                config = json.load(stream)
            conn_str = str(config["conn_str"]).strip()
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise TrainingJobStoreError("训练任务数据库配置不可用") from exc
        if not conn_str:
            raise TrainingJobStoreError("训练任务数据库连接串为空")
        return cls(conn_str)

    def _connect(self):
        pyodbc = _load_pyodbc()
        try:
            return pyodbc.connect(self.conn_str, autocommit=True)
        except Exception as exc:
            raise TrainingJobStoreError("训练任务数据库连接失败") from exc

    @staticmethod
    def _to_job(row: Any) -> TrainingJob:
        return TrainingJob(
            job_id=str(row[0]),
            equipment_code=str(row[1]),
            meas_code=str(row[2]),
            model_info_id=str(row[3]),
            model_type=str(row[4]),
            plan_json=str(row[5]),
            plan_hash=str(row[6]),
            status=str(row[7]),
            error_code=str(row[8]) if row[8] is not None else None,
            best_val=float(row[9]) if row[9] is not None else None,
            test_loss=float(row[10]) if row[10] is not None else None,
            created_at=row[11],
            started_at=row[12],
            finished_at=row[13],
        )

    def create_queued(self, job_id: str, plan: TrainingPlan) -> TrainingJob:
        query = f"""
            INSERT INTO {TRAINING_JOB_TABLE} (
                job_id, equipment_code, meas_code, model_info_id, model_type,
                plan_json, plan_hash, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED')
        """
        conn = self._connect()
        try:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        query,
                        job_id,
                        plan.equipment_code,
                        plan.meas_code,
                        plan.model_info_id,
                        plan.model_type,
                        plan.to_json(),
                        plan.plan_hash,
                    )
            except Exception as exc:
                pyodbc = _load_pyodbc()
                if isinstance(exc, pyodbc.IntegrityError):
                    raise TrainingJobConflict("ModelInfoID 对应的训练任务已存在") from exc
                raise TrainingJobStoreError("创建训练任务失败") from exc
        finally:
            conn.close()
        job = self.get(job_id)
        if job is None:
            raise TrainingJobStoreError("创建训练任务后无法读取任务")
        return job

    def get(self, job_id: str) -> TrainingJob | None:
        query = f"SELECT {self._SELECT_COLUMNS} FROM {TRAINING_JOB_TABLE} WHERE job_id = ?"
        return self._fetch_one(query, job_id)

    def get_by_identity(
        self,
        equipment_code: str,
        meas_code: str,
        model_info_id: str,
    ) -> TrainingJob | None:
        query = f"""
            SELECT {self._SELECT_COLUMNS}
            FROM {TRAINING_JOB_TABLE}
            WHERE equipment_code = ? AND meas_code = ? AND model_info_id = ?
        """
        return self._fetch_one(query, equipment_code, meas_code, model_info_id)

    def _fetch_one(self, query: str, *params: Any) -> TrainingJob | None:
        conn = self._connect()
        try:
            try:
                with conn.cursor() as cursor:
                    row = cursor.execute(query, *params).fetchone()
            except Exception as exc:
                raise TrainingJobStoreError("读取训练任务失败") from exc
        finally:
            conn.close()
        return self._to_job(row) if row is not None else None

    def claim(self, job_id: str) -> TrainingJob | None:
        query = f"""
            UPDATE {TRAINING_JOB_TABLE} WITH (ROWLOCK)
            SET status = 'RUNNING', started_at = SYSUTCDATETIME(), error_code = NULL
            OUTPUT {self._OUTPUT_COLUMNS}
            WHERE job_id = ? AND status = 'QUEUED'
        """
        conn = self._connect()
        try:
            try:
                with conn.cursor() as cursor:
                    row = cursor.execute(query, job_id).fetchone()
            except Exception as exc:
                raise TrainingJobStoreError("领取训练任务失败") from exc
        finally:
            conn.close()
        return self._to_job(row) if row is not None else None

    def mark_succeeded(
        self,
        job_id: str,
        *,
        best_val: float | None,
        test_loss: float | None,
    ) -> None:
        self._transition_from_running(
            job_id,
            status="SUCCEEDED",
            error_code=None,
            best_val=best_val,
            test_loss=test_loss,
        )

    def mark_failed(self, job_id: str, *, error_code: str) -> None:
        self._transition_from_running(
            job_id,
            status="FAILED",
            error_code=error_code,
            best_val=None,
            test_loss=None,
        )

    def _transition_from_running(
        self,
        job_id: str,
        *,
        status: Literal["SUCCEEDED", "FAILED"],
        error_code: str | None,
        best_val: float | None,
        test_loss: float | None,
    ) -> None:
        query = f"""
            UPDATE {TRAINING_JOB_TABLE} WITH (ROWLOCK)
            SET status = ?, error_code = ?, best_val = ?, test_loss = ?,
                finished_at = SYSUTCDATETIME()
            WHERE job_id = ? AND status = 'RUNNING'
        """
        conn = self._connect()
        try:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        query,
                        status,
                        error_code,
                        best_val,
                        test_loss,
                        job_id,
                    )
                    if cursor.rowcount != 1:
                        raise TrainingJobStoreError("训练任务状态转换冲突")
            except TrainingJobStoreError:
                raise
            except Exception as exc:
                raise TrainingJobStoreError("更新训练任务状态失败") from exc
        finally:
            conn.close()

    def delete_queued(self, job_id: str) -> bool:
        query = f"DELETE FROM {TRAINING_JOB_TABLE} WHERE job_id = ? AND status = 'QUEUED'"
        conn = self._connect()
        try:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query, job_id)
                    return cursor.rowcount == 1
            except Exception as exc:
                raise TrainingJobStoreError("撤销未发布训练任务失败") from exc
        finally:
            conn.close()
