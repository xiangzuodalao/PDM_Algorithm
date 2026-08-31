from __future__ import annotations

from typing import Any

from valeo_pdm.db.training_jobs import SqlServerTrainingJobRepository
from valeo_pdm.training.celery_app import app
from valeo_pdm.training.execution import execute_training_plan
from valeo_pdm.training.plan import TrainingPlan, TrainingPlanError


def _job_repository() -> SqlServerTrainingJobRepository:
    return SqlServerTrainingJobRepository.from_json()


@app.task(name="valeo_pdm.training.train_model")
def train_model(job_id: str) -> dict[str, Any]:
    """领取并执行一个训练任务；非 QUEUED 消息按幂等无操作处理。"""

    repository = _job_repository()
    job = repository.claim(job_id)
    if job is None:
        return {"job_id": job_id, "executed": False}

    try:
        plan = TrainingPlan.from_json(job.plan_json, expected_hash=job.plan_hash)
        result = execute_training_plan(plan)
        repository.mark_succeeded(
            job_id,
            best_val=result.best_val,
            test_loss=result.test_loss,
        )
    except Exception as exc:
        error_code = exc.code if isinstance(exc, TrainingPlanError) else "TRAINING_FAILED"
        try:
            repository.mark_failed(job_id, error_code=error_code)
        except Exception:
            print(f"训练任务[{job_id}]失败状态写入 SQL Server 失败。")
        raise

    return {"job_id": job_id, "executed": True, "status": "SUCCEEDED"}
