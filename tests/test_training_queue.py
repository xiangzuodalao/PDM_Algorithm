from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from valeo_pdm.db.training_jobs import SqlServerTrainingJobRepository
from valeo_pdm.training.celery_app import app
from valeo_pdm.training.plan import TrainingPlan, TrainingPlanError, build_training_plan


def _plan() -> TrainingPlan:
    return build_training_plan(
        equipment_code="EQ-1",
        meas_code="MEAS-1",
        model_info_id="queue-plan-1",
        requested_model_type="informer",
        param_items=[("epochs", 1)],
        requested_source="db",
        execution_mode="local_only",
        model_config={
            "model_type": "informer",
            "source": "db",
            "freq": "15min",
            "days_back": 30,
            "train_params": {
                "seq_len": 48,
                "label_len": 24,
                "pred_len": 12,
                "d_model": 64,
                "n_heads": 4,
            },
        },
    )


def test_training_plan_round_trips_without_configuration_reread() -> None:
    original = _plan()

    restored = TrainingPlan.from_json(original.to_json(), expected_hash=original.plan_hash)

    assert restored == original
    with pytest.raises(TrainingPlanError) as caught:
        TrainingPlan.from_json(original.to_json(), expected_hash="0" * 64)
    assert caught.value.code == "INVALID_STORED_PLAN"


def test_celery_is_configured_for_one_long_training_per_worker() -> None:
    assert app.conf.task_default_queue == "training"
    assert app.conf.task_acks_late is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_ignore_result is True
    assert app.conf.task_serializer == "json"
    assert app.conf.accept_content == ["json"]
    assert app.conf.result_backend is None
    assert app.conf.broker_transport_options == {"visibility_timeout": 86400}


def test_sql_migration_has_four_state_constraint() -> None:
    migration = Path("sql/001_create_training_jobs.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE dbo.training_jobs" in migration
    assert "UNIQUE (equipment_code, meas_code, model_info_id)" in migration
    assert "CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED'))" in migration


def test_sql_repository_claims_only_queued_job(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    job_id = str(uuid4())
    row = (
        job_id,
        "EQ-1",
        "MEAS-1",
        "queue-plan-1",
        "informer",
        _plan().to_json(),
        _plan().plan_hash,
        "RUNNING",
        None,
        None,
        None,
        now,
        now,
        None,
    )

    class Cursor:
        query = ""
        params: tuple[object, ...] = ()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query: str, *params: object):
            self.query = query
            self.params = params
            return self

        def fetchone(self):
            return row

    class Connection:
        def __init__(self):
            self.cursor_instance = Cursor()
            self.closed = False

        def cursor(self):
            return self.cursor_instance

        def close(self):
            self.closed = True

    connection = Connection()
    repository = SqlServerTrainingJobRepository("unused")
    monkeypatch.setattr(repository, "_connect", lambda: connection)

    claimed = repository.claim(job_id)

    assert claimed is not None
    assert claimed.status == "RUNNING"
    assert "WHERE job_id = ? AND status = 'QUEUED'" in connection.cursor_instance.query
    assert "OUTPUT INSERTED.job_id" in connection.cursor_instance.query
    assert connection.cursor_instance.params == (job_id,)
    assert connection.closed is True
