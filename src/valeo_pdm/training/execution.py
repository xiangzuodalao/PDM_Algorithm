from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from valeo_pdm.db.train_status import SqlServerTrainStatusUpdater, upload_forecast_image
from valeo_pdm.paths import configs_dir, resolve_data_path
from valeo_pdm.training.from_config import _format_metrics_desc, save_train_params
from valeo_pdm.training.plan import TrainingPlan
from valeo_pdm.training.registry import get_trainer
from valeo_pdm.training.run_control import (
    create_training_run,
    finite_metric,
    read_manifest,
    training_run_lock,
    update_manifest,
)
from valeo_pdm.transformer.artifacts import checkpoint_filename


@dataclass(frozen=True)
class TrainingExecutionResult:
    best_path: str
    run_dir: str
    best_val: float | None
    test_loss: float | None


def _postgres_config_path() -> str:
    return os.getenv("VALEO_PDM_POSTGRES_CONFIG") or str(
        (configs_dir() / "postgres_config.json").resolve()
    )


def _sqlserver_config_path() -> str:
    return os.getenv("VALEO_PDM_SQLSERVER_CONFIG") or str(
        (configs_dir() / "sqlserver_config.json").resolve()
    )


def _database_config_for_plan(plan: TrainingPlan) -> str | None:
    if plan.source == "sqlserver":
        return _sqlserver_config_path()
    if plan.source in {"db", "postgres"}:
        return _postgres_config_path()
    return None


def _invoke_trainer(plan: TrainingPlan, save_dir: str) -> dict[str, Any]:
    params = plan.train_params
    trainer = get_trainer(plan.model_type)
    common_kwargs = dict(
        source=plan.source,
        data=str(resolve_data_path(plan.data_path)) if plan.source == "csv" else "",
        equipment_code=plan.equipment_code,
        meas_code=plan.meas_code,
        days_back=plan.days_back,
        freq=plan.freq,
        config=_database_config_for_plan(plan),
    )
    if plan.model_type == "informer":
        return trainer(
            **common_kwargs,
            seq_len=params["seq_len"],
            label_len=params["label_len"],
            pred_len=params["pred_len"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            lr=params["lr"],
            save=save_dir,
            d_model=params["d_model"],
            n_heads=params["n_heads"],
            d_ff=params["d_ff"],
            dropout=params["dropout"],
            e_layers=params["e_layers"],
            d_layers=params["d_layers"],
            attn_type=params["attn_type"],
            distil_flag="true" if params["distil"] else "false",
            early_stop=True,
            patience=params["patience"],
            lr_sched=True,
            lr_factor=params["lr_factor"],
            lr_patience=params["lr_patience"],
            weight_decay=params["weight_decay"],
            grad_clip=params["grad_clip"],
            stride=params["stride"],
        )
    return trainer(
        **common_kwargs,
        seq_len=params["seq_len"],
        label_len=params["label_len"],
        pred_len=params["pred_len"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        lr=params["lr"],
        save=save_dir,
        moving_avg=params["moving_avg"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        e_layers=params["e_layers"],
        d_layers=params["d_layers"],
        early_stop=True,
        patience=params["patience"],
        lr_sched=True,
        lr_factor=params["lr_factor"],
        lr_patience=params["lr_patience"],
        weight_decay=params["weight_decay"],
        grad_clip=params["grad_clip"],
        stride=params["stride"],
    )


def _validated_training_checkpoint(
    plan: TrainingPlan, run_dir: Path, result: dict[str, Any]
) -> Path:
    raw = result.get("best_path")
    if not isinstance(raw, (str, os.PathLike)):
        raise RuntimeError("trainer did not return best_path")
    declared = Path(raw)
    if declared.is_symlink():
        raise RuntimeError("trainer checkpoint cannot be a symlink")
    try:
        resolved = declared.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("trainer checkpoint does not exist") from exc
    expected = (run_dir / checkpoint_filename(plan.model_type)).resolve(strict=False)
    if resolved != expected or not resolved.is_file():
        raise RuntimeError("trainer checkpoint is outside the reserved run directory")
    return resolved


def _platform_status_updater(plan: TrainingPlan) -> SqlServerTrainStatusUpdater | None:
    if plan.execution_mode != "platform":
        return None
    try:
        return SqlServerTrainStatusUpdater.from_json(_sqlserver_config_path())
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    except Exception:
        print("初始化 SQL Server 训练状态回写失败，已继续本地训练。")
        return None


def _mark_manifest_failed(run_dir: Path | None) -> None:
    if run_dir is None:
        return
    try:
        manifest = read_manifest(run_dir) or {}
        if manifest.get("status") not in {"failed", "succeeded"}:
            update_manifest(
                run_dir,
                status="failed",
                finished_at=datetime.now(UTC).isoformat(),
                error_code="TRAINING_FAILED",
            )
    except Exception:
        pass


def execute_training_plan(plan: TrainingPlan) -> TrainingExecutionResult:
    """执行已冻结的训练计划；调用方负责队列状态流转。"""

    run_dir: Path | None = None
    status_updater: SqlServerTrainStatusUpdater | None = None
    try:
        with training_run_lock(plan):
            run_dir = create_training_run(plan)
            update_manifest(
                run_dir,
                status="running",
                started_at=datetime.now(UTC).isoformat(),
                pid=os.getpid(),
            )
            status_updater = _platform_status_updater(plan)
            if status_updater is not None:
                try:
                    status_updater.mark_running(plan.model_info_id, "Training")
                except Exception:
                    print("SQL Server 状态回写(训练中)失败，已继续本地训练。")

            try:
                result = _invoke_trainer(plan, str(run_dir))
                if not isinstance(result, dict):
                    raise TypeError("trainer result must be a mapping")
                best_path = _validated_training_checkpoint(plan, run_dir, result)
                save_train_params(
                    str(run_dir),
                    plan.equipment_code,
                    plan.meas_code,
                    plan.model_type,
                    plan.source,
                    plan.freq,
                    plan.days_back,
                    plan.data_path,
                    plan.train_params,
                )
            except Exception:
                _mark_manifest_failed(run_dir)
                if status_updater is not None:
                    try:
                        status_updater.mark_failed(plan.model_info_id, "Training failed")
                    except Exception:
                        print("SQL Server 状态回写(失败)失败。")
                raise

            best_val = finite_metric(result.get("best_val"))
            test_loss = finite_metric(result.get("test_loss"))
            if status_updater is not None:
                try:
                    description = _format_metrics_desc(best_val, test_loss)
                    forecast_path = (result.get("plots") or {}).get("forecast")
                    attachments = upload_forecast_image(forecast_path) if forecast_path else None
                    status_updater.mark_success(
                        plan.model_info_id,
                        description,
                        attachments=attachments,
                    )
                except Exception:
                    print("SQL Server 状态回写或预测图上传失败，本地训练结果仍然有效。")

            update_manifest(
                run_dir,
                status="succeeded",
                finished_at=datetime.now(UTC).isoformat(),
                best_val=best_val,
                test_loss=test_loss,
                error_code=None,
            )
            return TrainingExecutionResult(
                best_path=str(best_path),
                run_dir=str(run_dir),
                best_val=best_val,
                test_loss=test_loss,
            )
    except Exception:
        _mark_manifest_failed(run_dir)
        raise
