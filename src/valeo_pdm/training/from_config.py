from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
from typing import Any

import yaml
import math
from valeo_pdm.paths import configs_dir, resolve_repo_path
from valeo_pdm.training.registry import get_trainer
from valeo_pdm.training.plan import TrainingPlan, validate_model_info_id
from valeo_pdm.training.run_control import (
    create_training_run,
    finite_metric,
    training_run_lock,
    update_manifest,
)
from valeo_pdm.transformer.artifacts import training_output_dir
from valeo_pdm.transformer.config import get_model_config, list_all_models
from valeo_pdm.db.train_status import SqlServerTrainStatusUpdater, upload_forecast_image


def _postgres_config_path() -> str:
    return os.getenv("VALEO_PDM_POSTGRES_CONFIG") or str(
        (configs_dir() / "postgres_config.json").resolve()
    )


def _sqlserver_config_path() -> str:
    return os.getenv("VALEO_PDM_SQLSERVER_CONFIG") or str(
        (configs_dir() / "sqlserver_config.json").resolve()
    )


def _resolve_csv_path(path_str: str) -> str:
    return str(resolve_repo_path(path_str))


def save_train_params(
    save_dir: str,
    equipment_code: str,
    meas_code: str,
    model_type: str,
    source: str,
    freq: str,
    days_back: int,
    data_path: str,
    train_params: dict,
):
    params_dir = os.path.join(save_dir, "train_params")
    os.makedirs(params_dir, exist_ok=True)

    config_data = {
        "model_type": model_type,
        "equipment_code": equipment_code,
        "meas_code": meas_code,
        "freq": freq,
        "days_back": days_back,
        "source": source,
    }
    if data_path:
        config_data["data_path"] = data_path
    config_data.update(train_params)

    config_path = os.path.join(params_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"训练参数已保存到: {config_path}")


def _maybe_build_status_updater(sqlserver_config: str | None) -> SqlServerTrainStatusUpdater | None:
    if not sqlserver_config:
        return None
    try:
        return SqlServerTrainStatusUpdater.from_json(sqlserver_config)
    except FileNotFoundError:
        print(f"未找到 SQL Server 配置文件: {sqlserver_config}，跳过训练状态回写。")
    except ModuleNotFoundError as e:
        print(f"SQL Server 训练状态回写缺少依赖: {e}")
    except Exception as e:
        print(f"初始化 SQL Server 训练状态回写失败: {e}")
    return None


def _safe_mark(
    updater: SqlServerTrainStatusUpdater | None,
    action: str,
    model_info_id: str,
    description: str,
    attachments: str | None = None,
) -> None:
    if not updater:
        return
    try:
        if action == "running":
            updater.mark_running(model_info_id, description)
        elif action == "success":
            updater.mark_success(model_info_id, description, attachments=attachments)
        elif action == "failed":
            updater.mark_failed(model_info_id, description)
    except Exception as e:  # 不阻塞训练
        print(f"SQL Server 训练状态回写失败({action}): {e}")


def _format_metrics_desc(best_val: Any, test_loss: Any, std: float = 1.0) -> str:
    try:
        val_mse = float(best_val)
        test_mse = float(test_loss)

        # 1. 计算拟合优度 (准确率)
        acc = max(0, (1 - test_mse) * 100)

        # 2. 计算物理误差范围
        physical_err = math.sqrt(test_mse) * std

        print(f"Training OK val={val_mse} test={test_mse}")
        return f"训练成功 | 趋势准确率: {acc:.1f}% | 平均波动误差: ±{physical_err:.2f}"
    except Exception:
        return f"Training OK val={best_val} test={test_loss}"


def _train_from_config_impl(
    equipment_code: str,
    meas_code: str,
    model_info_id: str | None = None,
    sqlserver_config: str | None = None,
    model_type_override: str | None = None,
    _save_dir_override: str | None = None,
    _config_override: dict[str, Any] | None = None,
):
    config = (
        _config_override
        if _config_override is not None
        else get_model_config(equipment_code, meas_code)
    )
    if not config:
        raise ValueError(f"未找到设备[{equipment_code}]参数[{meas_code}]的配置")

    train_params = config.get("train_params", {})
    model_type = model_type_override or config.get("model_type", "informer")
    source = config.get("source", "db")
    freq = config.get("freq", "15min")
    days_back = int(config.get("days_back", 365))
    data_path = config.get("data_path", "")

    # 根据 source 决定使用哪个数据库的配置文件
    if source == "sqlserver":
        db_config_path = _sqlserver_config_path()

    elif source in ["db", "postgres"]:
        db_config_path = _postgres_config_path()
    else:
        db_config_path = None  # CSV 不需要数据库配置
    print(f"数据库配置: {'已选择' if db_config_path else '不适用'}")
    save_dir = _save_dir_override or str(
        training_output_dir(equipment_code, meas_code, model_type).resolve()
    )
    status_updater = (
        _maybe_build_status_updater(sqlserver_config or _sqlserver_config_path())
        if model_info_id
        else None
    )

    print("=" * 60)
    print(f"开始训练: {equipment_code} - {meas_code}")
    print(f"模型类型: {model_type}")
    print(f"数据源: {source}")
    print(f"保存目录: {save_dir}")
    print("=" * 60)

    trainer = get_trainer(model_type)

    common_kwargs = dict(
        source=source,
        data=_resolve_csv_path(data_path) if source == "csv" else "",
        equipment_code=equipment_code,
        meas_code=meas_code,
        days_back=days_back,
        freq=freq,
        config=db_config_path,
    )

    mt = str(model_type).lower()
    if mt == "informer":
        _safe_mark(status_updater, "running", model_info_id, "Training") if model_info_id else None
        try:
            result = trainer(
                **common_kwargs,
                seq_len=train_params.get("seq_len", 672),
                label_len=train_params.get("label_len", 192),
                pred_len=train_params.get("pred_len", 288),
                batch_size=train_params.get("batch_size", 32),
                epochs=train_params.get("epochs", 100),
                lr=train_params.get("lr", 5e-4),
                save=save_dir,
                d_model=train_params.get("d_model", 256),
                n_heads=train_params.get("n_heads", 8),
                d_ff=train_params.get("d_ff", 512),
                dropout=train_params.get("dropout", 0.1),
                e_layers=train_params.get("e_layers", 2),
                d_layers=train_params.get("d_layers", 2),
                attn_type=train_params.get("attn_type", "prob"),
                distil_flag="true" if train_params.get("distil", True) else "false",
                early_stop=True,
                patience=train_params.get("patience", 10),
                lr_sched=True,
                lr_factor=train_params.get("lr_factor", 0.5),
                lr_patience=train_params.get("lr_patience", 5),
                weight_decay=train_params.get("weight_decay", 0.0),
                grad_clip=train_params.get("grad_clip", 0.5),
                stride=train_params.get("stride", 1),
            )
            save_train_params(
                save_dir,
                equipment_code,
                meas_code,
                model_type,
                source,
                freq,
                days_back,
                data_path,
                train_params,
            )
            print(f"训练完成! 最佳模型保存在: {result['best_path']}")
            # if model_info_id:
            #     desc = _format_metrics_desc(result.get("best_val", "N/A"), result.get("test_loss", "N/A"))
            #     attachments = (result.get("plots") or {}).get("forecast")
            #     _safe_mark(status_updater, "success", model_info_id, desc, attachments=attachments)
            if model_info_id:
                desc = _format_metrics_desc(
                    result.get("best_val", "N/A"), result.get("test_loss", "N/A")
                )
                local_attachment_path = (result.get("plots") or {}).get("forecast")

                db_attachments = None
                if local_attachment_path:
                    db_attachments = upload_forecast_image(local_attachment_path)

                _safe_mark(
                    status_updater, "success", model_info_id, desc, attachments=db_attachments
                )
            return result
        except Exception as e:
            if model_info_id:
                _safe_mark(status_updater, "failed", model_info_id, f"Training failed: {e}")
            raise
    if mt == "testmodel":
        _safe_mark(status_updater, "running", model_info_id, "Training") if model_info_id else None
        try:
            result = trainer(
                **common_kwargs,
                seq_len=train_params.get("seq_len", 672),
                label_len=train_params.get("label_len", 192),
                pred_len=train_params.get("pred_len", 288),
                batch_size=train_params.get("batch_size", 32),
                epochs=train_params.get("epochs", 50),
                lr=train_params.get("lr", 5e-4),
                save=save_dir,
                d_model=train_params.get("d_model", 128),
                n_heads=train_params.get("n_heads", 4),
                d_ff=train_params.get("d_ff", 256),
                dropout=train_params.get("dropout", 0.1),
                e_layers=train_params.get("e_layers", 2),
                d_layers=train_params.get("d_layers", 2),
                attn_type=train_params.get("attn_type", "prob"),
                distil_flag="true" if train_params.get("distil", True) else "false",
                early_stop=True,
                patience=train_params.get("patience", 10),
                lr_sched=True,
                lr_factor=train_params.get("lr_factor", 0.5),
                lr_patience=train_params.get("lr_patience", 5),
                weight_decay=train_params.get("weight_decay", 0.0),
                grad_clip=train_params.get("grad_clip", 0.5),
                stride=train_params.get("stride", 1),
            )
            save_train_params(
                save_dir,
                equipment_code,
                meas_code,
                model_type,
                source,
                freq,
                days_back,
                data_path,
                train_params,
            )
            print(f"训练完成! 最佳模型保存在: {result['best_path']}")
            # if model_info_id:
            #     desc = _format_metrics_desc(result.get("best_val", "N/A"), result.get("test_loss", "N/A"))
            #     attachments = (result.get("plots") or {}).get("forecast")
            #     _safe_mark(status_updater, "success", model_info_id, desc, attachments=attachments)
            if model_info_id:
                desc = _format_metrics_desc(
                    result.get("best_val", "N/A"), result.get("test_loss", "N/A")
                )
                local_attachment_path = (result.get("plots") or {}).get("forecast")

                db_attachments = None
                if local_attachment_path:
                    db_attachments = upload_forecast_image(local_attachment_path)

                _safe_mark(
                    status_updater, "success", model_info_id, desc, attachments=db_attachments
                )
            return result
        except Exception as e:
            if model_info_id:
                _safe_mark(status_updater, "failed", model_info_id, f"Training failed: {e}")
            raise

    if mt == "autoformer":
        if source != "csv":
            raise ValueError("Autoformer 目前仅支持 CSV 数据源")
        _safe_mark(status_updater, "running", model_info_id, "Training") if model_info_id else None
        try:
            result = trainer(
                **common_kwargs,
                seq_len=train_params.get("seq_len", 672),
                label_len=train_params.get("label_len", 288),
                pred_len=train_params.get("pred_len", 288),
                batch_size=train_params.get("batch_size", 128),
                epochs=train_params.get("epochs", 120),
                lr=train_params.get("lr", 3e-4),
                save=save_dir,
                moving_avg=train_params.get("moving_avg", 25),
                d_model=train_params.get("d_model", 256),
                n_heads=train_params.get("n_heads", 8),
                d_ff=train_params.get("d_ff", 512),
                dropout=train_params.get("dropout", 0.1),
                e_layers=train_params.get("e_layers", 2),
                d_layers=train_params.get("d_layers", 2),
                early_stop=True,
                patience=train_params.get("patience", 8),
                lr_sched=True,
                lr_factor=train_params.get("lr_factor", 0.5),
                lr_patience=train_params.get("lr_patience", 4),
                weight_decay=train_params.get("weight_decay", 1e-4),
                grad_clip=train_params.get("grad_clip", 0.5),
                stride=train_params.get("stride", 1),
            )
            save_train_params(
                save_dir,
                equipment_code,
                meas_code,
                model_type,
                source,
                freq,
                days_back,
                data_path,
                train_params,
            )
            print(f"训练完成! 最佳模型保存在: {result['best_path']}")
            # if model_info_id:
            #     desc = _format_metrics_desc(result.get("best_val", "N/A"), result.get("test_loss", "N/A"))
            #     attachments = (result.get("plots") or {}).get("forecast")
            #     _safe_mark(status_updater, "success", model_info_id, desc, attachments=attachments)
            if model_info_id:
                desc = _format_metrics_desc(
                    result.get("best_val", "N/A"), result.get("test_loss", "N/A")
                )
                local_attachment_path = (result.get("plots") or {}).get("forecast")

                db_attachments = None
                if local_attachment_path:
                    db_attachments = upload_forecast_image(local_attachment_path)

                _safe_mark(
                    status_updater, "success", model_info_id, desc, attachments=db_attachments
                )
            return result
        except Exception as e:
            if model_info_id:
                _safe_mark(status_updater, "failed", model_info_id, f"Training failed: {e}")
            raise

    if mt == "xlstm":  # 新增模型
        # if source != "csv":
        #     raise ValueError("Autoformer 目前仅支持 CSV 数据源")
        # _safe_mark(status_updater, "running", model_info_id, "Training") if model_info_id else None
        try:
            result = trainer(
                **common_kwargs,
                seq_len=train_params.get("seq_len", 672),
                pred_len=train_params.get("pred_len", 288),
                batch_size=train_params.get("batch_size", 128),
                epochs=train_params.get("epochs", 120),
                lr=train_params.get("lr", 3e-4),
                save=save_dir,
                # moving_avg=train_params.get("moving_avg", 25),
                d_model=train_params.get("d_model", 256),
                n_heads=train_params.get("n_heads", 8),
                d_ff=train_params.get("d_ff", 512),
                dropout=train_params.get("dropout", 0.1),
                e_layers=train_params.get("e_layers", 2),
                d_layers=train_params.get("d_layers", 2),
                early_stop=True,
                patience=train_params.get("patience", 8),
                lr_sched=True,
                lr_factor=train_params.get("lr_factor", 0.5),
                lr_patience=train_params.get("lr_patience", 4),
                weight_decay=train_params.get("weight_decay", 1e-4),
                grad_clip=train_params.get("grad_clip", 0.5),
                label_len=train_params.get("label_len", 192),
                attn_type=train_params.get("attn_type", "prob"),
                distil_flag="true" if train_params.get("distil", True) else "false",
                # 新加参数
                # qk_dim_factor=train_params.get("qk_dim_factor", 0.5),
                # v_dim_factor=train_params.get("v_dim_factor", 1.0),
                # gate_soft_cap=train_params.get("gate_soft_cap", 15.0),
                # ffn_proj_factor=train_params.get("ffn_proj_factor", 2.6667),
                # output_logit_soft_cap=train_params.get("output_logit_soft_cap", 30.0),
                # norm_eps=train_params.get("norm_eps", 1e-6),
                # chunk_size=train_params.get("chunk_size", 64),
                # use_bias=False,
            )
            save_train_params(
                save_dir,
                equipment_code,
                meas_code,
                model_type,
                source,
                freq,
                days_back,
                data_path,
                train_params,
            )
            print(f"训练完成! 最佳模型保存在: {result['best_path']}")
            # if model_info_id:
            #     desc = _format_metrics_desc(result.get("best_val", "N/A"), result.get("test_loss", "N/A"))
            #     attachments = (result.get("plots") or {}).get("forecast")
            #     _safe_mark(status_updater, "success", model_info_id, desc, attachments=attachments)
            if model_info_id:
                desc = _format_metrics_desc(
                    result.get("best_val", "N/A"), result.get("test_loss", "N/A")
                )
                local_attachment_path = (result.get("plots") or {}).get("forecast")

                db_attachments = None
                if local_attachment_path:
                    db_attachments = upload_forecast_image(local_attachment_path)

                _safe_mark(
                    status_updater, "success", model_info_id, desc, attachments=db_attachments
                )
            return result
        except Exception as e:
            if model_info_id:
                _safe_mark(status_updater, "failed", model_info_id, f"Training failed: {e}")
            raise

    raise ValueError(f"不支持的模型类型: {model_type}")


def train_from_config(
    equipment_code: str,
    meas_code: str,
    model_info_id: str | None = None,
    sqlserver_config: str | None = None,
    model_type_override: str | None = None,
):
    """从配置训练；显式 ModelInfoID 使用与 API 相同的独占运行目录和文件锁。"""

    if model_info_id is None:
        return _train_from_config_impl(
            equipment_code,
            meas_code,
            model_info_id,
            sqlserver_config,
            model_type_override,
        )

    validate_model_info_id(model_info_id)
    config = get_model_config(equipment_code, meas_code)
    if not config:
        raise ValueError(f"未找到设备[{equipment_code}]参数[{meas_code}]的配置")
    model_type = str(model_type_override or config.get("model_type", "informer")).lower()
    identity = TrainingPlan(
        equipment_code=equipment_code,
        meas_code=meas_code,
        model_info_id=model_info_id,
        model_type=model_type,
        source=str(config.get("source", "db")),
        freq=str(config.get("freq", "15min")),
        days_back=int(config.get("days_back", 365)),
        data_path=str(config.get("data_path", "")),
        train_params=dict(config.get("train_params", {}) or {}),
        execution_mode="platform",
    )
    with training_run_lock(identity):
        # 锁内重新解析模型类型；CLI 没有 preview hash，仍避免与 API/其他 CLI 并发覆盖。
        current = get_model_config(equipment_code, meas_code)
        if not current:
            raise ValueError(f"未找到设备[{equipment_code}]参数[{meas_code}]的配置")
        current_model_type = str(
            model_type_override or current.get("model_type", "informer")
        ).lower()
        current_identity = TrainingPlan(
            equipment_code=equipment_code,
            meas_code=meas_code,
            model_info_id=model_info_id,
            model_type=current_model_type,
            source=str(current.get("source", "db")),
            freq=str(current.get("freq", "15min")),
            days_back=int(current.get("days_back", 365)),
            data_path=str(current.get("data_path", "")),
            train_params=dict(current.get("train_params", {}) or {}),
            execution_mode="platform",
        )
        run_dir = create_training_run(current_identity)
        update_manifest(
            run_dir,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
            pid=os.getpid(),
        )
        try:
            result = _train_from_config_impl(
                equipment_code,
                meas_code,
                model_info_id,
                sqlserver_config,
                current_model_type,
                str(run_dir),
                _config_override=current,
            )
        except Exception:
            update_manifest(
                run_dir,
                status="failed",
                finished_at=datetime.now(UTC).isoformat(),
                error_code="TRAINING_FAILED",
            )
            raise
        update_manifest(
            run_dir,
            status="succeeded",
            finished_at=datetime.now(UTC).isoformat(),
            best_val=finite_metric(result.get("best_val")),
            test_loss=finite_metric(result.get("test_loss")),
            error_code=None,
        )
        return result


def list_available_configs():
    models = list_all_models()
    print("\n可用的训练配置:")
    print("-" * 60)
    for equipment_code, params in models.items():
        for meas_code, config in params.items():
            model_type = config.get("model_type", "informer")
            source = config.get("source", "db")
            print(f"  {equipment_code} / {meas_code}")
            print(f"    模型: {model_type}, 数据源: {source}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="从配置文件训练模型")
    parser.add_argument("--equipment", "-e", type=str, help="设备编码")
    parser.add_argument("--meas", "-m", type=str, help="参数编码")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有可用配置")
    parser.add_argument("--all", "-a", action="store_true", help="训练所有已配置的模型")

    args = parser.parse_args()

    if args.list:
        list_available_configs()
        return

    if args.all:
        models = list_all_models()
        for equipment_code, params in models.items():
            for meas_code in params.keys():
                train_from_config(equipment_code, meas_code)
        return

    if not args.equipment or not args.meas:
        parser.print_help()
        return

    train_from_config(args.equipment, args.meas)


if __name__ == "__main__":
    main()
