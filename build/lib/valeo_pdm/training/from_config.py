from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from valeo_pdm.paths import configs_dir
from valeo_pdm.training.registry import get_trainer
from valeo_pdm.transformer.artifacts import training_output_dir
from valeo_pdm.transformer.config import get_model_config, list_all_models


def _postgres_config_path() -> str:
    return os.getenv("VALEO_PDM_POSTGRES_CONFIG") or str((configs_dir() / "postgres_config.json").resolve())


def _resolve_csv_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())
    return str(p)


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


def train_from_config(equipment_code: str, meas_code: str):
    config = get_model_config(equipment_code, meas_code)
    if not config:
        raise ValueError(f"未找到设备[{equipment_code}]参数[{meas_code}]的配置")

    train_params = config.get("train_params", {})
    model_type = config.get("model_type", "informer")
    source = config.get("source", "db")
    freq = config.get("freq", "15min")
    days_back = int(config.get("days_back", 365))
    data_path = config.get("data_path", "")

    save_dir = str(training_output_dir(equipment_code, meas_code, model_type).resolve())
    db_config = _postgres_config_path()

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
        config=db_config,
    )

    if str(model_type).lower() == "informer":
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
            stride=train_params.get("stride", 12),
        )
        save_train_params(
            save_dir, equipment_code, meas_code, model_type, source, freq, days_back, data_path, train_params
        )
        print(f"训练完成! 最佳模型保存在: {result['best_path']}")
        return result

    if str(model_type).lower() == "autoformer":
        if source != "csv":
            raise ValueError("Autoformer 目前仅支持 CSV 数据源")
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
        )
        save_train_params(
            save_dir, equipment_code, meas_code, model_type, source, freq, days_back, data_path, train_params
        )
        print(f"训练完成! 最佳模型保存在: {result['best_path']}")
        return result

    raise ValueError(f"不支持的模型类型: {model_type}")


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
