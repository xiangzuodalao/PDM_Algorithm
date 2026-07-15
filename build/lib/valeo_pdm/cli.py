from __future__ import annotations

import argparse
import os


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="valeo-pdm",
        description="VALEO predictive maintenance utilities",
        epilog="快捷用法示例：\n"
        "  valeo-pdm -l                 # 列出可训练/推理的配置\n"
        "  valeo-pdm serve              # 启动 API 服务\n"
        "  valeo-pdm serve --host 0.0.0.0 --port 8000 --reload\n"
        "  valeo-pdm train -e V-SZ-M-085 -m WC98_WS010",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # 顶层快捷开关：无需记忆子命令
    parser.add_argument("--list-models", "--list", "-l", action="store_true", dest="list_models", help="列出已注册的训练/推理配置")
    parser.add_argument("--serve", "-s", action="store_true", dest="serve_flag", help="启动 FastAPI 服务（等价于子命令 serve）")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="serve 时绑定的 host")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="serve 时绑定的端口")
    parser.add_argument("--reload", action="store_true", help="serve 时启用热重载")

    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("list-models", aliases=["ls", "list"], help="列出已注册的训练/推理配置").set_defaults(cmd="list-models")

    p_train = sub.add_parser("train", help="从配置训练模型")
    p_train.add_argument("--equipment", "-e", type=str, help="设备编码")
    p_train.add_argument("--meas", "-m", type=str, help="参数编码")
    p_train.add_argument("--all", "-a", action="store_true", help="训练所有已配置模型")
    p_train.set_defaults(cmd="train")

    p_serve = sub.add_parser("serve", aliases=["s"], help="启动 FastAPI 服务")
    p_serve.set_defaults(cmd="serve")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "list_models", False):
        from valeo_pdm.training.from_config import list_available_configs

        list_available_configs()
        return 0
    if getattr(args, "serve_flag", False):
        import uvicorn

        uvicorn.run(
            "valeo_pdm.api.app:app",
            host=args.host,
            port=args.port,
            reload=bool(args.reload),
        )
        return 0

    if args.cmd == "list-models":
        from valeo_pdm.training.from_config import list_available_configs

        list_available_configs()
        return 0

    if args.cmd == "train":
        from valeo_pdm.training.from_config import list_available_configs, train_from_config
        from valeo_pdm.transformer.config import list_all_models

        if args.all:
            models = list_all_models()
            for equipment_code, params in models.items():
                for meas_code in params.keys():
                    train_from_config(equipment_code, meas_code)
            return 0

        if not args.equipment or not args.meas:
            list_available_configs()
            parser.error("train 需要同时指定 --equipment 与 --meas，或使用 --all")
        train_from_config(args.equipment, args.meas)
        return 0

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run(
            "valeo_pdm.api.app:app",
            host=args.host,
            port=args.port,
            reload=bool(args.reload),
        )
        return 0

    # 未提供子命令时，打印帮助
    if not args.cmd:
        parser.print_help()
        return 0

    parser.error(f"未知命令: {args.cmd}")
    return 2
