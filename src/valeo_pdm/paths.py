from __future__ import annotations

import os
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    env = os.getenv("VALEO_PDM_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    probe = (start or Path.cwd()).resolve()
    for parent in (probe, *probe.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return probe


def configs_dir() -> Path:
    return repo_root() / "configs"


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def data_dir() -> Path:
    return repo_root() / "data"


def resolve_repo_path(path: str | Path) -> Path:
    """将相对路径稳定地按项目根目录解析，而不是依赖进程工作目录。"""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root(Path(__file__).resolve().parent) / candidate).resolve()


def resolve_data_path(path: str | Path) -> Path:
    """只解析 `data/` 下的相对路径，并阻止 `..` 与符号链接越界。"""

    raw = Path(path)
    if raw.is_absolute() or ".." in raw.parts or len(raw.parts) < 2 or raw.parts[0] != "data":
        raise ValueError("CSV 路径必须是 data/ 下的相对路径")
    root = data_dir().resolve()
    candidate = resolve_repo_path(raw)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("CSV 路径解析后超出 data 目录") from exc
    if not relative.parts:
        raise ValueError("CSV 路径必须指向 data 目录内的文件")
    return candidate


def checkpoints_dir() -> Path:
    return artifacts_dir() / "checkpoints"
