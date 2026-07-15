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


def checkpoints_dir() -> Path:
    return artifacts_dir() / "checkpoints"
