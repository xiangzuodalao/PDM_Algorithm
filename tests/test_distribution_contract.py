from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def project_config() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def dependency_name(specifier: str) -> str:
    raw_name = re.split(r"[\s<>=!~;\[]", specifier, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", raw_name).lower()


def runtime_dependency_specifiers() -> dict[str, str]:
    dependencies = project_config()["project"]["dependencies"]
    return {dependency_name(specifier): specifier for specifier in dependencies}


def locked_package_names() -> set[str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {dependency_name(package["name"]) for package in lock["package"]}


def test_runtime_dependencies_match_the_approved_cpu_contract() -> None:
    assert runtime_dependency_specifiers() == {
        "fastapi": "fastapi==0.125.0",
        "filelock": "filelock==3.25.2",
        "httpx": "httpx==0.28.1",
        "matplotlib": "matplotlib==3.10.8",
        "numpy": "numpy==2.3.5",
        "pandas": "pandas==2.3.3",
        "psycopg2-binary": "psycopg2-binary==2.9.11",
        "pydantic": "pydantic==2.12.5",
        "pyodbc": "pyodbc==5.3.0",
        "pyyaml": "pyyaml==6.0.3",
        "requests": "requests==2.32.5",
        "starlette": "starlette==0.50.0",
        "torch": "torch==2.10.0",
        "tqdm": "tqdm==4.67.1",
        "urllib3": "urllib3==2.6.3",
        "uvicorn": "uvicorn==0.38.0",
    }


def test_locked_packages_exclude_gpu_and_optional_runtime_stacks() -> None:
    names = locked_package_names()
    forbidden = {
        "debugpy",
        "ipykernel",
        "ipython",
        "jupyter-client",
        "jupyter-core",
        "mlstm-kernels",
        "torchvision",
        "transformers",
        "triton",
        "xlstm",
    }
    assert not names.intersection(forbidden)
    assert not any(name.startswith("nvidia-") for name in names)


def test_torch_uses_the_explicit_cpu_index() -> None:
    config = project_config()
    assert config["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    indices = {index["name"]: index for index in config["tool"]["uv"]["index"]}
    assert indices["pytorch-cpu"] == {
        "name": "pytorch-cpu",
        "url": "https://download.pytorch.org/whl/cpu",
        "explicit": True,
    }


def test_test_and_lint_tools_are_dev_dependencies() -> None:
    config = project_config()
    runtime = set(runtime_dependency_specifiers())
    dev = {dependency_name(specifier) for specifier in config["dependency-groups"]["dev"]}
    assert {"pytest", "ruff"} <= dev
    assert not {"pytest", "ruff"}.intersection(runtime)
