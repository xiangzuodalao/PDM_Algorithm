from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def project_config() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def dependency_name(specifier: str) -> str:
    return re.split(r"[\s<>=!~;\[]", specifier, maxsplit=1)[0].lower()


def runtime_dependency_names() -> set[str]:
    dependencies = project_config()["project"]["dependencies"]
    return {dependency_name(specifier) for specifier in dependencies}


def test_runtime_dependencies_are_cpu_only_and_first_party_needs_only() -> None:
    names = runtime_dependency_names()
    forbidden = {
        "debugpy",
        "ipykernel",
        "ipython",
        "jupyter-client",
        "jupyter-core",
        "mlstm-kernels",
        "pytest",
        "ruff",
        "torchvision",
        "transformers",
        "triton",
        "xlstm",
    }
    assert "torch" in names
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
    runtime = runtime_dependency_names()
    dev = {dependency_name(specifier) for specifier in config["dependency-groups"]["dev"]}
    assert {"pytest", "ruff"} <= dev
    assert not {"pytest", "ruff"}.intersection(runtime)
