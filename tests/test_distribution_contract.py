from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]

EXPLICIT_FORBIDDEN_LOCK_PACKAGES = frozenset(
    {
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
)


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


def is_forbidden_locked_package(package_name: str) -> bool:
    normalized_name = dependency_name(package_name)
    return normalized_name in EXPLICIT_FORBIDDEN_LOCK_PACKAGES or normalized_name.startswith(
        ("cuda-", "nvidia-")
    )


def compose_service() -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip(
            "Docker CLI is unavailable; semantic Compose validation requires Docker Compose v2"
        )

    version = subprocess.run(
        [docker, "compose", "version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode != 0:
        pytest.skip(
            "Docker Compose v2 is unavailable; semantic Compose validation requires "
            f"`docker compose version` (stderr: {version.stderr.strip()!r})"
        )

    config_help = subprocess.run(
        [docker, "compose", "config", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if config_help.returncode != 0 or "--format" not in config_help.stdout:
        pytest.skip(
            "Docker Compose `config --format json` is unavailable; semantic Compose validation "
            f"requires JSON expansion (stderr: {config_help.stderr.strip()!r})"
        )

    try:
        result = subprocess.run(
            [
                docker,
                "compose",
                "-f",
                str(ROOT / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(
            "docker compose config --format json failed unexpectedly "
            f"(stdout: {error.stdout!r}; stderr: {error.stderr!r})"
        )

    try:
        compose = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        pytest.fail(
            "docker compose config --format json returned invalid JSON "
            f"(stdout: {result.stdout!r}; stderr: {result.stderr!r}; error: {error})"
        )
    assert isinstance(compose, dict)
    services = compose.get("services")
    assert isinstance(services, dict)
    service = services.get("valeo-pdm-api")
    assert isinstance(service, dict)
    return service


def nested_value_contains_gpu(value: object) -> bool:
    if isinstance(value, str):
        return value.casefold() == "gpu"
    if isinstance(value, (list, tuple)):
        return any(nested_value_contains_gpu(item) for item in value)
    return False


def dockerfile_instructions() -> list[tuple[str, str]]:
    instructions: list[tuple[str, str]] = []
    logical_line_parts: list[str] = []
    for raw_line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        continues = line.endswith("\\")
        logical_line_parts.append(line[:-1].rstrip() if continues else line)
        if continues:
            continue
        instruction, separator, value = " ".join(logical_line_parts).partition(" ")
        assert separator
        instructions.append((instruction.upper(), value))
        logical_line_parts = []
    assert not logical_line_parts
    return instructions


def run_instructions() -> list[str]:
    return [value for instruction, value in dockerfile_instructions() if instruction == "RUN"]


def option_has_value(tokens: list[str], option: str, expected: str) -> bool:
    return token_has_value(tokens, option, expected) or f"{option}={expected}" in tokens


def token_has_value(tokens: list[str], option: str, expected: str) -> bool:
    return any(
        token == option and next_token == expected for token, next_token in zip(tokens, tokens[1:])
    )


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
        "rfc8785": "rfc8785==0.1.4",
        "starlette": "starlette==0.50.0",
        "torch": "torch==2.10.0",
        "tqdm": "tqdm==4.67.1",
        "urllib3": "urllib3==2.6.3",
        "uvicorn": "uvicorn==0.38.0",
    }


def test_locked_packages_exclude_gpu_and_optional_runtime_stacks() -> None:
    names = locked_package_names()
    assert not {name for name in names if is_forbidden_locked_package(name)}


@pytest.mark.parametrize(
    ("package_name", "expected"),
    [
        ("cuda-runtime", True),
        ("CUDA_runtime", True),
        ("cuda.toolkit", True),
        ("nvidia-cublas-cu12", True),
        ("NVIDIA_cudnn_cu12", True),
        ("debugpy", True),
        ("ipykernel", True),
        ("ipython", True),
        ("jupyter_client", True),
        ("jupyter.core", True),
        ("mlstm_kernels", True),
        ("torchvision", True),
        ("transformers", True),
        ("triton", True),
        ("xlstm", True),
        ("torch", False),
        ("cudf", False),
    ],
)
def test_forbidden_lock_package_name_normalization(package_name: str, expected: bool) -> None:
    assert is_forbidden_locked_package(package_name) is expected


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


def test_dockerfile_has_one_frozen_sync_and_no_second_torch_install() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count("uv sync") == 1
    assert "uv sync --frozen --no-dev --python 3.12" in dockerfile
    assert "uv pip install torch" not in dockerfile
    assert '"--workers", "1"' in dockerfile


def test_dockerfile_pins_external_images_to_proven_linux_amd64_digests() -> None:
    instructions = dockerfile_instructions()
    assert (
        "FROM",
        "docker.io/library/ubuntu@sha256:"
        "4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90",
    ) in instructions
    assert any(
        instruction == "COPY"
        and value.startswith(
            "--from=ghcr.io/astral-sh/uv@sha256:"
            "df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c "
        )
        for instruction, value in instructions
    )


def test_dockerfile_redirects_stderr_to_dev_null() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "2>dev/null" not in dockerfile


def test_current_docs_use_locked_cpu_install_workflow() -> None:
    current_docs = "\n".join(
        (ROOT / path).read_text(encoding="utf-8") for path in ("README.md", "AGENTS.md")
    )
    forbidden_markers = (
        "/whl/cu",
        "pip install torch",
        "pip install torchvision",
        "pip install mlstm",
        "pip install transformers",
        "torchvision==",
        "python3 -m pip install -e .",
    )
    assert not {marker for marker in forbidden_markers if marker in current_docs}


def test_default_compose_does_not_request_gpu() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "driver: nvidia" not in compose
    assert "capabilities: [ gpu ]" not in compose


def test_default_compose_semantically_has_no_gpu_request() -> None:
    service = compose_service()
    assert "gpus" not in service
    assert str(service.get("runtime", "")).casefold() != "nvidia"

    deploy = service.get("deploy", {})
    assert isinstance(deploy, dict)
    resources = deploy.get("resources", {})
    assert isinstance(resources, dict)
    reservations = resources.get("reservations", {})
    assert isinstance(reservations, dict)
    devices = reservations.get("devices", [])
    assert isinstance(devices, list)
    for device in devices:
        assert isinstance(device, dict)
        assert str(device.get("driver", "")).casefold() != "nvidia"
        assert not nested_value_contains_gpu(device.get("capabilities", []))


def test_dockerfile_semantically_uses_cpu_only_runtime_contract() -> None:
    run_values = run_instructions()
    sync_runs = [
        value
        for value in run_values
        if any(pair == ("uv", "sync") for pair in zip(shlex.split(value), shlex.split(value)[1:]))
    ]
    assert len(sync_runs) == 1
    sync_tokens = shlex.split(sync_runs[0])
    assert "--frozen" in sync_tokens
    assert "--no-dev" in sync_tokens
    assert option_has_value(sync_tokens, "--python", "3.12")

    installer = re.compile(
        r"\b(?:uv\s+pip|uv\s+tool|pip(?:\d+(?:\.\d+)*)?|python(?:\d+(?:\.\d+)*)?\s+-m\s+pip|"
        r"pipx|conda|mamba|poetry|apt(?:-get)?|apk|dnf|yum)\s+(?:install|add)\b",
        re.IGNORECASE,
    )
    forbidden_package = re.compile(
        r"\b(?:torch(?:vision)?|triton|xlstm|mlstm-kernels|nvidia[\w.-]*|cuda[\w.-]*)\b",
        re.IGNORECASE,
    )
    cuda_index = re.compile(r"/whl/cu[\w.-]*", re.IGNORECASE)
    for value in run_values:
        assert not cuda_index.search(value)
        if installer.search(value):
            assert not forbidden_package.search(value)

    cmd_values = [value for instruction, value in dockerfile_instructions() if instruction == "CMD"]
    assert len(cmd_values) == 1
    command = json.loads(cmd_values[0])
    assert isinstance(command, list)
    worker_positions = [index for index, argument in enumerate(command) if argument == "--workers"]
    assert len(worker_positions) == 1
    assert worker_positions[0] < len(command) - 1
    assert command[worker_positions[0] + 1] == "1"
