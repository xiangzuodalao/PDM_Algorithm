"""Startup guard for the isolated installed-console CPU smoke child."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import NoReturn


def _required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"{variable} is required by the child-process guard")
    return Path(value)


NETWORK_ATTEMPT_PATH = _required_path("PDM_CHILD_NETWORK_SENTINEL")
GUARD_REPORT_PATH = _required_path("PDM_CHILD_GUARD_REPORT")


def _record_and_block(connector: str, *_args: object, **_kwargs: object) -> NoReturn:
    NETWORK_ATTEMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NETWORK_ATTEMPT_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"connector": connector, "pid": os.getpid()}, sort_keys=True))
        stream.write("\n")
        stream.flush()
    raise RuntimeError(f"child-process guard blocked {connector}")


def _deny(connector: str):
    def blocked(*args: object, **kwargs: object) -> NoReturn:
        return _record_and_block(connector, *args, **kwargs)

    blocked._pdm_blocked_connector = connector
    return blocked


socket.create_connection = _deny("socket.create_connection")
socket.socket.connect = _deny("socket.socket.connect")
socket.socket.connect_ex = _deny("socket.socket.connect_ex")

import psycopg2  # noqa: E402

psycopg2.connect = _deny("psycopg2.connect")

import pyodbc  # noqa: E402

pyodbc.connect = _deny("pyodbc.connect")

import torch  # noqa: E402

BLOCKED_CONNECTORS = sorted(
    {
        connector
        for guarded_callable in (
            psycopg2.connect,
            pyodbc.connect,
            socket.create_connection,
            socket.socket.connect,
            socket.socket.connect_ex,
        )
        if (connector := getattr(guarded_callable, "_pdm_blocked_connector", None))
    }
)

report = {
    "blocked_connectors": BLOCKED_CONNECTORS,
    "cuda_available": torch.cuda.is_available(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "environment_keys": sorted(os.environ),
    "executable": sys.executable,
    "guard_file": str(Path(__file__).resolve()),
    "guard_loaded": True,
    "pid": os.getpid(),
    "torch_version": torch.__version__,
    "torch_version_cuda": torch.version.cuda,
}
GUARD_REPORT_PATH.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
