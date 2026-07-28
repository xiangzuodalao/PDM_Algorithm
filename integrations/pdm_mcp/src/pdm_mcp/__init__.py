"""Valeo PdM Algorithm MCP Server。"""

from __future__ import annotations


def main() -> None:
    """延迟导入，避免仅导入包时就读取环境变量。"""

    from pdm_mcp.server import main as run_server

    run_server()


__all__ = ["main"]
