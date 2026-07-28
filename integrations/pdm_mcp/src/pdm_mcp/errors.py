"""对 MCP 调用方稳定且不泄露内部路径的错误协议。"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorPayload:
    code: str
    message: str
    retryable: bool = False
    http_status: int | None = None

    def as_dict(self) -> dict[str, str | bool | int | None]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "http_status": self.http_status,
        }


class PdmToolError(RuntimeError):
    """FastMCP 会将该异常转换成 `isError=true` 的工具结果。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
    ) -> None:
        self.payload = ErrorPayload(code, message, retryable, http_status)
        rendered = json.dumps(self.payload.as_dict(), ensure_ascii=False, separators=(",", ":"))
        super().__init__(rendered)
