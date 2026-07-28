"""PdM FastAPI 的受控异步客户端。"""

from __future__ import annotations

from typing import Any

import httpx

from pdm_mcp.config import Settings
from pdm_mcp.errors import PdmToolError

_SAFE_HTTP_MESSAGES: dict[int, tuple[str, str, bool]] = {
    400: ("PDM_BAD_REQUEST", "请求参数不符合 PdM 服务约束", False),
    404: ("PDM_NOT_FOUND", "未找到请求的场景、模型或训练状态", False),
    409: ("PDM_CONFLICT", "训练计划或 ModelInfoID 存在冲突", False),
    422: ("PDM_VALIDATION_ERROR", "请求未通过 PdM API 校验", False),
    428: ("PDM_PLAN_REQUIRED", "服务要求先预览并确认训练计划", False),
}


class PdmApiClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        training_dispatch: bool = False,
    ) -> dict[str, Any]:
        timeout = (
            self.settings.train_timeout_seconds
            if training_dispatch
            else self.settings.api_timeout_seconds
        )
        try:
            async with httpx.AsyncClient(
                base_url=f"{self.settings.api_base_url}/",
                timeout=timeout,
                transport=self.transport,
                # PdM API 通常位于 loopback/内网；不继承宿主机代理，避免请求绕出本机。
                trust_env=False,
            ) as client:
                response = await client.request(
                    method,
                    path.lstrip("/"),
                    json=json_body,
                    params=params,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if training_dispatch:
                raise PdmToolError(
                    "PDM_TRAIN_OUTCOME_UNKNOWN",
                    "训练请求已提交，但未能确认最终结果；请查询训练状态，禁止自动重试",
                    retryable=False,
                ) from exc
            code = (
                "PDM_API_TIMEOUT"
                if isinstance(exc, httpx.TimeoutException)
                else "PDM_API_UNAVAILABLE"
            )
            message = "PdM API 调用超时" if code == "PDM_API_TIMEOUT" else "无法连接 PdM API"
            raise PdmToolError(code, message, retryable=True) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            if training_dispatch:
                raise PdmToolError(
                    "PDM_TRAIN_OUTCOME_UNKNOWN",
                    "训练请求已提交，但响应无法解析；请查询训练状态，禁止自动重试",
                    http_status=response.status_code,
                ) from exc
            raise PdmToolError(
                "PDM_INVALID_RESPONSE",
                "PdM API 返回了无法解析的响应",
                retryable=response.status_code >= 500,
                http_status=response.status_code,
            ) from exc

        if response.is_error:
            raise self._http_error(
                response.status_code,
                payload,
                training_dispatch,
                response.headers.get("x-pdm-error-code"),
            )
        if not isinstance(payload, dict):
            if training_dispatch:
                raise PdmToolError(
                    "PDM_TRAIN_OUTCOME_UNKNOWN",
                    "训练请求已提交，但响应结构无效；请查询训练状态，禁止自动重试",
                    http_status=response.status_code,
                )
            raise PdmToolError(
                "PDM_INVALID_RESPONSE",
                "PdM API 响应结构无效",
                http_status=response.status_code,
            )
        return payload

    @staticmethod
    def _http_error(
        status_code: int,
        payload: Any,
        training_dispatch: bool,
        upstream_code: str | None,
    ) -> PdmToolError:
        detail = payload.get("detail", "") if isinstance(payload, dict) else ""
        detail_text = detail if isinstance(detail, str) else ""
        if status_code == 409:
            lowered = detail_text.lower()
            if upstream_code == "PLAN_HASH_MISMATCH" or (
                "hash" in lowered or "计划" in detail_text or "preview" in lowered
            ):
                return PdmToolError(
                    "PDM_PLAN_CHANGED",
                    "训练计划已变化，请重新预览并再次确认",
                    http_status=status_code,
                )
            if upstream_code == "TRAINING_ALREADY_RUNNING" or (
                "正在" in detail_text or "running" in lowered
            ):
                return PdmToolError(
                    "PDM_TRAINING_BUSY",
                    "相同 ModelInfoID 的训练正在执行",
                    http_status=status_code,
                )
        if status_code >= 500:
            if training_dispatch:
                return PdmToolError(
                    "PDM_TRAIN_OUTCOME_UNKNOWN",
                    "训练请求返回服务端错误，最终结果未知；请查询训练状态，禁止自动重试",
                    http_status=status_code,
                )
            return PdmToolError(
                "PDM_UPSTREAM_ERROR",
                "PdM API 内部错误",
                retryable=True,
                http_status=status_code,
            )
        code, message, retryable = _SAFE_HTTP_MESSAGES.get(
            status_code,
            ("PDM_HTTP_ERROR", "PdM API 拒绝了请求", False),
        )
        return PdmToolError(
            code,
            message,
            retryable=retryable,
            http_status=status_code,
        )
