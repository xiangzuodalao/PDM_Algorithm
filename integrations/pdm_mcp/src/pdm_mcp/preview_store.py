"""进程内、短时、单次消费的训练预览存储。"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pdm_mcp.errors import PdmToolError


@dataclass
class PreviewEntry:
    payload: dict[str, Any]
    expires_monotonic: float
    expires_at: str
    state: Literal["ready", "consumed"] = "ready"


class PreviewStore:
    def __init__(self, ttl_seconds: int = 900) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, PreviewEntry] = {}
        self._lock = asyncio.Lock()

    async def create(self, payload: dict[str, Any]) -> tuple[str, str]:
        now = datetime.now(UTC)
        preview_id = str(uuid.uuid4())
        expires_at = (now + timedelta(seconds=self.ttl_seconds)).isoformat()
        entry = PreviewEntry(
            payload=copy.deepcopy(payload),
            expires_monotonic=time.monotonic() + self.ttl_seconds,
            expires_at=expires_at,
        )
        async with self._lock:
            self._prune_expired()
            self._entries[preview_id] = entry
        return preview_id, expires_at

    async def consume(self, preview_id: str) -> dict[str, Any]:
        async with self._lock:
            entry = self._entries.get(preview_id)
            if entry is None:
                self._prune_expired()
                raise PdmToolError(
                    "PDM_PREVIEW_NOT_FOUND",
                    "训练预览不存在；请重新准备训练计划",
                )
            if entry.expires_monotonic <= time.monotonic():
                del self._entries[preview_id]
                raise PdmToolError(
                    "PDM_PREVIEW_EXPIRED",
                    "训练预览已过期；请重新准备训练计划",
                )
            if entry.state == "consumed":
                raise PdmToolError(
                    "PDM_PREVIEW_ALREADY_USED",
                    "训练预览已被使用，禁止重试；请查询状态或重新准备新任务",
                )
            entry.state = "consumed"
            return copy.deepcopy(entry.payload)

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._entries.items() if entry.expires_monotonic <= now]
        for key in expired:
            del self._entries[key]
