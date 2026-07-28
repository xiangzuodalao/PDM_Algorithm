"""MCP Server 环境配置。"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import urlparse


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是布尔值")


def _tcp_port(name: str, default: int) -> int:
    value = _positive_int(name, default)
    if value > 65535:
        raise ValueError(f"{name} 必须在 1..65535 范围内")
    return value


def _http_host(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if (
        not value
        or "://" in value
        or "/" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} 必须是主机名或 IP 地址，不包含协议、路径或空白")
    return value


def _http_path(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if (
        not value.startswith("/")
        or value == "/"
        or "//" in value
        or "?" in value
        or "#" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} 必须是非根绝对 URL 路径，且不包含查询、片段或空白")
    return value.rstrip("/")


def _csv_values(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Settings:
    """只从环境读取部署参数，不读取平台密钥。"""

    api_base_url: str = "http://127.0.0.1:10021"
    api_timeout_seconds: float = 30.0
    train_timeout_seconds: float = 7200.0
    enable_train: bool = False
    preview_ttl_seconds: int = 900

    @classmethod
    def from_env(cls) -> Settings:
        base_url = os.getenv("PDM_API_BASE_URL", cls.api_base_url).strip().rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("PDM_API_BASE_URL 必须是 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("PDM_API_BASE_URL 不应包含认证信息")
        return cls(
            api_base_url=base_url,
            api_timeout_seconds=_positive_float("PDM_API_TIMEOUT_SECONDS", cls.api_timeout_seconds),
            train_timeout_seconds=_positive_float(
                "PDM_TRAIN_TIMEOUT_SECONDS", cls.train_timeout_seconds
            ),
            enable_train=_boolean("PDM_ENABLE_TRAIN", cls.enable_train),
            preview_ttl_seconds=_positive_int("PDM_PREVIEW_TTL_SECONDS", cls.preview_ttl_seconds),
        )


@dataclass(frozen=True)
class HttpSettings:
    """Streamable HTTP 监听配置；与 STDIO 配置分离，避免影响本地入口。"""

    host: str = "127.0.0.1"
    port: int = 8765
    path: str = "/mcp"
    allowed_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> HttpSettings:
        host = _http_host("PDM_MCP_HTTP_HOST", cls.host)
        allowed_hosts = _csv_values("PDM_MCP_HTTP_ALLOWED_HOSTS")
        if not _is_loopback_host(host) and not allowed_hosts:
            raise ValueError(
                "非本地 PDM_MCP_HTTP_HOST 必须同时设置 PDM_MCP_HTTP_ALLOWED_HOSTS；"
                "该入口仍须置于提供 TLS 与鉴权的网关后"
            )
        return cls(
            host=host,
            port=_tcp_port("PDM_MCP_HTTP_PORT", cls.port),
            path=_http_path("PDM_MCP_HTTP_PATH", cls.path),
            allowed_hosts=allowed_hosts,
            allowed_origins=_csv_values("PDM_MCP_HTTP_ALLOWED_ORIGINS"),
        )
