"""FastMCP STDIO 与 Streamable HTTP 入口。"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from pdm_mcp.client import PdmApiClient
from pdm_mcp.config import HttpSettings, Settings
from pdm_mcp.schemas import DataSource, ExecutionMode, TrainParams
from pdm_mcp.tools import PdmTools


def create_server(
    settings: Settings | None = None,
    *,
    api_client: PdmApiClient | None = None,
    http_settings: HttpSettings | None = None,
) -> FastMCP:
    settings = settings or Settings.from_env()
    http_settings = http_settings or HttpSettings()
    client = api_client or PdmApiClient(settings)
    tools = PdmTools(settings, client)
    transport_security = None
    if http_settings.allowed_hosts:
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(http_settings.allowed_hosts),
            allowed_origins=list(http_settings.allowed_origins),
        )
    mcp = FastMCP(
        "pdm-algorithm",
        instructions=(
            "先做健康检查和模型精确匹配。训练必须先调用 pdm_prepare_training，"
            "把完整计划展示给用户并等待下一轮明确确认，之后才可调用 pdm_train_model；"
            "训练请求不得自动重试。"
        ),
        host=http_settings.host,
        port=http_settings.port,
        streamable_http_path=http_settings.path,
        # PreviewStore 仍为进程内状态，不能开启多 worker 的无状态 HTTP 模式。
        stateless_http=False,
        json_response=False,
        transport_security=transport_security,
    )

    @mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def http_health(_: Request) -> JSONResponse:
        """仅表示 MCP HTTP 进程存活；上游 PdM API 健康由 MCP 工具检查。"""

        return JSONResponse({"status": "ok", "server": "pdm-algorithm"})

    @mcp.tool(
        name="pdm_health_check",
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def pdm_health_check() -> dict[str, Any]:
        """检查 PdM API 是否可用。"""

        return await tools.health_check()

    @mcp.tool(
        name="pdm_list_models",
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def pdm_list_models(
        equipment_code: str | None = None,
        meas_code: str | None = None,
        model_type: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """列出模型场景；所有过滤条件均为精确匹配。"""

        return await tools.list_models(equipment_code, meas_code, model_type, source)

    @mcp.tool(
        name="pdm_check_training_data",
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def pdm_check_training_data(
        equipment_code: str,
        meas_code: str,
        source: DataSource,
        seq_len: int = 672,
        label_len: int = 192,
        pred_len: int = 288,
        stride: int = 1,
        freq: str = "15min",
        days_back: int = 365,
        data_path: str | None = None,
    ) -> dict[str, Any]:
        """按实际清洗与滑窗规则检查训练数据是否可用。CSV 路径仅允许 data/ 下相对路径。"""

        return await tools.check_training_data(
            equipment_code,
            meas_code,
            source,
            seq_len,
            label_len,
            pred_len,
            stride,
            freq,
            days_back,
            data_path,
        )

    @mcp.tool(
        name="pdm_predict",
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    async def pdm_predict(
        equipment_code: str,
        meas_code: str,
        model_info_id: str | None = None,
        model_type: str | None = None,
        max_points: int = 100,
    ) -> dict[str, Any]:
        """执行预测并返回统计值及最多 100 个均匀抽样点；不接收原始历史数据。"""

        return await tools.predict(equipment_code, meas_code, model_info_id, model_type, max_points)

    @mcp.tool(
        name="pdm_prepare_training",
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    async def pdm_prepare_training(
        equipment_code: str,
        meas_code: str,
        model_info_id: str,
        data_source: DataSource,
        train_params: TrainParams | None = None,
        model_type: Literal["informer", "autoformer"] | None = None,
        execution_mode: ExecutionMode = "local_only",
    ) -> dict[str, Any]:
        """无副作用地解析有效训练计划并检查数据，成功后签发 15 分钟单次预览凭证。"""

        return await tools.prepare_training(
            equipment_code,
            meas_code,
            model_info_id,
            data_source,
            train_params,
            model_type,
            execution_mode,
        )

    @mcp.tool(
        name="pdm_train_model",
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    async def pdm_train_model(preview_id: str) -> dict[str, Any]:
        """消费一次预览凭证并训练。只能在下一轮得到用户明确确认后调用，绝不自动重试。"""

        return await tools.train_model(preview_id)

    @mcp.tool(
        name="pdm_get_training_status",
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def pdm_get_training_status(
        equipment_code: str,
        meas_code: str,
        model_info_id: str,
        model_type: Literal["informer", "autoformer"] | None = None,
    ) -> dict[str, Any]:
        """查询某个 ModelInfoID 的本地训练状态；结果未知时应调用本工具而不是重试训练。"""

        return await tools.get_training_status(equipment_code, meas_code, model_info_id, model_type)

    # 仅供同进程测试取业务实现，不参与 MCP 协议。
    mcp._pdm_tools = tools  # type: ignore[attr-defined]
    return mcp


def main() -> None:
    """以 STDIO 方式启动；stdout 只留给 MCP JSON-RPC。"""

    create_server().run(transport="stdio")


def http_main() -> None:
    """以 Streamable HTTP 方式启动；默认仅监听本机 127.0.0.1:8765/mcp。"""

    create_server(http_settings=HttpSettings.from_env()).run(transport="streamable-http")


if __name__ == "__main__":
    main()
