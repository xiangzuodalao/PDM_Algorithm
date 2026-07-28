# PdM MCP Server

这是 Valeo PdM Algorithm 的 MCP 适配器。它可通过 STDIO 或 Streamable HTTP 向客户端提供
七个受控工具，通过 HTTP 调用 PdM FastAPI，不直接读取数据库配置，也不依赖主项目的
PyTorch 环境。

## 安装与验证

```bash
cd /home/vm/code/PDM_Algorithm/integrations/pdm_mcp
uv sync --all-groups
uv run ruff check .
uv run pytest
```

以 STDIO 启动 MCP Server：

```bash
PDM_API_BASE_URL=http://127.0.0.1:10021 \
  uv run --frozen --no-dev pdm-mcp
```

STDIO 的标准输出属于 MCP JSON-RPC，不要在工具代码中使用 `print()` 输出日志。

以 Streamable HTTP 启动：

```bash
PDM_API_BASE_URL=http://127.0.0.1:10021 \
  uv run --frozen --no-dev pdm-mcp-http
```

默认地址为 `http://127.0.0.1:8765/mcp`，进程存活检查为
`http://127.0.0.1:8765/healthz`。HTTP 入口默认保持有状态会话，并与 STDIO 入口复用完全相同
的工具实现；两种入口按部署方式选择其一即可。

## 工具

| 工具 | 属性 | 用途 |
|---|---|---|
| `pdm_health_check` | 只读 | 检查 FastAPI 健康状态 |
| `pdm_list_models` | 只读 | 列出并精确过滤已注册场景 |
| `pdm_check_training_data` | 只读 | 按真实清洗、切分规则检查数据量 |
| `pdm_predict` | 只读 | 执行预测，返回统计值和最多 100 个抽样点 |
| `pdm_prepare_training` | 只读 | 解析最终计划、校验数据并签发单次预览凭证 |
| `pdm_train_model` | 写操作 | 消费预览凭证并训练；禁止自动重试 |
| `pdm_get_training_status` | 只读 | 查询训练清单状态 |

训练采用两阶段协议：先 `pdm_prepare_training`，向用户展示完整计划并结束当前轮；只有在
下一轮获得明确确认后，才可调用一次 `pdm_train_model`。预览凭证默认 15 分钟过期、只能消费
一次，并只保存在 MCP 进程内；重启 MCP 后须重新预览。`local_only` 是默认执行模式，不回写
SQL Server 状态、不上传图片。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `PDM_API_BASE_URL` | `http://127.0.0.1:10021` | FastAPI 基地址，仅支持 HTTP(S) |
| `PDM_API_TIMEOUT_SECONDS` | `30` | 普通 API 超时 |
| `PDM_TRAIN_TIMEOUT_SECONDS` | `7200` | 单次训练超时 |
| `PDM_ENABLE_TRAIN` | `false` | 是否允许消费凭证并触发训练 |
| `PDM_PREVIEW_TTL_SECONDS` | `900` | 预览凭证有效期 |
| `PDM_MCP_HTTP_HOST` | `127.0.0.1` | Streamable HTTP 监听地址 |
| `PDM_MCP_HTTP_PORT` | `8765` | Streamable HTTP 监听端口 |
| `PDM_MCP_HTTP_PATH` | `/mcp` | MCP HTTP 端点路径 |
| `PDM_MCP_HTTP_ALLOWED_HOSTS` | 空 | 逗号分隔的 Host allowlist；非本地监听时必填 |
| `PDM_MCP_HTTP_ALLOWED_ORIGINS` | 空 | 逗号分隔的浏览器 Origin allowlist |

MCP 客户端不会继承宿主机 HTTP 代理，避免 loopback/内网请求意外绕出本机。CSV 路径只允许
`data/` 下的 POSIX 相对路径。训练响应不会暴露 checkpoint、运行目录或平台配置路径。

当前 HTTP 入口本身不提供用户认证或 TLS。默认仅绑定 loopback；如需绑定 `0.0.0.0`，必须
显式配置 `PDM_MCP_HTTP_ALLOWED_HOSTS`，并将服务放在提供 TLS、认证、授权和限流的网关之后。
例如：

```bash
PDM_MCP_HTTP_HOST=0.0.0.0 \
PDM_MCP_HTTP_ALLOWED_HOSTS=pdm-mcp.internal.example.com \
  uv run --frozen --no-dev pdm-mcp-http
```

预览凭证仍保存在单个 MCP 进程内，因此当前不要启动多个 worker，也不要对 HTTP 实例做无
会话粘性的负载均衡。

反向代理必须在同一个 MCP 路径上放行 `POST`、`GET` 和 `DELETE`，关闭 SSE/响应缓冲，并将
读取与空闲超时设置得大于 `PDM_TRAIN_TIMEOUT_SECONDS`；否则长时间训练可能在代理层被提前
断开。

## 注册到 Codex

```bash
codex mcp add pdm-algorithm \
  --env PDM_API_BASE_URL=http://127.0.0.1:10021 \
  --env PDM_ENABLE_TRAIN=true \
  --env PDM_TRAIN_TIMEOUT_SECONDS=7200 \
  -- /home/vm/.local/bin/uv run \
  --project /home/vm/code/PDM_Algorithm/integrations/pdm_mcp \
  --frozen --no-dev pdm-mcp
```

由于模型训练可能超过 Codex 的默认工具超时，还需在 `~/.codex/config.toml` 对应服务段设置：

```toml
[mcp_servers.pdm-algorithm]
startup_timeout_sec = 30
tool_timeout_sec = 7300
default_tools_approval_mode = "writes"

[mcp_servers.pdm-algorithm.tools.pdm_train_model]
approval_mode = "prompt"
```

注册后新开一个 Codex 会话，再使用仓库 Skill `pdm-train-model`。当前会话不会动态加载刚注册的
MCP Server。

## 通过 Streamable HTTP 注册到 Codex

先让 `pdm-mcp-http` 作为常驻进程运行，再注册 URL：

```bash
codex mcp add pdm-algorithm-http --url http://127.0.0.1:8765/mcp
```

HTTP 服务端的 `PDM_*` 环境变量必须在启动 `pdm-mcp-http` 时设置，而不是放到 Codex 的远程
MCP 注册项中。注册后同样需要新开 Codex 会话。
