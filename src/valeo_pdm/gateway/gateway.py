import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

# 模拟微服务注册表 (实际项目中可存储在 Redis 或数据库中)
ROUTE_MAP = {
    "user": "http://localhost:8001",
    "order": "http://localhost:8002",
}

# 1. 使用 lifespan 管理全局的 httpx 客户端连接池
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：初始化一个全局复用的异步客户端
    # limits 设定了最大连接数，这是高并发的关键
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    app.state.client = httpx.AsyncClient(limits=limits)
    print("Gateway Started. Connection pool initialized.")
    yield
    # 关闭时：优雅清理连接池
    await app.state.client.aclose()
    print("Gateway Shutdown. Connection pool closed.")

app = FastAPI(title="FastAPI Microservice Gateway", lifespan=lifespan)

# 2. 核心路由：捕获所有带有服务前缀的请求
@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def gateway_proxy(service: str, path: str, request: Request):
    # 查找目标服务 URL
    target_base = ROUTE_MAP.get(service)
    if not target_base:
        raise HTTPException(status_code=404, detail="Service not found in registry")

    target_url = f"{target_base}/{path}"

    # 处理 Headers (必须剔除原有的 host 字段，否则目标服务器可能会拒绝请求)
    headers = dict(request.headers)
    headers.pop("host", None)

    client: httpx.AsyncClient = request.app.state.client

    try:
        # 3. 构建转发请求
        # 注意：这里我们直接透传了前端的请求体 request.stream()
        httpx_req = client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=request.query_params,
            content=request.stream()
        )

        # 4. 发送请求并获取流式响应
        # stream=True 确保网关不会将整个响应体加载到内存中
        httpx_resp = await client.send(httpx_req, stream=True)

        # 5. 将后端响应以流的形式返回给客户端
        return StreamingResponse(
            content=httpx_resp.aiter_raw(), # 异步迭代获取响应块
            status_code=httpx_resp.status_code,
            headers=dict(httpx_resp.headers),
            # 重要：当响应发送完毕后，由后台任务关闭后端的响应流
            background=BackgroundTask(httpx_resp.aclose)
        )

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Target service is down")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Target service timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gateway internal error: {str(e)}")