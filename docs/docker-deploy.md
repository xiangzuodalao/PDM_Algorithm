# Docker 部署（生产）

本项目推荐以容器方式部署 FastAPI 服务，并通过挂载配置文件管理数据库连接与模型配置。

默认 Docker 镜像使用仅 CPU 的 PyTorch，且不会请求 NVIDIA 设备。它保留训练与预测 API，但生产规模训练可能较慢。如需 GPU 镜像，必须使用独立的 Dockerfile/profile，并且只配置一个 CUDA/PyTorch 索引；不要向默认 lock 文件添加 CUDA 包。

Dockerfile 的 Ubuntu 与 uv 外部镜像摘要已在 Linux/amd64 本地构建中验证。
这些摘要不代表其他架构也已验证；在其他架构部署前应单独验证并更新。

## 0. 前置条件

- 安装 Docker / Docker Compose
- 准备配置文件：
  - `configs/model_registry.yaml`
  - `configs/postgres_config.json`（由 `configs/postgres_config.example.json` 复制并填写）

## 1. 构建镜像

在项目根目录执行：

```bash
docker build -t valeo-pdm:latest .
```

## 2. 使用 docker compose 启动

建议先在宿主机创建持久化目录，避免 Docker 自动创建为 `root:root` 导致训练/写文件权限问题：

```bash
mkdir -p artifacts data
```

```bash
docker compose up -d --build
```

检查：

```bash
docker compose ps
curl http://127.0.0.1:10021/healthz
```

## 3. 日志与停止

```bash
docker compose logs -f valeo-pdm-api
docker compose down
```

## 4. 配置说明

- `VALEO_PDM_POSTGRES_CONFIG`：数据库连接配置 JSON 的路径（建议挂载到容器内再指向它）
- `VALEO_PDM_MODEL_REGISTRY`：模型注册表 YAML 的路径（默认 `/app/configs/model_registry.yaml`）
- `VALEO_PDM_HTTP_PORT`：对外暴露的 HTTP 端口（默认 10021），对应 `docker-compose.yml`

为兼容既有部署，Compose 保留 `./src:/app/src` 源码挂载。该挂载会让容器
运行宿主机当前 checkout 中的源码，而不是镜像内复制的源码；部署时应确保
checkout 与所构建版本一致。

### Linux 服务器 + DB 跑在宿主机（常见）

容器内访问宿主机 PostgreSQL 时，`host=127.0.0.1` 不会指向宿主机，而是指向“容器自己”。

本项目已在 `docker-compose.yml` 增加：

- `extra_hosts: ["host.docker.internal:host-gateway"]`

因此你可以在 `configs/postgres_config.json` 里把 `conn_str` 写成：

- `host=host.docker.internal port=5432 ...`

如果你的 DB 是远程服务器，把 `host=` 改成远程 IP/域名即可。

## 5. 生产建议

- 不要把 `configs/postgres_config.json` 直接提交到仓库（`.dockerignore`/`.gitignore` 已屏蔽）
- 建议在 k8s 或企业编排平台里使用 Secret 管理 DB 配置
- 如需更多并发，可调整 `Dockerfile` 中 `uvicorn --workers` 数量（与 CPU 核数相关）
