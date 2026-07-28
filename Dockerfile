# 使用官方轻量级 Python 镜像
# FROM python:3.12-slim
FROM ubuntu:24.04

# 从官方镜像中直接复制已编译好的 uv 二进制文件
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN sed -i 's/deb.ubuntu.org/mirrors.tsinghua.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.org/mirrors.tsinghua.com/g' /etc/apt/sources.list && \
    apt-get update && \
    (sed -i 's/deb.debian.org/mirrors.tsinghua.com/g' /etc/apt/sources.list.d/debian.sources 2>dev/null || true) && \
    (sed -i 's/security.debian.org/mirrors.tsinghua.com/g' /etc/apt/sources.list.d/debian.sources 2>dev/null || true) && \
    (sed -i 's/deb.debian.org/mirrors.tsinghua.com/g' /etc/apt/sources.list 2>dev/null || true) && \
    (sed -i 's/security.debian.org/mirrors.tsinghua.com/g' /etc/apt/sources.list 2>dev/null || true) && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    ca-certificates \
    unixodbc \
    unixodbc-dev \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# 安装 Microsoft SQL Server ODBC 驱动
RUN mkdir -p /usr/share/keyrings && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/ubuntu/24.04/prod $(lsb_release -cs) main" | \
        tee /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装 PostgreSQL ODBC 驱动
RUN apt-get update && apt-get install -y odbc-postgresql \
    && apt-get clean

# Copy the application only after system dependencies are ready. .dockerignore
# excludes virtual environments, secrets, data, artifacts, and local integrations.
COPY . /app

# The frozen uv lock selects the explicit PyTorch CPU index from pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_HTTP_TIMEOUT=120 \
    uv sync --frozen --no-dev --python 3.12


# 创建非特权用户。代码与虚拟环境保持只读，仅预建运行时可写目录；
# 避免递归 chown 复制整套 PyTorch/CUDA 环境形成巨大的镜像层。
RUN useradd -m -u 10001 appuser \
    && install -d -o appuser -g appuser /app/artifacts /app/data

# 切换到非特权用户
USER appuser

# 暴露端口
EXPOSE 10021

# 健康检查 (由于前面已经将 .venv 加入 PATH，这里使用的是虚拟环境内的 python)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10021/healthz').read()" || exit 1

# 启动命令 (此时系统能直接识别 uvicorn)
CMD ["uvicorn", "valeo_pdm.api.app:app", "--host", "0.0.0.0", "--port", "10021", "--workers", "1"]
