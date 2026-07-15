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

# 优先复制依赖文件以利用 Docker 缓存层
COPY pyproject.toml uv.lock ./

# 挂载 uv 缓存加速构建，并同步依赖包 (不安装项目自身的代码和开发依赖)
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_HTTP_TIMEOUT=120 \
    uv sync --frozen --no-install-project --no-dev --python 3.12


# 复制项目所有代码
COPY . /app

# 再次执行 sync，安装项目本身
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --python 3.12




RUN uv pip install torch torchvision --index-url  https://download.pytorch.org/whl/cu128 --no-cache


# 创建非特权用户，并将 /app 目录的所有权赋给它，避免权限问题
RUN useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app

# 切换到非特权用户
USER appuser

# 暴露端口
EXPOSE 10021

# 健康检查 (由于前面已经将 .venv 加入 PATH，这里使用的是虚拟环境内的 python)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10021/healthz').read()" || exit 1

# 启动命令 (此时系统能直接识别 uvicorn)
CMD ["uvicorn", "valeo_pdm.api.app:app", "--host", "0.0.0.0", "--port", "10021", "--workers", "2"]