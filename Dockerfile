# 基础镜像：官方 Python 3.12 精简版
FROM python:3.12-slim

# 安装 uv（固定版本，保证两台电脑一致）
RUN pip install --no-cache-dir uv==0.12.19

WORKDIR /app

# 先只复制依赖清单再安装：代码改动时这一层可以走缓存，构建更快
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 再复制代码
COPY api ./api

EXPOSE 8000
# --reload：代码一改，服务自动重启（开发阶段用）
CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
