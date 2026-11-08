# SupplyPilot

零售补货决策 Copilot：帮门店补货和采购人员判断哪些商品有断货风险、该补多少、对应供应商的合同条款是什么，并在人工审批后生成补货单。

> 直通硅谷 AI 工程项目课的课程项目。本分支 `lession_1`：仓库骨架 + 第 1 节课堂实战（新增 `/health` 健康检查接口和对应测试）。

## 快速开始

需要先安装 Docker Desktop、uv、Git 和 make。

```bash
cp .env.example .env   # 创建本地配置
uv sync                # 安装本地 Python 依赖（用于测试）
make up                # 启动 Postgres 和 API
make test              # 运行测试
```

服务启动后：

- API：http://localhost:8000
- 接口文档：http://localhost:8000/docs

`make help` 列出所有命令。

## 项目结构

```
api/                 后端服务（FastAPI）
tests/               自动化测试（pytest）
docker-compose.yml   本地服务编排
Dockerfile           API 镜像
Makefile             常用命令
pyproject.toml       项目信息和依赖
uv.lock              依赖锁定版本
```
