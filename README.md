# SupplyPilot

零售补货决策 Copilot：帮门店补货和采购人员判断哪些商品有断货风险、该补多少、对应供应商的合同条款是什么，并在人工审批后生成补货单。

> 直通硅谷 AI 工程项目课的课程项目。本分支 `lession_2`：在第 1 节骨架之上，新增数据层（销量/物流在途/异常事件表）、时间序列预测、异常检测，以及带事件驱动调整的补货工具 `forecast_and_reorder`。

## 快速开始

需要先安装 Docker Desktop、uv、Git 和 make。

```bash
cp .env.example .env   # 创建本地配置
uv sync                # 安装本地 Python 依赖
make test              # 运行全部测试
make demo              # 离线跑一遍补货决策（不需要数据库）
```

用上数据库的完整流程：

```bash
make up                # 启动 Postgres（和 API）
make data              # 准备数据并灌进 Postgres
```

**销量数据：默认用真实 M5（Walmart）。** 在 Kaggle “M5 Forecasting - Accuracy” 接受规则并下载，把 `calendar.csv`、`sell_prices.csv`、`sales_train_validation.csv` 放到 `data/raw/`（软链接即可，几百 MB 不必复制；`data/raw/` 不进 Git）。`make data` / `make demo` 检测到就自动用真实 M5；没下载则回退到合成销量，项目照常可跑。物流/在途和异常事件 M5 没有，始终由 `etl/` 合成，再按 `item_id` 和真实销量拼在一起。

服务启动后：API http://localhost:8000 ，接口文档 http://localhost:8000/docs 。`make help` 列出所有命令。

## 项目结构

```
api/                 后端服务（FastAPI）
db/schema.sql        建表脚本（销量、供应商、库存、在途、事件）
etl/                 合成数据生成 + 入库（gen_*、load、build；load_m5 可选接真实 M5）
forecast/            预测与分析（baseline 分位数预测、anomaly 异常检测、backtest 回测、chronos 可选）
tools/               补货工具（replenishment 决策逻辑、repository 数据访问、demo 离线示例）
sql/analytics/       分析用 SQL 范例（断货风险、在途延误）
tests/               自动化测试（pytest）
docker-compose.yml   本地服务编排
Makefile             常用命令
pyproject.toml       项目信息和依赖
uv.lock              依赖锁定版本
```

## 第 2 节的核心：forecast_and_reorder

补货量用分位数算，**不假设需求服从正态分布**（间歇性需求不是正态的）：

```
Q_raw = max(0, D^0.95_{L+R} − 现有库存 − 能及时到货的在途量)
Q     = 向上取整到 MOQ 的整数倍
```

并做事件驱动调整：供应商停产→换备选供应商；航线拥堵/在途延误→拉长交期 L、剔除会晚到的在途货；召回→暂停补货；促销→需求按近期水平自动抬高。返回结果带 `provenance`，每个数字可追溯。

