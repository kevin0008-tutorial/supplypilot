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

**三方回测对比（季节性 vs LightGBM vs Chronos-2）：**

```bash
make forecast-report                                 # 季节性 + LightGBM
uv sync --extra forecast                             # 装 torch + chronos（可选、较大）
uv run --extra forecast python -m forecast.report    # 三方（含 Chronos）
```

LightGBM（`forecast/lightgbm_model.py`）需要 OpenMP：**macOS 先 `brew install libomp`**，WSL/Ubuntu 一般自带。Chronos（`forecast/chronos.py`）是可选依赖（`forecast` extra，含 torch）。报告会自动跳过未安装的方法；Chronos 在单独子进程里跑以避开 torch 与 LightGBM 的 OpenMP 冲突。

## 第 2.1 节补充课：三个"看得见"的小实验

补充课深挖 forecasting 三条路线的原理，配套 `forecast/explain.py` 把概念变成能跑、能看的输出。三条命令各自独立进程、各自懒加载，所以跑 Chronos 不会把 LightGBM 拉进来（天然避开 OpenMP 冲突）。序列默认取销量日度总和，没有真实 M5 时自动回退合成销量。

```bash
# 准备（Mac/WSL 相同）：拿到补充课分支并装依赖
git switch lession_2.1
uv sync                      # 主依赖（含 lightgbm、scikit-learn）
# macOS 还需一次：brew install libomp

# ① 季节性朴素法：打印"抄上周"的循环预测 + WAPE
make explain-seasonal        # = uv run python -m forecast.explain seasonal

# ② LightGBM：num_leaves 扫描现场演示过拟合/偏差-方差 + 特征重要性
make explain-lgbm            # = uv run python -m forecast.explain lgbm

# ③ Chronos：分位数扇形（概率预测）。先装可选依赖（含 torch，较大）
uv sync --extra forecast
make explain-chronos         # = uv run --extra forecast python -m forecast.explain chronos
```

配套课件（相对链接、离线可开）在 `courseware/lesson2-1-*.html`。

## 第 3 节：大模型推理服务部署

`serving/` 下四个脚本，配合 `courseware/lession03/` 的课件。**`benchmark`、`kvcache_calc`、`openai_client_demo` 只用 Python 标准库**，Kaggle/Colab/本地零额外安装即可跑；只有 `kv_cache_demo` 需要 torch+transformers。

```bash
git switch lession_3

# ① 显存 / KV Cache 估算（纯标准库，课上现场算）
make kvcalc
uv run python -m serving.kvcache_calc --model qwen2.5-7b --seq 4096 --vram 8 --weight-dtype int4

# ② 开/关 KV Cache 的生成提速（HF Transformers，CPU 即可；首次下载 gpt2 约 250MB）
uv sync --extra serving
make kv-demo                 # gpt2 上实测约 4× 加速，输出完全一致

# ③ 在免费 GPU 上起 vLLM 服务并压测：在 Kaggle/Colab 打开并运行
#    serving/vllm_server.ipynb （T4/P100，开源免登录模型 Qwen2.5-1.5B-Instruct）

# ④ 压测任意 OpenAI 兼容服务（vLLM 或 Ollama），量 TTFT/TPOT/吞吐
make bench MODEL=Qwen/Qwen2.5-1.5B-Instruct BASE=http://localhost:8000/v1
# 打 Ollama：先 ollama pull qwen2.5:1.5b，再
make bench MODEL=qwen2.5:1.5b BASE=http://localhost:11434/v1
```

性能报告模板在 [docs/serving_report.md](docs/serving_report.md)。三个核心指标：TTFT（首 token 延迟）、TPOT（每 token 时间）、吞吐量。

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

