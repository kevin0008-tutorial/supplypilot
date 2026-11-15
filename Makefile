.PHONY: help up down ps logs test clean data demo forecast-report explain-seasonal explain-lgbm explain-chronos

help:  ## 列出所有命令
	@grep -E "^[a-z]+:.*##" Makefile | sed "s/:.*##/ —/"

data:  ## 生成合成数据，写出 CSV 并（数据库在跑时）灌入 Postgres
	uv run python -m etl.build

demo:  ## 离线跑一遍补货决策 demo（不需要数据库）
	uv run python -m tools.demo

forecast-report:  ## 三方回测对比：季节性 vs LightGBM vs Chronos（装了才跑）
	uv run python -m forecast.report

explain-seasonal:  ## 补充课实验：季节性朴素法怎么"抄上周"
	uv run python -m forecast.explain seasonal

explain-lgbm:  ## 补充课实验：LightGBM 过拟合/偏差-方差 + 特征重要性
	uv run python -m forecast.explain lgbm

explain-chronos:  ## 补充课实验：Chronos 分位数扇形（需 uv sync --extra forecast）
	uv run --extra forecast python -m forecast.explain chronos

up:  ## 构建镜像并在后台启动全部服务
	docker compose up -d --build

down:  ## 停止并删除容器（数据卷保留）
	docker compose down

ps:  ## 查看服务状态
	docker compose ps

logs:  ## 实时查看 API 日志（Ctrl+C 退出）
	docker compose logs -f api

test:  ## 运行全部测试
	uv run pytest -q

clean:  ## 停止服务并删除数据卷（数据库会被清空，慎用）
	docker compose down -v
