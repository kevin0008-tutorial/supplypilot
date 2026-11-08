"""三方回测报告：季节性朴素法 vs LightGBM vs Chronos-2（装了才跑）。

在同一份数据、同一套"按时间切分 + WAPE"的规则下公平比较。这正是第2节作业要交的
"三方回测对比"。默认用真实 M5 的日度总销量（比单个 SKU 平稳，对比更清楚）。

跑法：
    make forecast-report                              # 季节性 + LightGBM
    uv sync --extra forecast                          # 先装 torch + chronos（可选）
    uv run --extra forecast python -m forecast.report # 三方（含 Chronos）

工程细节（macOS）：torch 自带一套 OpenMP，LightGBM 用系统的 libomp，两者在**同一个
进程**里会冲突、卡死。所以这里把 Chronos 放到**单独的子进程**里跑，主进程只管
LightGBM——两套原生运行时永不同处，稳。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import numpy as np

from etl.sales_source import load_sales, sales_source_name
from forecast.backtest import rolling_origin_backtest
from forecast.baseline import seasonal_naive_daily
from forecast.lightgbm_model import lgbm_available, lgbm_forecaster


def daily_total_series() -> np.ndarray:
    sales = load_sales()
    daily = sales.groupby("d")["units"].sum().sort_index()
    return daily.to_numpy(dtype=float)


def _chronos_installed() -> bool:
    # 只查有没有装，不 import（import torch 会把 OpenMP 拉进主进程）。
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("chronos") is not None
    )


def _chronos_backtest_subprocess(horizon: int, folds: int) -> dict | None:
    """在单独的子进程里跑 Chronos 回测，避开和 LightGBM 的 OpenMP 冲突。"""
    code = (
        "import json;"
        "from etl.sales_source import load_sales;"
        "from forecast.backtest import rolling_origin_backtest;"
        "from forecast.chronos import chronos_forecaster;"
        "y=load_sales().groupby('d')['units'].sum().sort_index().to_numpy(float);"
        f"r=rolling_origin_backtest(y, chronos_forecaster, horizon={horizon}, folds={folds});"
        "print('CHRONOS_JSON='+json.dumps(r))"
    )
    env = {"OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    import os
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env={**os.environ, **env},
    )
    for line in proc.stdout.splitlines():
        if line.startswith("CHRONOS_JSON="):
            return json.loads(line[len("CHRONOS_JSON="):])
    return None


def main(horizon: int = 14, folds: int = 4) -> None:
    y = daily_total_series()
    print(f"销量来源：{sales_source_name()}  序列长度：{y.size} 天  "
          f"回测：horizon={horizon}, folds={folds}\n")

    rows: list[tuple[str, dict]] = [
        ("季节性朴素法 (seasonal naive)",
         rolling_origin_backtest(y, seasonal_naive_daily, horizon, folds))
    ]
    if lgbm_available():
        rows.append(("LightGBM (特征工程+GBDT)",
                     rolling_origin_backtest(y, lgbm_forecaster, horizon, folds)))
    else:
        print("（LightGBM 不可用，跳过；Mac 需 brew install libomp）\n")

    if _chronos_installed():
        res = _chronos_backtest_subprocess(horizon, folds)
        if res:
            rows.append(("Chronos-2 (零样本基础模型)", res))
    else:
        print("（未装 Chronos，跳过；uv sync --extra forecast 后可纳入对比）\n")

    print(f"{'方法':<30}{'WAPE':>8}   逐折")
    print("-" * 62)
    for name, res in rows:
        print(f"{name:<30}{res['wape']:>8.4f}   {res['per_fold']}")
    print("\nWAPE 越小越好。基础要求：复杂方法至少不输给季节性朴素法。")


if __name__ == "__main__":
    main()
