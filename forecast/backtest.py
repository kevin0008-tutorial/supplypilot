"""回测：用历史数据公平地比较几种预测方法。

两条铁律（第2节重点）：
1. **按时间切分**，绝不能随机打乱——用未来预测过去等于作弊。
2. 用 **WAPE** 而不是 MAPE：间歇性需求里真实值常为 0，MAPE 的分母一除就爆炸/失效。
   WAPE = Σ|实际-预测| / Σ|实际|，用总量做分母，稳健。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from forecast.baseline import seasonal_naive_daily

Forecaster = Callable[[np.ndarray, int], np.ndarray]


def wape(actual: Sequence[float], pred: Sequence[float]) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.abs(a).sum()
    if denom == 0:
        return 0.0 if np.abs(p).sum() == 0 else float("inf")
    return float(np.abs(a - p).sum() / denom)


def moving_average_daily(units: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    arr = np.asarray(units, dtype=float)
    mean = arr[-window:].mean() if arr.size else 0.0
    return np.full(horizon, mean)


def rolling_origin_backtest(
    units: Sequence[float],
    forecaster: Forecaster,
    horizon: int = 14,
    folds: int = 4,
    season: int = 7,
) -> dict:
    """滚动起点回测：切好几个时间点，每个点只用它之前的数据预测之后 horizon 天。"""
    y = np.asarray(units, dtype=float)
    n = y.size
    min_train = max(season * 4, horizon * 2)
    if n < min_train + horizon:
        raise ValueError("历史太短，不够做回测")

    # 均匀选 folds 个切分点，最后一个切分点后至少留 horizon 天做验证。
    cuts = np.linspace(min_train, n - horizon, folds, dtype=int)
    per_fold = []
    for cut in cuts:
        train, actual = y[:cut], y[cut : cut + horizon]
        pred = np.asarray(forecaster(train, horizon), dtype=float)
        per_fold.append(wape(actual, pred))

    return {"wape": float(np.mean(per_fold)), "per_fold": [round(x, 4) for x in per_fold]}


def compare_baselines(units: Sequence[float], horizon: int = 14, folds: int = 4) -> dict:
    """同一份数据上比较季节性朴素法 vs 移动平均。Chronos 装了的话可另行加入对比。"""
    return {
        "seasonal_naive": rolling_origin_backtest(
            units, seasonal_naive_daily, horizon, folds
        )["wape"],
        "moving_average": rolling_origin_backtest(
            units, moving_average_daily, horizon, folds
        )["wape"],
    }
