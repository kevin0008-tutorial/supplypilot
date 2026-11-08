"""基线预测：季节性朴素法 + 窗口需求分位数。

两个作用：
1. `seasonal_naive_daily`：最便宜的逐日点预测（"上周同一天"），作为回测基线和
   Chronos 没装时的兜底预测。
2. `window_demand_quantile`：补货公式要的核心量——未来 L+R 天总需求的 0.95 分位数。
   直接用历史“窗口总需求”的经验分位数，再按近期水平缩放，**不假设需求服从正态分布**
   （间歇性需求根本不是正态的，这是第2节要讲的关键点）。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SEASON = 7  # 周季节性


def seasonal_naive_daily(units: Sequence[float], horizon: int, season: int = SEASON) -> np.ndarray:
    """未来 horizon 天 = 最近一个季节周期的循环重复。"""
    arr = np.asarray(units, dtype=float)
    if arr.size == 0:
        return np.zeros(horizon)
    last = arr[-season:] if arr.size >= season else arr
    reps = int(np.ceil(horizon / last.size))
    return np.tile(last, reps)[:horizon]


def window_demand_quantile(
    units: Sequence[float], window: int, q: float = 0.95
) -> float:
    """未来 `window`(=L+R) 天总需求的 q 分位数。

    做法：对历史日销量取长度为 window 的滑动窗口求和，得到一堆“窗口总需求”，
    取它们的经验 q 分位数；再乘以 (近期日均 / 历史日均) 做水平校正（近期更旺就调高）。
    """
    arr = np.asarray(units, dtype=float)
    if arr.size == 0 or window <= 0:
        return 0.0

    if arr.size >= window:
        # 长度为 window 的滑动窗口求和。
        window_sums = np.convolve(arr, np.ones(window), mode="valid")
        base_q = float(np.quantile(window_sums, q))
    else:
        # 历史不足一个窗口：退化成“日分位数 × window”。
        base_q = float(np.quantile(arr, q)) * window

    overall_mean = float(arr.mean())
    recent_mean = float(arr[-window:].mean()) if arr.size >= window else overall_mean
    if overall_mean > 0:
        ratio = recent_mean / overall_mean
        ratio = min(max(ratio, 0.5), 2.0)   # 限幅，避免个别极端值把结果拉飞
    else:
        ratio = 1.0

    return max(0.0, base_q * ratio)
