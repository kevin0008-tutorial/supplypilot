"""需求异常检测。

"异常事件"（促销、停产）是外部原因，"异常值"是它在销量里的统计表现。这里做的是后者：
在**去掉周季节性之后**的残差上，用稳健的 MAD-z 分数找出尖峰和骤降。

为什么用中位数/MAD 而不用均值/标准差：销量里本来就有大尖峰，均值和标准差会被尖峰
自己带偏，导致“检测不出尖峰”。中位数和 MAD 对离群点稳健，更适合挑异常。
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np

SEASON = 7
_MAD_TO_STD = 1.4826  # 正态下把 MAD 换算成标准差的系数


def detect_demand_anomalies(
    dates: Sequence[dt.date],
    units: Sequence[float],
    k: float = 3.5,
    season: int = SEASON,
) -> list[dict]:
    """返回异常点列表：[{date, units, zscore, kind}]，kind 为 spike / drop。"""
    y = np.asarray(units, dtype=float)
    if y.size <= season:
        return []

    # 季节性残差：和上周同一天比。
    resid = y[season:] - y[:-season]
    idx = np.arange(season, y.size)

    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = mad * _MAD_TO_STD
    if scale == 0:  # 残差几乎不动（常量序列），用标准差兜底
        scale = resid.std()
    if scale == 0:
        return []

    z = (resid - med) / scale

    out = []
    for i, zi in zip(idx, z):
        if abs(zi) > k:
            out.append(
                {
                    "date": dates[i],
                    "units": float(y[i]),
                    "zscore": round(float(zi), 2),
                    "kind": "spike" if zi > 0 else "drop",
                }
            )
    return out
