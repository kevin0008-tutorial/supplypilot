"""LightGBM 需求预测基线（传统机器学习路线）。

和 Chronos 这种"基础模型"路线形成对照：LightGBM 不懂"时间序列"，它只会学
"一堆特征 → 目标值"的映射。所以关键在**特征工程**——我们手工把时间序列的信息
（滞后值、滚动均值、星期几）做成特征喂给它。

LightGBM 是梯度提升树（GBDT）：把很多棵浅决策树依次相加，每棵树去纠正前面所有树
合起来还没拟合好的残差。工程上快、省内存、在表格数据上常是强基线。
文档/原理见课件⑦里的外链。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_MIN_HISTORY = 28  # 需要至少 28 天历史才能算出 lag28 / 滚动均值


def lgbm_available() -> bool:
    """LightGBM 能否真正加载（Mac 还需 libomp、scikit-learn）。"""
    try:
        from lightgbm import LGBMRegressor  # noqa: F401
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def _features(series: np.ndarray, t: int) -> list[float]:
    """用 series[:t] 的信息，构造预测 series[t] 用的特征向量。"""
    return [
        series[t - 1],                 # 昨天
        series[t - 7],                 # 上周同一天
        series[t - 14],
        series[t - 28],
        float(series[t - 7:t].mean()),   # 近 7 天滚动均值
        float(series[t - 28:t].mean()),  # 近 28 天滚动均值
        float(t % 7),                    # 星期几（0–6）
    ]


def lgbm_forecaster(units: Sequence[float], horizon: int, seed: int = 42) -> np.ndarray:
    """训练 LightGBM 并递归预测未来 horizon 天（返回逐日点预测）。"""
    from lightgbm import LGBMRegressor  # 延迟导入：没装/缺 libomp 时不影响其它模块

    y = np.asarray(units, dtype=float)
    if y.size < _MIN_HISTORY + 7:
        # 历史太短，退回季节性朴素法（避免硬报错）
        from forecast.baseline import seasonal_naive_daily
        return seasonal_naive_daily(y, horizon)

    X = np.array([_features(y, t) for t in range(_MIN_HISTORY, y.size)])
    target = y[_MIN_HISTORY:]

    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
        random_state=seed, verbosity=-1,
    )
    model.fit(X, target)

    # 递归预测：每预测一天就接到序列尾部，用于算下一天的特征。
    series = list(y)
    preds = []
    for _ in range(horizon):
        feat = np.array([_features(np.asarray(series), len(series))])
        yhat = float(model.predict(feat)[0])
        yhat = max(0.0, yhat)          # 销量不为负
        preds.append(yhat)
        series.append(yhat)
    return np.asarray(preds)
