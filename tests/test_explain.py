import numpy as np
import pytest

from forecast.explain import FEATURE_NAMES, seasonal_demo
from forecast.lightgbm_model import lgbm_available


def _weekly_series(weeks: int = 40) -> np.ndarray:
    return np.array([5, 5, 6, 6, 7, 9, 8] * weeks, dtype=float)


def test_seasonal_demo_tiles_last_week():
    y = np.arange(60, dtype=float)       # 0..59
    r = seasonal_demo(y, horizon=7, season=7)
    # 训练集最后 7 天是 46..52，预测应就是它们循环出来。
    assert r["pred"].tolist() == [46, 47, 48, 49, 50, 51, 52]
    assert r["actual"].tolist() == [53, 54, 55, 56, 57, 58, 59]
    assert r["wape"] >= 0


@pytest.mark.skipif(not lgbm_available(), reason="LightGBM 不可用（未装或缺 libomp）")
def test_overfit_sweep_and_importance():
    from forecast.explain import feature_importance, overfit_sweep

    y = _weekly_series()
    sweep = overfit_sweep(y, val_days=40)
    assert len(sweep["rows"]) == 8
    # 训练 WAPE 应随 num_leaves 单调不增（容量变大、偏差下降）。
    train = [r["train_wape"] for r in sweep["rows"]]
    assert all(a >= b - 1e-9 for a, b in zip(train, train[1:]))

    imp = feature_importance(y)
    assert {n for n, _ in imp} == set(FEATURE_NAMES)
    assert abs(sum(v for _, v in imp) - 1.0) < 1e-6
