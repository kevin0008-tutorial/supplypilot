import numpy as np
import pytest

from forecast.backtest import wape
from forecast.lightgbm_model import lgbm_available, lgbm_forecaster

if not lgbm_available():
    pytest.skip("LightGBM 不可用（未装或缺 libomp）", allow_module_level=True)


def test_lgbm_shape_and_nonnegative():
    units = ([5, 5, 6, 6, 7, 9, 8] * 40)
    pred = lgbm_forecaster(units[:-14], horizon=14)
    assert pred.shape == (14,)
    assert np.all(pred >= 0)
    assert np.all(np.isfinite(pred))


def test_lgbm_beats_trivial_zero_forecast():
    rng = np.random.default_rng(0)
    base = np.array([5, 5, 6, 6, 7, 9, 8] * 40, dtype=float)
    units = base + rng.normal(0, 0.5, size=base.size)   # 周期 + 小噪声
    actual = units[-14:]
    pred = lgbm_forecaster(units[:-14], horizon=14)
    assert wape(actual, pred) < wape(actual, np.zeros(14))


def test_lgbm_falls_back_on_short_history():
    pred = lgbm_forecaster([1, 2, 3], horizon=7)   # 太短 → 退回季节性朴素法
    assert pred.shape == (7,)
