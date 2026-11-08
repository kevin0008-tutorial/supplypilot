import numpy as np

from forecast.baseline import seasonal_naive_daily, window_demand_quantile


def test_seasonal_naive_repeats_last_week():
    units = list(range(14))  # 两周 0..13
    pred = seasonal_naive_daily(units, horizon=7)
    # 预测应等于最后 7 天（7..13）。
    assert pred.tolist() == [7, 8, 9, 10, 11, 12, 13]


def test_window_quantile_on_constant_series():
    # 常量序列：每个 14 天窗口总和都是 140，任何分位数都应是 140。
    units = [10.0] * 60
    assert window_demand_quantile(units, window=14, q=0.95) == 140.0


def test_window_quantile_is_nonnegative_and_reasonable():
    rng = np.random.default_rng(0)
    units = rng.poisson(3, size=200).astype(float)
    q = window_demand_quantile(units, window=14, q=0.95)
    assert q >= 0
    # 0.95 分位的 14 天总需求，应明显高于均值水平（14 × 3 = 42）。
    assert q > 42
