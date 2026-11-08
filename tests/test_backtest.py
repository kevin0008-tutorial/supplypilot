from forecast.backtest import compare_baselines, rolling_origin_backtest, wape
from forecast.baseline import seasonal_naive_daily


def test_wape_basic():
    assert wape([10, 10, 10], [10, 10, 10]) == 0.0
    assert wape([10, 0, 10], [8, 0, 12]) == (2 + 0 + 2) / 20


def test_wape_handles_all_zero_actual():
    # 实际全 0：WAPE 的分母会是 0，不能崩。完美预测(全 0)应是 0，否则 inf。
    assert wape([0, 0], [0, 0]) == 0.0
    assert wape([0, 0], [1, 0]) == float("inf")


def test_seasonal_naive_perfect_on_periodic_series():
    units = [5, 5, 6, 6, 7, 9, 8] * 30     # 严格周期序列
    result = rolling_origin_backtest(units, seasonal_naive_daily, horizon=14, folds=4)
    assert result["wape"] < 1e-9           # 周期数据上季节性朴素法应几乎零误差


def test_seasonal_beats_moving_average_on_seasonal_data():
    units = [5, 5, 6, 6, 7, 9, 8] * 30
    scores = compare_baselines(units, horizon=14, folds=4)
    assert scores["seasonal_naive"] <= scores["moving_average"]
