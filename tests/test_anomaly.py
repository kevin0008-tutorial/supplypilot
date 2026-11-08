import datetime as dt

from forecast.anomaly import detect_demand_anomalies


def _weekly_series(weeks: int):
    pattern = [5, 5, 6, 6, 7, 9, 8]  # 一周的形状
    return pattern * weeks


def test_detects_injected_spike():
    units = _weekly_series(20)              # 140 天平稳的周季节性
    spike_idx = 100
    units[spike_idx] = 60                   # 注入一个大尖峰
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(len(units))]

    anomalies = detect_demand_anomalies(dates, units, k=3.5)
    flagged_dates = {a["date"] for a in anomalies}

    assert dates[spike_idx] in flagged_dates
    spike = next(a for a in anomalies if a["date"] == dates[spike_idx])
    assert spike["kind"] == "spike"


def test_clean_series_has_no_anomalies():
    units = _weekly_series(20)
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(len(units))]
    assert detect_demand_anomalies(dates, units, k=3.5) == []
