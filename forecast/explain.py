"""第 2.1 节（补充课）配套：三个"看得见"的小实验。

目的不是再写一个预测器，而是把补充课里讲的概念变成**能跑、能看**的输出：

- `seasonal`：季节性朴素法到底怎么"抄上周"——打印最近一周和它循环出来的预测。
- `lgbm`：用一次 **num_leaves 扫描**现场演示**过拟合 / 偏差-方差权衡**：
  训练集 WAPE 一路降，验证集 WAPE 先降后升（U 形）；再打印**特征重要性**。
- `chronos`：打印 Chronos 的**分位数扇形**（0.1 / 0.5 / 0.9），看"概率预测"长什么样。

每个子命令各自**独立进程**运行、各自**懒加载**依赖，所以跑 `chronos` 不会把 LightGBM
拉进来（反之亦然），天然避开 torch 与 LightGBM 的 OpenMP 冲突。

跑法（任选其一；序列默认取销量日度总和，没有真实 M5 时自动用合成销量）：

    uv run python -m forecast.explain seasonal
    uv run python -m forecast.explain lgbm
    uv sync --extra forecast                        # 先装 torch + chronos
    uv run --extra forecast python -m forecast.explain chronos
"""
from __future__ import annotations

import argparse

import numpy as np


# --------------------------------------------------------------------------- #
# 数据：销量日度总和（和 forecast.report 用的是同一条序列）
# --------------------------------------------------------------------------- #
def daily_total_series() -> np.ndarray:
    from etl.sales_source import load_sales, sales_source_name

    sales = load_sales()
    daily = sales.groupby("d")["units"].sum().sort_index().to_numpy(dtype=float)
    print(f"销量来源：{sales_source_name()}   序列长度：{daily.size} 天\n")
    return daily


# --------------------------------------------------------------------------- #
# 1) 季节性朴素法：把"上周同一天"循环出来
# --------------------------------------------------------------------------- #
def seasonal_demo(y: np.ndarray, horizon: int = 14, season: int = 7) -> dict:
    from forecast.backtest import wape
    from forecast.baseline import seasonal_naive_daily

    train, actual = y[:-horizon], y[-horizon:]
    last_week = train[-season:]
    pred = seasonal_naive_daily(train, horizon, season)
    return {
        "last_week": last_week,
        "pred": pred,
        "actual": actual,
        "wape": wape(actual, pred),
    }


def _print_seasonal(y: np.ndarray, horizon: int) -> None:
    r = seasonal_demo(y, horizon)
    print("季节性朴素法 = 把最近一个周期（7 天）首尾相接地重复。\n")
    print("最近一周（训练集最后 7 天）：")
    print("  " + "  ".join(f"{v:6.1f}" for v in r["last_week"]))
    print(f"\n未来 {horizon} 天预测（就是上面那 7 个数循环出来）：")
    print("  " + "  ".join(f"{v:6.1f}" for v in r["pred"]))
    print(f"\n真实值：")
    print("  " + "  ".join(f"{v:6.1f}" for v in r["actual"]))
    print(f"\n这一段的 WAPE = {r['wape']:.4f}   （Σ|实际-预测| / Σ|实际|，越小越好）")
    print("注意：它完全不会学趋势或节假日，只会机械地重复——这正是它作为"
          "\n'谁都得赢过它' 基线的意义。")


# --------------------------------------------------------------------------- #
# 2) LightGBM：num_leaves 扫描现场演示过拟合 / 偏差-方差
# --------------------------------------------------------------------------- #
FEATURE_NAMES = ["lag1", "lag7", "lag14", "lag28", "rollmean7", "rollmean28", "weekday"]


def _build_xy(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from forecast.lightgbm_model import _MIN_HISTORY, _features

    X = np.array([_features(y, t) for t in range(_MIN_HISTORY, y.size)])
    target = y[_MIN_HISTORY:]
    return X, target


def overfit_sweep(y: np.ndarray, val_days: int = 60) -> dict:
    """固定其它超参，只扫 num_leaves，比较训练集 vs 验证集的一步 WAPE。"""
    from lightgbm import LGBMRegressor

    from forecast.backtest import wape

    X, target = _build_xy(y)
    split = X.shape[0] - val_days
    X_tr, y_tr = X[:split], target[:split]
    X_va, y_va = X[split:], target[split:]

    rows = []
    for leaves in [2, 4, 8, 16, 31, 64, 127, 255]:
        model = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=leaves,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            random_state=42, verbosity=-1,
        )
        model.fit(X_tr, y_tr)
        rows.append({
            "num_leaves": leaves,
            "train_wape": wape(y_tr, model.predict(X_tr)),
            "val_wape": wape(y_va, model.predict(X_va)),
        })
    return {"rows": rows, "n_train": split, "n_val": val_days}


def feature_importance(y: np.ndarray) -> list[tuple[str, float]]:
    from lightgbm import LGBMRegressor

    X, target = _build_xy(y)
    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
        random_state=42, verbosity=-1, importance_type="gain",
    )
    model.fit(X, target)
    pairs = list(zip(FEATURE_NAMES, model.feature_importances_.astype(float)))
    total = sum(v for _, v in pairs) or 1.0
    return sorted([(n, v / total) for n, v in pairs], key=lambda t: -t[1])


def _print_lgbm(y: np.ndarray) -> None:
    print("A) num_leaves 扫描：同一份数据，只把树放大，看训练集 vs 验证集\n")
    sweep = overfit_sweep(y)
    print(f"   训练样本 {sweep['n_train']} 天，验证样本 {sweep['n_val']} 天（按时间切，不打乱）\n")
    print(f"   {'num_leaves':>10} {'训练WAPE':>10} {'验证WAPE':>10}   说明")
    print("   " + "-" * 52)
    best = min(sweep["rows"], key=lambda r: r["val_wape"])
    for r in sweep["rows"]:
        tag = ""
        if r is best:
            tag = "← 验证集最低（最佳复杂度）"
        elif r["num_leaves"] == sweep["rows"][-1]["num_leaves"]:
            tag = "← 训练最低但验证变差＝过拟合"
        print(f"   {r['num_leaves']:>10} {r['train_wape']:>10.4f} "
              f"{r['val_wape']:>10.4f}   {tag}")
    print("\n   读法：训练 WAPE 随树变大一路降（容量变强＝偏差降）；"
          "\n   验证 WAPE 先降后升（U 形）——升的那一段就是方差变大、开始背题。"
          "\n   这就是偏差-方差权衡，面试常问。")

    print("\nB) 特征重要性（gain 占比）：模型最看重哪些特征\n")
    for name, share in feature_importance(y):
        bar = "█" * max(1, round(share * 40))
        print(f"   {name:>11}  {share*100:5.1f}%  {bar}")
    print("\n   滚动均值（近期水平）和 lag1/weekday 往往占大头——"
          "\n   印证'是手工特征把时间规律喂给了树'，树本身并不懂时间。")


# --------------------------------------------------------------------------- #
# 3) Chronos：分位数扇形（概率预测）
# --------------------------------------------------------------------------- #
def chronos_fan(y: np.ndarray, horizon: int = 14,
                levels: tuple[float, ...] = (0.1, 0.5, 0.9)) -> dict:
    from forecast.chronos import _predict_quantiles, chronos_available

    if not chronos_available():
        raise ImportError("未安装 Chronos。先 `uv sync --extra forecast`。")
    q = _predict_quantiles(y[:-horizon], horizon, list(levels))  # (horizon, 3)
    return {"levels": levels, "quantiles": q, "actual": y[-horizon:]}


def _print_chronos(y: np.ndarray, horizon: int) -> None:
    r = chronos_fan(y, horizon)
    lv = r["levels"]
    print("Chronos 一次就吐出整条分位数扇形——不用假设正态，直接给不确定性区间。\n")
    print(f"   {'day':>4}  {'p'+str(int(lv[0]*100)):>8} {'p50(中位)':>10} "
          f"{'p'+str(int(lv[2]*100)):>8}   {'真实':>8}")
    print("   " + "-" * 50)
    for i in range(horizon):
        lo, mid, hi = r["quantiles"][i]
        print(f"   {i+1:>4}  {lo:>8.1f} {mid:>10.1f} {hi:>8.1f}   "
              f"{r['actual'][i]:>8.1f}")
    print("\n   p10–p90 之间就是 80% 预测区间。补货取高分位（如 p95）＝宁可备多一点，"
          "\n   少冒断货风险——和第 3 页的分位数补货公式正好对上。")
    print("   而且这是**零样本**：模型一个字都没在我们的数据上训练过。")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="第2.1节补充课：三个看得见的小实验")
    parser.add_argument("topic", choices=["seasonal", "lgbm", "chronos"],
                        help="要演示的主题")
    parser.add_argument("--horizon", type=int, default=14, help="预测天数（默认14）")
    args = parser.parse_args()

    y = daily_total_series()

    if args.topic == "seasonal":
        _print_seasonal(y, args.horizon)
    elif args.topic == "lgbm":
        from forecast.lightgbm_model import lgbm_available
        if not lgbm_available():
            print("LightGBM 不可用（Mac 需 `brew install libomp`）。")
            return
        _print_lgbm(y)
    elif args.topic == "chronos":
        _print_chronos(y, args.horizon)


if __name__ == "__main__":
    main()
