"""可选：把真实的 M5（Kaggle）子集读成和本项目 sales 表一样的结构。

默认流程用 `etl.gen_sales` 的合成销量，离线、可复现、秒级可跑。想换成真实数据的学员：
1. 在 Kaggle 的 "M5 Forecasting - Accuracy" 比赛页接受规则并下载数据；
2. 把 calendar.csv、sell_prices.csv、sales_train_validation.csv 放到 data/raw/；
3. 调用 `load_m5_subset("data/raw")`，得到的 DataFrame 可直接喂给 etl.load.write_table。

原始数据不进 Git（见 .gitignore 的 data/raw/），只提交这个脚本。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etl import config


def load_m5_subset(raw_dir: str | Path) -> pd.DataFrame:
    raw = Path(raw_dir)
    calendar = pd.read_csv(raw / "calendar.csv")
    prices = pd.read_csv(raw / "sell_prices.csv")
    sales_wide = pd.read_csv(raw / "sales_train_validation.csv")

    # 只取目标门店 + 目标品类的子集。
    sub = sales_wide[
        (sales_wide["store_id"] == config.STORE_ID)
        & (sales_wide["dept_id"] == config.DEPT_ID)
    ].head(config.N_ITEMS)

    d_cols = [c for c in sub.columns if c.startswith("d_")]
    long = sub.melt(
        id_vars=["item_id", "dept_id", "store_id"],
        value_vars=d_cols,
        var_name="d",
        value_name="units",
    )
    # M5 用 d_1/d_2… 编号，用 calendar 映射成真实日期。
    long = long.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left")
    long = long.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    long["d"] = pd.to_datetime(long["date"]).dt.date
    return long[["store_id", "item_id", "dept_id", "d", "units", "sell_price"]]
