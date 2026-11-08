"""读取真实的 M5（Kaggle）子集，结构和本项目 sales 表完全一致。

第2节**默认用真实 M5 销量**：只要 data/raw/ 下有 M5 的三个 CSV，`make data` 和
`make demo` 就自动用它们；没有就回退到 etl.gen_sales 的合成销量（保证任何机器都能跑）。

怎么准备数据（老师和学员各自做一次）：
1. 在 Kaggle "M5 Forecasting - Accuracy" 比赛页接受规则并下载；
2. 把 calendar.csv、sell_prices.csv、sales_train_validation.csv 放到 data/raw/
   （可以用软链接，不必复制几百 MB）。
原始数据不进 Git（见 .gitignore 的 data/raw/），只提交这个脚本。

注意：M5 只有销量和价格，**没有**物流/在途和异常事件——那些仍由 etl 合成，
再按 item_id 和真实销量拼在一起。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etl import config

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
_REQUIRED = ["calendar.csv", "sell_prices.csv", "sales_train_validation.csv"]


def m5_available(raw_dir: str | Path = RAW_DIR) -> bool:
    raw = Path(raw_dir)
    return all((raw / f).exists() for f in _REQUIRED)


def load_m5_subset(raw_dir: str | Path = RAW_DIR) -> pd.DataFrame:
    raw = Path(raw_dir)
    items = set(config.item_ids())

    calendar = pd.read_csv(raw / "calendar.csv", usecols=["d", "date", "wm_yr_wk"])
    sales_wide = pd.read_csv(raw / "sales_train_validation.csv")

    # 精确取目标门店 + 我们那 20 个 item_id（和合成的供应商/库存等表对齐）。
    sub = sales_wide[
        (sales_wide["store_id"] == config.STORE_ID)
        & (sales_wide["item_id"].isin(items))
    ]

    d_cols = [c for c in sub.columns if c.startswith("d_")]
    long = sub.melt(
        id_vars=["item_id", "dept_id", "store_id"],
        value_vars=d_cols,
        var_name="d",
        value_name="units",
    ).merge(calendar, on="d", how="left")

    # 价格按 (店, 商品, 周) 关联；只读需要的列，避免整张 2 亿行的大表进内存。
    prices = pd.read_csv(
        raw / "sell_prices.csv",
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
    )
    prices = prices[(prices["store_id"] == config.STORE_ID) & (prices["item_id"].isin(items))]
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    long["d"] = pd.to_datetime(long["date"]).dt.date
    long["units"] = long["units"].astype(int)
    return long[["store_id", "item_id", "dept_id", "d", "units", "sell_price"]].sort_values(
        ["item_id", "d"]
    ).reset_index(drop=True)
