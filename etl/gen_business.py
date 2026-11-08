"""合成业务表：供应商、商品主数据、库存快照。

这些是补货决策的“静态参数”（交期、起订量）和“当前状态”（现有库存），
固定随机种子生成，两台电脑完全一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etl import config

# 四个供应商，交期和起订量各不相同，方便讲“换供应商会改变 L 和 MOQ”。
SUPPLIERS = [
    # supplier_id, name,            lead_time_days, moq, unit_cost
    ("SUP_A", "Alpha Foods",   7,  50,  2.50),
    ("SUP_B", "Bravo Supply", 10, 100,  2.10),
    ("SUP_C", "Cedar Trading", 14, 24,  2.80),
    ("SUP_D", "Delta Wholesale", 5, 200, 2.35),
]


def gen_suppliers() -> pd.DataFrame:
    return pd.DataFrame(
        SUPPLIERS,
        columns=["supplier_id", "name", "lead_time_days", "moq", "unit_cost"],
    )


def gen_item_master() -> pd.DataFrame:
    """给每个商品分配主供应商和备选供应商（两者必须不同）。"""
    items = config.item_ids()
    supplier_ids = [s[0] for s in SUPPLIERS]
    n = len(supplier_ids)
    rows = []
    for idx, item in enumerate(items):
        primary = supplier_ids[idx % n]
        # 备选取下一个供应商，保证和主供应商不同——停产时才有得切换。
        alt = supplier_ids[(idx + 1) % n]
        rows.append((item, config.DEPT_ID, primary, alt))
    return pd.DataFrame(
        rows, columns=["item_id", "dept_id", "primary_supplier_id", "alt_supplier_id"]
    )


def gen_inventory(sales: pd.DataFrame) -> pd.DataFrame:
    """库存快照：把现有库存设成“最近 28 天日均需求的若干天”，有多有少。

    有的商品库存充足（算出来不用补），有的紧张（需要补），覆盖不同情形。
    """
    rng = np.random.default_rng(config.SEED + 101)
    recent_start = config.DATA_TODAY - pd.Timedelta(days=28)
    recent = sales[sales["d"] >= recent_start]
    daily_mean = (
        recent.groupby("item_id")["units"].sum() / 28.0
    )  # 每个商品最近 28 天的日均销量

    rows = []
    for item in config.item_ids():
        mu = float(daily_mean.get(item, 0.0))
        # 覆盖天数在 2~12 天之间随机：天数少的会触发补货，多的不会。
        cover_days = rng.uniform(2.0, 12.0)
        on_hand = int(round(mu * cover_days))
        rows.append((config.STORE_ID, item, on_hand, config.DATA_TODAY))
    return pd.DataFrame(
        rows, columns=["store_id", "item_id", "on_hand", "as_of"]
    )
