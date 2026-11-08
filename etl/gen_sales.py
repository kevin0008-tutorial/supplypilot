"""合成销量表（和 M5 的长表结构一致，但规模小、可复现）。

每个商品一条泊松需求流：基础需求率 × 周季节性（周末更高），
促销期把需求率放大、价格打折。部分商品需求率很低 → 大量零销量（间歇性需求）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etl import config
from etl.plan import build_scenario

# 一周七天的需求乘子：食品类周末卖得多。索引 0=周一 ... 6=周日。
_WEEKDAY_MULT = np.array([0.9, 0.9, 0.95, 1.0, 1.1, 1.35, 1.25])


def gen_sales() -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED)
    items = config.item_ids()
    dates = pd.date_range(
        end=pd.Timestamp(config.DATA_TODAY), periods=config.HISTORY_DAYS, freq="D"
    )
    weekday_mult = _WEEKDAY_MULT[dates.day_of_week.to_numpy()]

    # 把促销剧本整理成 {item_id: [(start, end, uplift, price_factor), ...]}
    promos: dict[str, list] = {}
    for p in build_scenario().promos:
        promos.setdefault(p.item_id, []).append(
            (pd.Timestamp(p.start), pd.Timestamp(p.end), p.uplift, p.price_factor)
        )

    frames = []
    for item in items:
        base_rate = rng.uniform(0.2, 6.0)      # 每个商品自己的基础日需求率 λ
        base_price = round(rng.uniform(1.5, 4.0), 2)

        lam = base_rate * weekday_mult         # 逐日需求率
        price = np.full(len(dates), base_price)

        for start, end, uplift, price_factor in promos.get(item, []):
            mask = (dates >= start) & (dates <= end)
            lam = np.where(mask, lam * uplift, lam)
            price = np.where(mask, np.round(base_price * price_factor, 2), price)

        units = rng.poisson(lam)               # 泊松采样：天然产生 0 和整数销量

        frames.append(
            pd.DataFrame(
                {
                    "store_id": config.STORE_ID,
                    "item_id": item,
                    "dept_id": config.DEPT_ID,
                    "d": dates.date,
                    "units": units.astype(int),
                    "sell_price": price,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)
