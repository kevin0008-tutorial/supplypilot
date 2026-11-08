"""合成数据的全局配置。

一处改、处处一致：所有生成器都从这里读取同一个随机种子和参数，
保证老师和学员两台电脑 `make data` 出来的表完全一样（可复现）。
"""
from __future__ import annotations

import datetime as dt

# 固定随机种子：同一个种子 + 同一份代码 = 同一批数据。
SEED = 20260201

# “今天”。合成数据里一切和时间相关的东西（库存快照时点、在途单的预计到仓、
# 事件的起止日期）都以这个固定日期为基准，避免每次运行结果都变。
DATA_TODAY = dt.date(2016, 4, 24)

# 只取一家门店、FOODS 品类的一个子集，和 M5 的结构对齐但规模更小，笔记本秒级可跑。
STORE_ID = "CA_1"
DEPT_ID = "FOODS_3"
N_ITEMS = 20                     # 商品数（item_id 形如 FOODS_3_001 ...）
HISTORY_DAYS = 720               # 历史天数（约两年，足够看出周季节性）

# 补货决策用的检查周期 R（review period）：每隔多少天盘一次货、下一次单。
REVIEW_PERIOD_DAYS = 7

# 承运商和航线（物流/在途数据用）。
CARRIERS = ["CarrierX", "CarrierY", "CarrierZ"]
LANES = ["Shanghai-LA", "Shenzhen-LA", "Ningbo-Oakland"]


def item_ids() -> list[str]:
    """全部商品编号，形如 FOODS_3_001 ... FOODS_3_020。"""
    return [f"{DEPT_ID}_{i:03d}" for i in range(1, N_ITEMS + 1)]
