"""合成物流/在途表（purchase_orders）。

每个商品有 0~2 张“已下单、还没到仓”的采购单。补货时这些就是“在途库存”，
要从需求里扣掉。落在拥堵航线上的单会带延误天数——这正是异常事件的信号来源之一。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from etl import config
from etl.gen_business import SUPPLIERS, gen_item_master
from etl.plan import build_scenario

# 供应商 → 主航线（用来和港口拥堵/天气事件对齐）。
_SUPPLIER_LANE = {
    "SUP_A": "Shanghai-LA",
    "SUP_B": "Shenzhen-LA",
    "SUP_C": "Ningbo-Oakland",
    "SUP_D": "Shanghai-LA",
}


def gen_purchase_orders() -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED + 202)
    lead = {s[0]: s[2] for s in SUPPLIERS}   # supplier_id -> lead_time_days
    moq = {s[0]: s[3] for s in SUPPLIERS}     # supplier_id -> moq
    primary = dict(
        zip(gen_item_master()["item_id"], gen_item_master()["primary_supplier_id"])
    )

    # 当前生效的航线延误：{lane: extra_delay_days}
    active_lane_delay: dict[str, int] = {}
    for ld in build_scenario().lane_disruptions:
        if ld.start <= config.DATA_TODAY <= ld.end:
            active_lane_delay[ld.lane] = max(
                active_lane_delay.get(ld.lane, 0), ld.extra_delay_days
            )

    rows = []
    po_seq = 0
    for item in config.item_ids():
        supplier = primary[item]
        lane = _SUPPLIER_LANE[supplier]
        n_open = int(rng.choice([0, 1, 1, 2]))  # 偏向 1 张在途单
        for _ in range(n_open):
            po_seq += 1
            l = lead[supplier]
            # 下单日在“今天”往前数 1~l 天，使单子正好处在途中。
            order_date = config.DATA_TODAY - dt.timedelta(days=int(rng.integers(1, l + 1)))
            planned_eta = order_date + dt.timedelta(days=l)
            delay = active_lane_delay.get(lane, 0)
            effective_eta = planned_eta + dt.timedelta(days=delay)
            qty = int(moq[supplier] * rng.integers(1, 4))

            if effective_eta >= config.DATA_TODAY:
                status, actual_eta = "in_transit", None
            else:
                status, actual_eta = "delivered", effective_eta

            rows.append(
                (
                    f"PO_{po_seq:04d}", item, supplier,
                    rng.choice(config.CARRIERS), lane, qty,
                    order_date, planned_eta, actual_eta, delay, status,
                )
            )

    return pd.DataFrame(
        rows,
        columns=[
            "po_id", "item_id", "supplier_id", "carrier", "lane", "qty",
            "order_date", "planned_eta", "actual_eta", "delay_days", "status",
        ],
    )
