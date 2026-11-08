"""合成异常事件表。

把剧本里的停产、港口拥堵、天气、促销、召回统一展开成事件行。
scope_type + scope_id 指明每个事件影响谁，补货工具据此判断要不要调整决策。
"""
from __future__ import annotations

import pandas as pd

from etl.plan import build_scenario


def gen_events() -> pd.DataFrame:
    sc = build_scenario()
    rows = []
    seq = 0

    def add(event_type, scope_type, scope_id, start, end, severity, note):
        nonlocal seq
        seq += 1
        rows.append(
            (f"EV_{seq:04d}", event_type, scope_type, scope_id, start, end, severity, note)
        )

    for o in sc.outages:
        add("supplier_outage", "supplier", o.supplier_id, o.start, o.end, o.severity,
            f"{o.supplier_id} 停产，期间无法供货")
    for ld in sc.lane_disruptions:
        add(ld.event_type, "lane", ld.lane, ld.start, ld.end, ld.severity,
            f"{ld.lane} {ld.event_type}，在途预计 +{ld.extra_delay_days} 天")
    for p in sc.promos:
        add("promo", "sku", p.item_id, p.start, p.end, 3,
            f"{p.item_id} 促销：价格 ×{p.price_factor}，需求约 ×{p.uplift}")
    for r in sc.recalls:
        add("recall", "sku", r.item_id, r.start, r.end, r.severity,
            f"{r.item_id} 召回")

    return pd.DataFrame(
        rows,
        columns=[
            "event_id", "event_type", "scope_type", "scope_id",
            "start_date", "end_date", "severity", "note",
        ],
    )
