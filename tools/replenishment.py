"""M2 预测补货工具：forecast_and_reorder。

这是第2节的核心产出，也是第7节 Agent 会直接调用的工具。设计要点：

- **纯函数 + 结构化输入输出**：`compute_reorder` 不碰数据库、不发网络，只接收一个
  `ReplenishmentInput`、算出一个 `ReplenishmentResult`。好测、可复现，Agent 也能放心调。
- **每个数字可追溯**：返回里带 `provenance`，说明补货量是用哪些输入算出来的。
- **事件驱动调整**：供应商停产→换备选供应商；航线拥堵/在途延误→拉长交期、剔除会晚到的在途货；
  召回→暂停补货；促销→需求已按近期水平自动抬高。

补货公式（不假设正态分布，用分位数）：
    Q_raw = max(0, D^0.95_{L+R} - on_hand - on_order_在途可及时到货的部分)
    Q     = 向上取整到 MOQ 的整数倍
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, Field

from forecast.baseline import window_demand_quantile


class Supplier(BaseModel):
    supplier_id: str
    lead_time_days: int
    moq: int
    lane: str | None = None


class InTransitPO(BaseModel):
    po_id: str
    qty: int
    planned_eta: dt.date
    delay_days: int = 0
    lane: str | None = None


class Event(BaseModel):
    event_type: str      # supplier_outage / port_congestion / weather / promo / recall
    scope_type: str      # supplier / sku / lane / store
    scope_id: str
    start_date: dt.date
    end_date: dt.date
    severity: int = 1
    note: str | None = None

    def active_on(self, day: dt.date) -> bool:
        return self.start_date <= day <= self.end_date


class ReplenishmentInput(BaseModel):
    item_id: str
    as_of: dt.date
    history_units: list[float]          # 按日期升序的历史日销量
    on_hand: int
    primary_supplier: Supplier
    alt_supplier: Supplier | None = None
    in_transit: list[InTransitPO] = Field(default_factory=list)
    active_events: list[Event] = Field(default_factory=list)
    review_period_days: int = 7
    service_level: float = 0.95


class ReplenishmentResult(BaseModel):
    item_id: str
    reorder_qty: int                    # 最终建议补货量（已按 MOQ 取整）
    raw_reorder_qty: float              # 取整前的原始值
    demand_quantile: float              # D^q_{L+R}
    lead_time_used: int                 # 实际采用的交期 L（可能被事件拉长/换供应商改变）
    review_period_days: int
    supplier_used: str
    moq: int
    on_hand: int
    on_order_effective: int             # 能在窗口内到货的在途量
    adjustments: list[str] = Field(default_factory=list)  # 事件驱动的调整说明
    flags: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)


Forecaster = Callable[[Sequence[float], int, float], float]


def _round_up_to_moq(qty: float, moq: int) -> int:
    if qty <= 0:
        return 0
    if moq <= 1:
        return int(np.ceil(qty))
    return int(np.ceil(qty / moq) * moq)


def compute_reorder(
    inp: ReplenishmentInput,
    demand_quantile_fn: Forecaster = window_demand_quantile,
) -> ReplenishmentResult:
    adjustments: list[str] = []
    flags: list[str] = []

    def events(kind: str, scope_type: str, scope_id: str | None = None) -> list[Event]:
        return [
            e for e in inp.active_events
            if e.event_type == kind and e.scope_type == scope_type
            and e.active_on(inp.as_of)
            and (scope_id is None or e.scope_id == scope_id)
        ]

    # 1) 供应商停产 → 换备选供应商。
    supplier = inp.primary_supplier
    if events("supplier_outage", "supplier", supplier.supplier_id):
        if inp.alt_supplier is not None:
            adjustments.append(
                f"主供应商 {supplier.supplier_id} 停产，改用备选 {inp.alt_supplier.supplier_id}"
                f"（交期 {supplier.lead_time_days}→{inp.alt_supplier.lead_time_days} 天，"
                f"MOQ {supplier.moq}→{inp.alt_supplier.moq}）"
            )
            supplier = inp.alt_supplier
            flags.append("supplier_switched")
        else:
            flags.append("no_supplier")
            adjustments.append(f"主供应商 {supplier.supplier_id} 停产且无备选，需人工处理")

    # 2) 召回 → 暂停补货。
    if events("recall", "sku", inp.item_id):
        flags.append("recall_hold")
        adjustments.append("该商品处于召回事件中，暂停补货")

    # 3) 交期调整：用在途单观察到的延误天数（航线拥堵/天气造成的）拉长新单的交期。
    lane = supplier.lane
    observed_delay = max(
        [po.delay_days for po in inp.in_transit if lane is None or po.lane == lane],
        default=0,
    )
    lead_time = supplier.lead_time_days
    if observed_delay > 0:
        lead_time += observed_delay
        lane_events = [e for e in inp.active_events
                       if e.scope_type == "lane" and e.scope_id == lane
                       and e.active_on(inp.as_of)]
        reason = f"（{lane_events[0].event_type}）" if lane_events else ""
        adjustments.append(
            f"航线 {lane} 当前延误 {observed_delay} 天{reason}，"
            f"新单交期按 {supplier.lead_time_days}→{lead_time} 天估算"
        )

    window = lead_time + inp.review_period_days

    # 4) 在途库存：只算“能在窗口内到货”的；会晚到窗口之后的，不算进可用量。
    horizon_end = inp.as_of + dt.timedelta(days=window)
    on_order_effective = 0
    late_qty = 0
    for po in inp.in_transit:
        eta = po.planned_eta + dt.timedelta(days=po.delay_days)
        if eta <= horizon_end:
            on_order_effective += po.qty
        else:
            late_qty += po.qty
    if late_qty > 0:
        adjustments.append(
            f"有 {late_qty} 件在途货会晚于补货窗口（{window} 天）到仓，未计入可用量"
        )

    # 5) 促销说明（需求的抬升已经体现在近期水平里，这里只做解释）。
    if events("promo", "sku", inp.item_id):
        adjustments.append("该商品正在促销，需求分位数已按近期较高水平计算")

    # 6) 需求分位数 + 补货量。
    demand_q = float(demand_quantile_fn(inp.history_units, window, inp.service_level))
    if "recall_hold" in flags:
        raw = 0.0
    else:
        raw = max(0.0, demand_q - inp.on_hand - on_order_effective)
    reorder = _round_up_to_moq(raw, supplier.moq)
    if reorder > 0:
        flags.append("reorder_needed")

    return ReplenishmentResult(
        item_id=inp.item_id,
        reorder_qty=reorder,
        raw_reorder_qty=round(raw, 2),
        demand_quantile=round(demand_q, 2),
        lead_time_used=lead_time,
        review_period_days=inp.review_period_days,
        supplier_used=supplier.supplier_id,
        moq=supplier.moq,
        on_hand=inp.on_hand,
        on_order_effective=on_order_effective,
        adjustments=adjustments,
        flags=flags,
        provenance={
            "demand_quantile": f"forecast.baseline.window_demand_quantile(history, window={window}, q={inp.service_level})",
            "window": f"lead_time_used({lead_time}) + review_period({inp.review_period_days})",
            "reorder_qty": "max(0, demand_quantile - on_hand - on_order_effective) 再向上取整到 MOQ",
        },
    )
