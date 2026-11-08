"""数据访问层：把分散在各张表里的数据，组装成补货工具要的 ReplenishmentInput。

为什么单独一层？因为补货逻辑（compute_reorder）是纯函数、不该知道数据从哪来。
- `InMemoryRepository`：直接吃 DataFrame，测试和离线 demo 用，不需要数据库。
- `PostgresRepository`：从 Postgres 查，线上/课堂真实数据用。
两者接口一样，compute_reorder 对它们一视同仁——这就是"依赖倒置"。
"""
from __future__ import annotations

import datetime as dt
from typing import Protocol

import pandas as pd

from etl.gen_logistics import _SUPPLIER_LANE
from tools.replenishment import (
    Event,
    InTransitPO,
    ReplenishmentInput,
    ReplenishmentResult,
    Supplier,
    compute_reorder,
)


class Repository(Protocol):
    def load_input(self, item_id: str) -> ReplenishmentInput: ...


class InMemoryRepository:
    def __init__(
        self,
        sales: pd.DataFrame,
        inventory: pd.DataFrame,
        suppliers: pd.DataFrame,
        item_master: pd.DataFrame,
        purchase_orders: pd.DataFrame,
        events: pd.DataFrame,
        as_of: dt.date,
        review_period_days: int = 7,
    ):
        self.sales = sales
        self.inventory = inventory
        self.suppliers = suppliers.set_index("supplier_id")
        self.item_master = item_master.set_index("item_id")
        self.pos = purchase_orders
        self.events = events
        self.as_of = as_of
        self.review_period_days = review_period_days

    def _supplier(self, supplier_id: str) -> Supplier:
        row = self.suppliers.loc[supplier_id]
        return Supplier(
            supplier_id=supplier_id,
            lead_time_days=int(row["lead_time_days"]),
            moq=int(row["moq"]),
            lane=_SUPPLIER_LANE.get(supplier_id),
        )

    def load_input(self, item_id: str) -> ReplenishmentInput:
        hist = self.sales[self.sales["item_id"] == item_id].sort_values("d")
        on_hand = int(self.inventory.loc[self.inventory["item_id"] == item_id, "on_hand"].iloc[0])

        master = self.item_master.loc[item_id]
        primary = self._supplier(master["primary_supplier_id"])
        alt = self._supplier(master["alt_supplier_id"])

        in_transit = [
            InTransitPO(
                po_id=po.po_id, qty=int(po.qty),
                planned_eta=_as_date(po.planned_eta),
                delay_days=int(po.delay_days), lane=po.lane,
            )
            for po in self.pos.itertuples()
            if po.item_id == item_id and po.status == "in_transit"
        ]

        lanes = {primary.lane, alt.lane}
        suppliers = {primary.supplier_id, alt.supplier_id}
        active = []
        for e in self.events.itertuples():
            touches = (
                (e.scope_type == "sku" and e.scope_id == item_id)
                or (e.scope_type == "supplier" and e.scope_id in suppliers)
                or (e.scope_type == "lane" and e.scope_id in lanes)
            )
            if touches:
                active.append(
                    Event(
                        event_type=e.event_type, scope_type=e.scope_type,
                        scope_id=e.scope_id, start_date=_as_date(e.start_date),
                        end_date=_as_date(e.end_date), severity=int(e.severity),
                        note=e.note,
                    )
                )

        return ReplenishmentInput(
            item_id=item_id,
            as_of=self.as_of,
            history_units=hist["units"].astype(float).tolist(),
            on_hand=on_hand,
            primary_supplier=primary,
            alt_supplier=alt,
            in_transit=in_transit,
            active_events=active,
            review_period_days=self.review_period_days,
        )


def _as_date(value) -> dt.date:
    """把 date / pandas.Timestamp / 字符串统一成 datetime.date。"""
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    return pd.Timestamp(value).date()


def forecast_and_reorder(
    item_ids: list[str], repo: Repository
) -> list[ReplenishmentResult]:
    """对一批商品算补货建议。Agent（第7节）直接调用的就是这个函数。"""
    return [compute_reorder(repo.load_input(item_id)) for item_id in item_ids]
