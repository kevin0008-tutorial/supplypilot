import datetime as dt

from tools.replenishment import (
    Event,
    InTransitPO,
    ReplenishmentInput,
    Supplier,
    compute_reorder,
)

AS_OF = dt.date(2016, 4, 24)
HIST = [10.0] * 60  # 稳定日销量 10，便于手算


def _base_input(**overrides) -> ReplenishmentInput:
    defaults = dict(
        item_id="X",
        as_of=AS_OF,
        history_units=HIST,
        on_hand=0,
        primary_supplier=Supplier(supplier_id="SUP_A", lead_time_days=7, moq=50, lane="LANE_X"),
        alt_supplier=Supplier(supplier_id="SUP_B", lead_time_days=10, moq=100, lane="LANE_Y"),
        in_transit=[],
        active_events=[],
        review_period_days=7,
    )
    defaults.update(overrides)
    return ReplenishmentInput(**defaults)


def test_basic_reorder_rounds_up_to_moq():
    res = compute_reorder(_base_input())
    # window=14，常量序列 → D=140；raw=140-0-0=140；向上取整到 50 的倍数 = 150。
    assert res.demand_quantile == 140.0
    assert res.lead_time_used == 7
    assert res.reorder_qty == 150
    assert res.reorder_qty % 50 == 0
    assert "reorder_needed" in res.flags


def test_sufficient_inventory_no_reorder():
    res = compute_reorder(_base_input(on_hand=10_000))
    assert res.reorder_qty == 0
    assert "reorder_needed" not in res.flags


def test_supplier_outage_switches_to_alt():
    outage = Event(
        event_type="supplier_outage", scope_type="supplier", scope_id="SUP_A",
        start_date=AS_OF - dt.timedelta(days=1), end_date=AS_OF + dt.timedelta(days=10),
        severity=5,
    )
    res = compute_reorder(_base_input(active_events=[outage]))
    assert res.supplier_used == "SUP_B"      # 切到备选
    assert res.lead_time_used == 10          # 用备选的交期
    assert "supplier_switched" in res.flags
    # window=17 → D=170 → 取整到 100 的倍数 = 200。
    assert res.reorder_qty == 200


def test_recall_holds_reorder():
    recall = Event(
        event_type="recall", scope_type="sku", scope_id="X",
        start_date=AS_OF - dt.timedelta(days=1), end_date=AS_OF + dt.timedelta(days=10),
        severity=4,
    )
    res = compute_reorder(_base_input(active_events=[recall]))
    assert res.reorder_qty == 0
    assert "recall_hold" in res.flags


def test_late_po_excluded_from_available():
    in_transit = [
        InTransitPO(po_id="PO1", qty=100, planned_eta=AS_OF + dt.timedelta(days=3),
                    delay_days=0, lane="LANE_X"),            # 窗口内到货 → 计入
        InTransitPO(po_id="PO2", qty=500, planned_eta=AS_OF + dt.timedelta(days=10),
                    delay_days=10, lane="LANE_Y"),           # 实际 +20 天，超出窗口 → 剔除
    ]
    res = compute_reorder(_base_input(in_transit=in_transit))
    assert res.on_order_effective == 100
    assert any("晚于补货窗口" in a for a in res.adjustments)
    # raw = 140 - 0 - 100 = 40 → 取整到 50。
    assert res.reorder_qty == 50
