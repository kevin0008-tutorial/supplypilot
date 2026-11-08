from etl import config
from etl.gen_business import gen_item_master, gen_suppliers
from etl.gen_events import gen_events
from etl.gen_logistics import gen_purchase_orders
from etl.gen_sales import gen_sales
from tools.demo import build_demo_repository
from tools.repository import forecast_and_reorder


def test_sales_is_deterministic_and_nonnegative():
    a, b = gen_sales(), gen_sales()
    assert a.equals(b)                      # 同一种子 → 完全一样（可复现）
    assert (a["units"] >= 0).all()
    assert a["item_id"].nunique() == config.N_ITEMS


def test_item_master_primary_differs_from_alt():
    im = gen_item_master()
    assert (im["primary_supplier_id"] != im["alt_supplier_id"]).all()


def test_purchase_orders_reference_valid_suppliers():
    pos = gen_purchase_orders()
    supplier_ids = set(gen_suppliers()["supplier_id"])
    assert set(pos["supplier_id"]).issubset(supplier_ids)
    assert pos["status"].isin({"in_transit", "delivered", "ordered"}).all()
    assert (pos["delay_days"] >= 0).all()


def test_events_include_supplier_outage():
    ev = gen_events()
    outages = ev[ev["event_type"] == "supplier_outage"]
    assert "SUP_B" in set(outages["scope_id"])


def test_end_to_end_demo_reorder():
    # 显式用合成销量：测试要确定、要快，不依赖本机是否下载了 M5。
    repo = build_demo_repository(sales=gen_sales())
    items = config.item_ids()[:10]
    results = forecast_and_reorder(items, repo)

    assert len(results) == 10
    assert all(isinstance(r.reorder_qty, int) and r.reorder_qty >= 0 for r in results)
    # SUP_B 停产 → 它供货的商品应触发换供应商。
    assert any("supplier_switched" in r.flags for r in results)
    # 被召回的商品（第 10 个）应暂停补货。
    recalled = next(r for r in results if r.item_id == config.item_ids()[9])
    assert "recall_hold" in recalled.flags
    assert recalled.reorder_qty == 0
