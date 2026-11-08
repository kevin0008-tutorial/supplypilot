"""离线端到端 demo：不需要数据库，纯 Python 跑通"生成数据 → 组装 → 补货决策"。

`uv run python -m tools.demo` 就能看到几个商品的补货建议，包括事件驱动的调整说明。
测试也复用这里的 build_demo_repository()。
"""
from __future__ import annotations

from etl import config
from etl.gen_business import gen_inventory, gen_item_master, gen_suppliers
from etl.gen_events import gen_events
from etl.gen_logistics import gen_purchase_orders
from etl.gen_sales import gen_sales
from tools.repository import InMemoryRepository, forecast_and_reorder


def build_demo_repository() -> InMemoryRepository:
    sales = gen_sales()
    return InMemoryRepository(
        sales=sales,
        inventory=gen_inventory(sales),
        suppliers=gen_suppliers(),
        item_master=gen_item_master(),
        purchase_orders=gen_purchase_orders(),
        events=gen_events(),
        as_of=config.DATA_TODAY,
        review_period_days=config.REVIEW_PERIOD_DAYS,
    )


def main() -> None:
    repo = build_demo_repository()
    results = forecast_and_reorder(config.item_ids()[:10], repo)
    for r in results:
        print(f"\n{r.item_id}  供应商={r.supplier_used}  交期L={r.lead_time_used}  "
              f"MOQ={r.moq}")
        print(f"  未来{r.lead_time_used + r.review_period_days}天需求0.95分位={r.demand_quantile}  "
              f"现有库存={r.on_hand}  可及时到货在途={r.on_order_effective}")
        print(f"  → 建议补货 {r.reorder_qty}（原始 {r.raw_reorder_qty}）  flags={r.flags}")
        for a in r.adjustments:
            print(f"    · {a}")


if __name__ == "__main__":
    main()
