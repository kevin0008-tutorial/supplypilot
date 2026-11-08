"""一键生成全部合成数据，写出 CSV，并（若数据库可连）灌进 Postgres。

对应 `make data`。没启动数据库也不会报错：CSV 照样生成，只是跳过入库。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etl import config
from etl.gen_business import gen_inventory, gen_item_master, gen_suppliers
from etl.gen_events import gen_events
from etl.gen_logistics import gen_purchase_orders
from etl.gen_sales import gen_sales
from etl.load import load_all

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def generate_frames() -> dict[str, pd.DataFrame]:
    sales = gen_sales()
    return {
        "sales": sales,
        "suppliers": gen_suppliers(),
        "item_master": gen_item_master(),
        "inventory": gen_inventory(sales),
        "purchase_orders": gen_purchase_orders(),
        "events": gen_events(),
    }


def write_csvs(frames: dict[str, pd.DataFrame]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    frames = generate_frames()
    write_csvs(frames)
    print(f"已生成 CSV 到 {OUT_DIR}（{config.N_ITEMS} 个商品，{config.HISTORY_DAYS} 天）")
    try:
        load_all(frames)
        print("已灌入 Postgres。")
    except Exception as exc:  # noqa: BLE001 —— 数据库没起时不该让整条命令失败
        print(f"跳过入库（数据库连不上？先 make up）：{exc}")


if __name__ == "__main__":
    main()
