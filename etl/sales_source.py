"""销量数据来源的统一入口：有真实 M5 就用它，没有就回退合成。

这样 `make data` / `make demo` 默认吃真实 M5，而测试和没下 M5 的机器仍能用合成数据跑通。
"""
from __future__ import annotations

import pandas as pd

from etl.gen_sales import gen_sales
from etl.load_m5 import load_m5_subset, m5_available


def load_sales() -> pd.DataFrame:
    if m5_available():
        return load_m5_subset()
    return gen_sales()


def sales_source_name() -> str:
    return "真实 M5（data/raw/）" if m5_available() else "合成数据（etl.gen_sales）"
