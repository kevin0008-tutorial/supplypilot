"""把生成好的 DataFrame 灌进 Postgres（以及从 Postgres 读回）。

连接信息从环境变量拼：优先用 DATABASE_URL，否则用 .env 里的 POSTGRES_* 拼出来。
从宿主机（uv run）连的是 localhost:POSTGRES_PORT；容器内部才用主机名 db。
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import psycopg

_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA = _ROOT / "db" / "schema.sql"

# 灌数据的顺序：被外键引用的表（suppliers）要先于引用它的表（item_master）。
TABLES = ["suppliers", "item_master", "sales", "inventory", "purchase_orders", "events"]


def database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    user = os.getenv("POSTGRES_USER", "supplypilot")
    pwd = os.getenv("POSTGRES_PASSWORD", "supplypilot")
    db = os.getenv("POSTGRES_DB", "supplypilot")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(_SCHEMA.read_text())
    conn.commit()


def write_table(conn: psycopg.Connection, df: pd.DataFrame, table: str) -> None:
    """清空并用 COPY 批量写入一张表。"""
    cols = list(df.columns)
    col_list = ", ".join(cols)
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {table} CASCADE")
        with cur.copy(f"COPY {table} ({col_list}) FROM STDIN") as copy:
            for row in df.itertuples(index=False, name=None):
                copy.write_row(tuple(_clean(v) for v in row))
    conn.commit()


def _clean(value):
    # pandas 的缺失值（NaT/NaN）要变成 None，否则 COPY 会报类型错。
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value


def load_all(frames: dict[str, pd.DataFrame]) -> None:
    with psycopg.connect(database_url()) as conn:
        apply_schema(conn)
        for table in TABLES:
            write_table(conn, frames[table], table)


def read_table(conn: psycopg.Connection, table: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table}")
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)
