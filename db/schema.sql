-- SupplyPilot 第2节：数据层建表脚本。
-- 用 CREATE TABLE IF NOT EXISTS，可重复执行；加载脚本会先 TRUNCATE 再灌数据。
-- 约定：所有金额用 NUMERIC（精确小数，不用 float，避免钱算错）；日期用 DATE。

-- 1) 销量：事实表，每行是“某门店某商品某天卖了多少”。M5 的长表形态。
CREATE TABLE IF NOT EXISTS sales (
    store_id   TEXT    NOT NULL,
    item_id    TEXT    NOT NULL,
    dept_id    TEXT    NOT NULL,
    d          DATE    NOT NULL,          -- 日期（M5 里叫 d_1、d_2…，这里直接存真实日期）
    units      INTEGER NOT NULL,          -- 当天销量，可能是 0（间歇性需求）
    sell_price NUMERIC(10, 2),            -- 当天售价，促销时会降
    PRIMARY KEY (store_id, item_id, d)
);
-- 联合索引：几乎所有查询都是“按门店+商品，取一段日期”，这个顺序能走索引。
CREATE INDEX IF NOT EXISTS idx_sales_item_date ON sales (store_id, item_id, d);

-- 2) 供应商主数据：交期、最小起订量、单价。补货决策的关键参数。
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id    TEXT    PRIMARY KEY,
    name           TEXT    NOT NULL,
    lead_time_days INTEGER NOT NULL,      -- 正常交期 L（下单到到仓的天数）
    moq            INTEGER NOT NULL,      -- 最小起订量（起订/取整的步长）
    unit_cost      NUMERIC(10, 2) NOT NULL
);

-- 3) 商品主数据：每个商品挂一个主供应商和一个备选供应商（供应商停产时切换）。
CREATE TABLE IF NOT EXISTS item_master (
    item_id             TEXT PRIMARY KEY,
    dept_id             TEXT NOT NULL,
    primary_supplier_id TEXT NOT NULL REFERENCES suppliers (supplier_id),
    alt_supplier_id     TEXT NOT NULL REFERENCES suppliers (supplier_id)
);

-- 4) 库存快照：某个时点（DATA_TODAY）每个商品的现有库存。
CREATE TABLE IF NOT EXISTS inventory (
    store_id TEXT    NOT NULL,
    item_id  TEXT    NOT NULL,
    on_hand  INTEGER NOT NULL,            -- 现有库存
    as_of    DATE    NOT NULL,            -- 快照时点
    PRIMARY KEY (store_id, item_id)
);

-- 5) 采购单 / 在途：已经下了单、还没到仓的货。补货时算“在途库存”要用它。
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id       TEXT    PRIMARY KEY,
    item_id     TEXT    NOT NULL,
    supplier_id TEXT    NOT NULL,
    carrier     TEXT,
    lane        TEXT,                     -- 航线/港口，用来和物流异常事件对齐
    qty         INTEGER NOT NULL,
    order_date  DATE    NOT NULL,
    planned_eta DATE    NOT NULL,         -- 计划到仓日期
    actual_eta  DATE,                     -- 实际到仓（还在途就为 NULL）
    delay_days  INTEGER NOT NULL DEFAULT 0, -- 当前已知的延误天数（>0 表示会晚到）
    status      TEXT    NOT NULL          -- in_transit / delivered / ordered
);
CREATE INDEX IF NOT EXISTS idx_po_item ON purchase_orders (item_id);

-- 6) 异常事件：供应商停产、港口拥堵、天气、促销、召回……
--    scope_type + scope_id 指明事件影响谁（某供应商 / 某 SKU / 某航线 / 某门店）。
CREATE TABLE IF NOT EXISTS events (
    event_id   TEXT    PRIMARY KEY,
    event_type TEXT    NOT NULL,          -- supplier_outage / port_congestion / weather / promo / recall
    scope_type TEXT    NOT NULL,          -- supplier / sku / lane / store
    scope_id   TEXT    NOT NULL,          -- 对应的 id，比如 SUP_B、FOODS_3_003、Shanghai-LA
    start_date DATE    NOT NULL,
    end_date   DATE    NOT NULL,
    severity   INTEGER NOT NULL,          -- 1–5，越大影响越大
    note       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_scope ON events (scope_type, scope_id);
