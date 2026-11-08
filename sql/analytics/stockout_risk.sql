-- 断货风险：用窗口函数算每个商品最近 14 天的日均销量，再和现有库存比，得到“可供天数”。
-- 可供天数越小越危险。这是第2节“用 SQL 分析断货风险”的范例，学员课后再写更多条。
WITH recent AS (
    SELECT
        s.item_id,
        s.d,
        s.units,
        -- 窗口函数：对每个商品、按日期排序，取当前行往前 13 天共 14 天的滑动日均。
        AVG(s.units) OVER (
            PARTITION BY s.item_id
            ORDER BY s.d
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS avg_daily_14d
    FROM sales s
),
latest AS (
    -- 每个商品只取最新一天的那条（它的 avg_daily_14d 就是“当前”日均需求）。
    SELECT DISTINCT ON (item_id) item_id, d, avg_daily_14d
    FROM recent
    ORDER BY item_id, d DESC
)
SELECT
    i.item_id,
    i.on_hand,
    ROUND(l.avg_daily_14d, 2)                                   AS avg_daily_14d,
    ROUND((i.on_hand / NULLIF(l.avg_daily_14d, 0))::numeric, 1) AS days_of_cover
FROM inventory i
JOIN latest l USING (item_id)
ORDER BY days_of_cover ASC NULLS LAST;
