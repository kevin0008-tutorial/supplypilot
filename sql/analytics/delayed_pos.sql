-- 在途延误：找出还在途、且会晚到的采购单（实际预计到仓 = 计划到仓 + 延误天数）。
-- 这些延误正是补货工具要感知的异常事件信号之一。
SELECT
    po.po_id,
    po.item_id,
    po.supplier_id,
    po.lane,
    po.qty,
    po.planned_eta,
    po.delay_days,
    (po.planned_eta + (po.delay_days || ' days')::interval)::date AS effective_eta
FROM purchase_orders po
WHERE po.status = 'in_transit'
  AND po.delay_days > 0
ORDER BY po.delay_days DESC, effective_eta;
