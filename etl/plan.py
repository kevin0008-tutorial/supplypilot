"""合成数据的“剧本”。

销量、事件、在途三张表必须互相对齐：某个促销事件要在销量里真的造成尖峰，
某个供应商停产事件要正好落在某些商品的主供应商上，港口拥堵要正好压在某条航线的在途单上。
把这些“剧情”集中写在这里，三个生成器都读它，数据才会自洽。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from etl import config


@dataclass(frozen=True)
class Promo:
    item_id: str
    start: dt.date
    end: dt.date
    uplift: float        # 需求放大倍数（促销期间销量 ×uplift）
    price_factor: float  # 价格打几折（0.7 = 七折）


@dataclass(frozen=True)
class Outage:
    supplier_id: str
    start: dt.date
    end: dt.date
    severity: int


@dataclass(frozen=True)
class LaneDisruption:
    event_type: str      # port_congestion / weather
    lane: str
    start: dt.date
    end: dt.date
    extra_delay_days: int
    severity: int


@dataclass(frozen=True)
class Recall:
    item_id: str
    start: dt.date
    end: dt.date
    severity: int


@dataclass(frozen=True)
class Scenario:
    promos: list[Promo] = field(default_factory=list)
    outages: list[Outage] = field(default_factory=list)
    lane_disruptions: list[LaneDisruption] = field(default_factory=list)
    recalls: list[Recall] = field(default_factory=list)


def _d(days_from_today: int) -> dt.date:
    return config.DATA_TODAY + dt.timedelta(days=days_from_today)


def build_scenario() -> Scenario:
    """固定的剧本（不依赖随机数，纯确定性）。"""
    items = config.item_ids()

    promos = [
        # 历史里的两次促销尖峰：供第2节演示“异常检测”能不能找出来。
        Promo(items[2], _d(-300), _d(-294), uplift=5.0, price_factor=0.6),
        Promo(items[7], _d(-120), _d(-114), uplift=4.0, price_factor=0.7),
        Promo(items[13], _d(-60), _d(-56), uplift=4.0, price_factor=0.7),
        # 一个“正在进行”的促销：会抬高当前需求，影响本次补货决策。
        Promo(items[5], _d(-1), _d(5), uplift=3.0, price_factor=0.75),
    ]

    outages = [
        # SUP_B 从 3 天前开始停产，持续到 20 天后：现在正生效。
        Outage("SUP_B", _d(-3), _d(20), severity=5),
    ]

    lane_disruptions = [
        # 上海-洛杉矶航线港口拥堵，给这条线的在途单增加 7 天延误。
        LaneDisruption("port_congestion", "Shanghai-LA", _d(-5), _d(10),
                       extra_delay_days=7, severity=4),
        # 历史上一次天气事件（不影响当前决策，只丰富事件表）。
        LaneDisruption("weather", "Ningbo-Oakland", _d(-200), _d(-196),
                       extra_delay_days=4, severity=3),
    ]

    recalls = [
        # 正在进行的召回：演示"召回→暂停补货"。
        Recall(items[9], _d(-2), _d(12), severity=4),
    ]

    return Scenario(promos=promos, outages=outages,
                    lane_disruptions=lane_disruptions, recalls=recalls)
