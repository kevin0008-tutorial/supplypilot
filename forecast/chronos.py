"""Chronos-2 零样本预测（可选，课堂演示用）。

Chronos 是时间序列基础模型：把数值序列缩放、切块当作 "token"，在海量序列上预训练，
用的时候不训练、直接零样本预测，还能直接输出分位数。

它依赖 torch 和 chronos 包，体积大、下载慢，所以**不放进项目默认依赖**。
想跑这段的学员单独装：

    uv add chronos-forecasting      # 会一并装上 torch（CPU 版即可）

没装时本文件不会报错，整个项目照常运行、测试照常通过——补货工具会退回用
`forecast.baseline` 的经验分位数。这本身就是一个工程取舍：核心逻辑不绑死在重依赖上。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def chronos_available() -> bool:
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def chronos_quantile_forecast(
    units: Sequence[float], horizon: int, q: float = 0.95
):
    """用 Chronos 预测未来 horizon 天的 q 分位数日需求（numpy 数组）。

    没装 Chronos 时抛出带安装提示的 ImportError，由调用方决定是否回退到基线。
    """
    if not chronos_available():
        raise ImportError(
            "未安装 Chronos。运行 `uv add chronos-forecasting` 后再试，"
            "或改用 forecast.baseline 的分位数方法。"
        )

    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32
    )
    context = torch.tensor(np.asarray(units, dtype=float))
    quantiles, _ = pipeline.predict_quantiles(
        context=context, prediction_length=horizon, quantile_levels=[q]
    )
    return quantiles[0, :, 0].numpy()
