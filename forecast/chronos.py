"""Chronos-2 零样本预测（时间序列基础模型路线，可选依赖）。

和 LightGBM 的对照：LightGBM 要你手工做特征、而且只在"这个商品自己的历史"上学；
Chronos 是在**海量不同时间序列**上预训练过的"基础模型"，把数值序列切块、缩放当作
"token"，用的时候**零样本**（不训练）直接预测，还能直接输出分位数。

它依赖 torch + chronos 包，体积大，所以放在**可选依赖组**里，不进默认安装：

    uv sync --extra forecast     # 安装 torch + chronos（CPU 版即可）

没装时本文件不报错，`chronos_available()` 返回 False，报告和工具会自动跳过它。
原理/论文见课件⑦里的外链。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_MODEL = "amazon/chronos-bolt-small"   # 小模型，CPU 秒级
_pipeline = None


def chronos_available() -> bool:
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from chronos import BaseChronosPipeline
        _pipeline = BaseChronosPipeline.from_pretrained(
            _MODEL, device_map="cpu", torch_dtype=torch.float32
        )
    return _pipeline


def _predict_quantiles(units: Sequence[float], horizon: int, levels: list[float]) -> np.ndarray:
    import torch
    pipeline = _get_pipeline()
    context = torch.tensor(np.asarray(units, dtype=float))
    quantiles, _ = pipeline.predict_quantiles(
        context, prediction_length=horizon, quantile_levels=levels
    )
    return quantiles[0].numpy()        # 形状 (horizon, len(levels))


def chronos_forecaster(units: Sequence[float], horizon: int) -> np.ndarray:
    """逐日点预测（取 0.5 分位，即中位数），签名和其它 forecaster 一致。"""
    if not chronos_available():
        raise ImportError("未安装 Chronos。先 `uv sync --extra forecast`。")
    out = _predict_quantiles(units, horizon, [0.5])[:, 0]
    return np.clip(out, 0.0, None)


def chronos_quantile_forecast(units: Sequence[float], horizon: int, q: float = 0.95) -> np.ndarray:
    """逐日 q 分位数预测——Chronos 直接给分位数，不用假设分布。"""
    if not chronos_available():
        raise ImportError("未安装 Chronos。先 `uv sync --extra forecast`。")
    out = _predict_quantiles(units, horizon, [q])[:, 0]
    return np.clip(out, 0.0, None)
