import importlib.util
import subprocess
import sys

import pytest

from serving.benchmark import RequestResult, percentile, summarize
from serving.kvcache_calc import PRESETS


# --------------------------------------------------------------------------- #
# 显存估算：对着公式核对
# --------------------------------------------------------------------------- #
def test_kv_bytes_per_token_matches_formula():
    spec = PRESETS["qwen2.5-7b"]            # layers=28, kv_heads=4, head_dim=128
    # 2 × 28 × 4 × 128 × 2(fp16) = 57344 字节 = 56 KB/token
    assert spec.kv_bytes_per_token("fp16") == 57344.0
    # int8 正好减半
    assert spec.kv_bytes_per_token("int8") == 28672.0


def test_weight_bytes_and_total():
    spec = PRESETS["qwen2.5-7b"]
    assert spec.weight_bytes("fp16") == pytest.approx(7.62e9 * 2)
    total = spec.total_bytes(seq=1000, batch=1, weight_dtype="fp16", kv_dtype="fp16")
    assert total == pytest.approx(7.62e9 * 2 + 57344.0 * 1000)


def test_max_concurrency_drops_with_longer_seq():
    spec = PRESETS["qwen2.5-7b"]
    few = spec.max_concurrency(vram_gib=24, seq=8192)
    many = spec.max_concurrency(vram_gib=24, seq=1024)
    assert 0 < few < many            # 上下文越长，能并发的条数越少
    assert spec.max_concurrency(vram_gib=1, seq=1024) == 0   # 权重都放不下


# --------------------------------------------------------------------------- #
# 压测统计函数：纯函数，不碰网络
# --------------------------------------------------------------------------- #
def test_percentile_linear_interpolation():
    assert percentile([], 0.5) == 0.0
    assert percentile([42.0], 0.9) == 42.0
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([0.0, 1.0, 2.0, 3.0, 4.0], 0.5) == 2.0


def test_summarize_aggregates():
    rs = [
        RequestResult(ok=True, ttft=0.10, total_time=1.10, n_out=101),  # tpot=10ms
        RequestResult(ok=True, ttft=0.20, total_time=2.20, n_out=201),
        RequestResult(ok=False, error="boom"),
    ]
    m = summarize(rs, wall_time=2.0)
    assert m["requests"] == 3
    assert m["failed"] == 1
    assert m["total_output_tokens"] == 302          # 101 + 201
    assert m["throughput_tok_s"] == 151.0           # 302 / 2.0
    assert m["ttft_p50_ms"] == pytest.approx(150.0)  # p50 of {100,200}ms
    assert m["tpot_mean_ms"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# KV Cache 正确性：两种解码必须给出完全相同的 token（离线 tiny 模型，不下载）
#
# 注意：torch 和 LightGBM 各自带一套 OpenMP，同一个进程里一起 import 会死锁
# （和第2节 report.py 同一个坑）。所以这里把 torch 测试放到**单独子进程**跑，
# 主 pytest 进程（还要跑 test_lightgbm）永远不 import torch。
# --------------------------------------------------------------------------- #
def _transformers_installed() -> bool:
    # 只查是否安装，不 import（import torch 会把 OpenMP 拉进主进程）
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


_SUBPROC = """
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from serving.kv_cache_demo import greedy_no_cache, greedy_with_cache
torch.manual_seed(0)
cfg = GPT2Config(vocab_size=64, n_positions=128, n_embd=32, n_layer=2, n_head=2)
m = GPT2LMHeadModel(cfg).eval()
ids = torch.randint(0, 64, (1, 5))
a = greedy_no_cache(m, ids, 8)
b = greedy_with_cache(m, ids, 8)
assert a.shape == (1, 8)
assert torch.equal(a, b)
print("KVCACHE_OK")
"""


@pytest.mark.skipif(not _transformers_installed(), reason="未装 torch/transformers")
def test_kv_cache_gives_identical_output_offline():
    proc = subprocess.run([sys.executable, "-c", _SUBPROC],
                          capture_output=True, text=True, timeout=300)
    assert "KVCACHE_OK" in proc.stdout, proc.stderr[-2000:]
