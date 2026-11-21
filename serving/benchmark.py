"""推理服务压测：量 TTFT、TPOT、吞吐量（第3节项目实战）。

只用 Python 标准库（urllib + 线程），不装任何包——所以 Kaggle、Colab、本地都能直接跑，
打到任何 **OpenAI 兼容** 的 /v1/chat/completions 接口：vLLM、Ollama、云端 API 都行。

三个核心指标（面试必问）：
- TTFT  (Time To First Token)  首 token 延迟：按回车到看到第一个字的时间。决定"反应快不快"。
- TPOT  (Time Per Output Token) 每 token 生成时间：第一个字之后，平均每个字多久。决定"打字快不快"。
- 吞吐量 (throughput)           整个系统每秒总共吐多少 token。决定"能同时服务多少人"。
  单条请求的 TTFT/TPOT 会随并发上升而变差，但系统总吞吐通常先升后平——这正是要压测的权衡。

跑法：
    # 先起一个服务（vLLM 或 Ollama，见课件），然后：
    uv run python -m serving.benchmark --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-1.5B-Instruct --concurrency 1,4,8,16 --requests 32
    # 打 Ollama：--base-url http://localhost:11434/v1 --model qwen2.5:1.5b
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class RequestResult:
    ok: bool
    ttft: float = 0.0          # 秒
    total_time: float = 0.0    # 秒
    n_out: int = 0             # 输出 token 数
    error: str = ""

    @property
    def tpot(self) -> float:
        """每输出 token 时间（排除首 token）。"""
        if self.n_out <= 1:
            return 0.0
        return (self.total_time - self.ttft) / (self.n_out - 1)


def run_one(base_url: str, model: str, prompt: str, max_tokens: int,
            api_key: str = "EMPTY", timeout: float = 120.0) -> RequestResult:
    """发一条流式请求，计时首 token 和总时长。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},   # vLLM 会在最后给 usage
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    start = time.perf_counter()
    ttft = 0.0
    last = start
    chunks = 0
    usage_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens", 0)
                for choice in obj.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        now = time.perf_counter()
                        if ttft == 0.0:
                            ttft = now - start
                        last = now
                        chunks += 1
    except Exception as e:  # noqa: BLE001  压测里把错误计进失败率即可
        return RequestResult(ok=False, error=f"{type(e).__name__}: {e}")

    n_out = usage_tokens or chunks        # 优先用服务回报的 usage，否则按 chunk 近似
    return RequestResult(ok=True, ttft=ttft, total_time=last - start, n_out=n_out)


# --------------------------------------------------------------------------- #
# 纯统计函数（可单测，不碰网络）
# --------------------------------------------------------------------------- #
def percentile(values: list[float], q: float) -> float:
    """线性插值分位数，q ∈ [0,1]。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    frac = idx - lo
    if lo + 1 >= len(s):
        return s[-1]
    return s[lo] + frac * (s[lo + 1] - s[lo])


def summarize(results: list[RequestResult], wall_time: float) -> dict:
    """把一批请求结果汇总成指标。wall_time 是这批请求的墙钟总时长。"""
    ok = [r for r in results if r.ok]
    n_fail = len(results) - len(ok)
    total_out = sum(r.n_out for r in ok)
    ttfts = [r.ttft for r in ok if r.ttft > 0]
    tpots = [r.tpot for r in ok if r.tpot > 0]
    return {
        "requests": len(results),
        "failed": n_fail,
        "wall_time_s": round(wall_time, 3),
        "total_output_tokens": total_out,
        "throughput_tok_s": round(total_out / wall_time, 1) if wall_time > 0 else 0.0,
        "ttft_p50_ms": round(percentile(ttfts, 0.5) * 1000, 1),
        "ttft_p90_ms": round(percentile(ttfts, 0.9) * 1000, 1),
        "tpot_mean_ms": round(statistics.mean(tpots) * 1000, 1) if tpots else 0.0,
    }


def benchmark(base_url: str, model: str, concurrency: int, n_requests: int,
              prompt: str, max_tokens: int, api_key: str = "EMPTY") -> dict:
    """在固定并发下打 n_requests 条请求，返回汇总指标。"""
    results: list[RequestResult] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, base_url, model, prompt, max_tokens, api_key)
                   for _ in range(n_requests)]
        for f in futures:
            results.append(f.result())
    wall = time.perf_counter() - start
    out = summarize(results, wall)
    out["concurrency"] = concurrency
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI 兼容推理服务压测")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", default="1,4,8,16",
                   help="逗号分隔的并发档位，逐档压测")
    p.add_argument("--requests", type=int, default=32, help="每档总请求数")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--prompt",
                   default="用三句话解释什么是供应链里的安全库存。")
    args = p.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    print(f"服务 {args.base_url}  模型 {args.model}  每档 {args.requests} 请求\n")
    print(f"{'并发':>4} {'吞吐(tok/s)':>12} {'TTFT p50(ms)':>14} "
          f"{'TTFT p90(ms)':>14} {'TPOT(ms)':>10} {'失败':>5}")
    print("-" * 66)
    for c in levels:
        m = benchmark(args.base_url, args.model, c, args.requests,
                      args.prompt, args.max_tokens, args.api_key)
        print(f"{m['concurrency']:>4} {m['throughput_tok_s']:>12} "
              f"{m['ttft_p50_ms']:>14} {m['ttft_p90_ms']:>14} "
              f"{m['tpot_mean_ms']:>10} {m['failed']:>5}")
    print("\n看点：并发↑→系统吞吐先升后平；单请求 TTFT/TPOT 随并发变差。"
          "\n把这张表连同模型/量化/显存写进 docs/serving_report.md。")


if __name__ == "__main__":
    main()
