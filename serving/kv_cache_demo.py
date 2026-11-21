"""KV Cache 到底省了什么：手写两种贪心解码，对比速度（第3节项目实战）。

自回归生成每步只产出 1 个 token，但要算它就得对"之前所有 token"做注意力。
- **不用 KV Cache**：每一步都把整条序列重新喂进模型，重算所有层的 K、V。
  第 t 步要算 t 个位置 → 整段生成是 O(n²)。
- **用 KV Cache**：把每个 token 算过的 K、V 缓存下来，下一步只喂**新 token**、
  只算它一个位置的 Q，去和缓存里的 K、V 做注意力 → 整段生成是 O(n)。

两种方式在贪心解码下输出**完全相同**，只是速度差很多。这就是所有推理引擎默认开 KV Cache、
并且要专门管理这块显存（见 PagedAttention）的原因。

跑法（默认 gpt2，CPU 即可；首次会联网下载约 250MB 权重）：
    uv run python -m serving.kv_cache_demo
    uv run python -m serving.kv_cache_demo --model gpt2 --tokens 80
"""
from __future__ import annotations

import argparse
import time


def greedy_no_cache(model, input_ids, n_new: int):
    """每步重喂整条序列、重算全部 K/V（use_cache=False）。"""
    import torch

    seq = input_ids
    with torch.no_grad():
        for _ in range(n_new):
            out = model(input_ids=seq, use_cache=False)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
    return seq[:, input_ids.shape[1]:]


def greedy_with_cache(model, input_ids, n_new: int):
    """只有第一步喂整条 prompt；之后每步只喂新 token，复用缓存的 K/V。"""
    import torch

    past = None
    cur = input_ids
    generated = []
    with torch.no_grad():
        for _ in range(n_new):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(next_id)
            cur = next_id            # 关键：下一步只喂这一个新 token
    return torch.cat(generated, dim=1)


def run_demo(model_name: str = "gpt2",
             prompt: str = "Safety stock in supply chains is",
             n_new: int = 64) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    input_ids = tok(prompt, return_tensors="pt").input_ids

    # 预热一次，避免首次图构建/线程启动污染计时
    greedy_with_cache(model, input_ids, 2)

    t0 = time.perf_counter()
    out_no = greedy_no_cache(model, input_ids, n_new)
    t_no = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_yes = greedy_with_cache(model, input_ids, n_new)
    t_yes = time.perf_counter() - t0

    same = torch.equal(out_no, out_yes)
    return {
        "model": model_name,
        "n_new": n_new,
        "same_output": same,
        "no_cache_s": t_no,
        "with_cache_s": t_yes,
        "speedup": t_no / t_yes if t_yes > 0 else 0.0,
        "text": tok.decode(out_yes[0]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="对比开/关 KV Cache 的生成速度")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--prompt", default="Safety stock in supply chains is")
    p.add_argument("--tokens", type=int, default=64)
    args = p.parse_args()

    r = run_demo(args.model, args.prompt, args.tokens)
    print(f"\n模型 {r['model']}，生成 {r['n_new']} 个 token，贪心解码")
    print(f"  输出是否一致     : {r['same_output']}  （应为 True——KV Cache 不改变结果）")
    print(f"  不用 KV Cache    : {r['no_cache_s']:.2f} s   (每步重算整条，O(n²))")
    print(f"  用   KV Cache    : {r['with_cache_s']:.2f} s   (每步只算新 token，O(n))")
    print(f"  加速             : {r['speedup']:.1f}×")
    print(f"\n续写内容：{r['text']!r}")


if __name__ == "__main__":
    main()
