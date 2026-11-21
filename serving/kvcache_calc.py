"""显存估算器：模型权重 + KV Cache（第3节"现场估算显存"的工具）。

两条要记住的公式：

1) 模型权重显存 = 参数量 × 每个数的字节数
   FP16/BF16 = 2 字节，INT8 = 1 字节，INT4 ≈ 0.5 字节。
   例：7B 模型 FP16 ≈ 7e9 × 2 = 14 GB。

2) KV Cache 显存（真正随"并发 × 上下文长度"膨胀的那块）：
   每个 token 每层都要缓存 K 和 V 两个向量，大小 = num_kv_heads × head_dim。
   所以：
     每 token 的 KV 字节 = 2(K和V) × num_layers × num_kv_heads × head_dim × dtype_bytes
     KV Cache 总字节  = 每 token 字节 × 序列长度 × 并发条数(batch)

   注意现代模型用 GQA（分组查询注意力）：num_kv_heads < num_attention_heads，
   KV Cache 因此小很多——这正是它被发明出来的原因。

纯标准库，无依赖。直接跑：
    uv run python -m serving.kvcache_calc                 # 看几个预设模型
    uv run python -m serving.kvcache_calc --model qwen2.5-7b --seq 8192 --batch 16
    uv run python -m serving.kvcache_calc --vram 8        # 8GB 显存能塞多少并发
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

GiB = 1024 ** 3
DTYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: float          # 总参数量
    num_layers: int
    num_kv_heads: int      # GQA 下的 KV 头数（不是注意力头数）
    head_dim: int

    def weight_bytes(self, dtype: str = "fp16") -> float:
        return self.params * DTYPE_BYTES[dtype]

    def kv_bytes_per_token(self, dtype: str = "fp16") -> float:
        # 2 = K 和 V 各一份
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * DTYPE_BYTES[dtype]

    def kv_bytes(self, seq: int, batch: int, dtype: str = "fp16") -> float:
        return self.kv_bytes_per_token(dtype) * seq * batch

    def total_bytes(self, seq: int, batch: int,
                    weight_dtype: str = "fp16", kv_dtype: str = "fp16") -> float:
        return self.weight_bytes(weight_dtype) + self.kv_bytes(seq, batch, kv_dtype)

    def max_concurrency(self, vram_gib: float, seq: int,
                        weight_dtype: str = "fp16", kv_dtype: str = "fp16") -> int:
        """给定显存，扣掉权重后还能放下几条长度为 seq 的序列。"""
        free = vram_gib * GiB - self.weight_bytes(weight_dtype)
        if free <= 0:
            return 0
        per_seq = self.kv_bytes_per_token(kv_dtype) * seq
        return int(free // per_seq)


# 几个常见指令模型（取自各自的 config.json；GQA 的 kv_heads 已单列）
PRESETS: dict[str, ModelSpec] = {
    "qwen2.5-1.5b": ModelSpec("Qwen2.5-1.5B-Instruct", 1.54e9, 28, 2, 128),
    "qwen2.5-7b":   ModelSpec("Qwen2.5-7B-Instruct", 7.62e9, 28, 4, 128),
    "llama3-8b":    ModelSpec("Llama-3-8B-Instruct", 8.03e9, 32, 8, 128),
}


def _fmt_gb(b: float) -> str:
    return f"{b / GiB:6.2f} GB"


def report(spec: ModelSpec, seq: int, batch: int,
           weight_dtype: str, kv_dtype: str) -> None:
    print(f"\n模型 {spec.name}  "
          f"(layers={spec.num_layers}, kv_heads={spec.num_kv_heads}, head_dim={spec.head_dim})")
    print(f"  权重({weight_dtype:>4})          {_fmt_gb(spec.weight_bytes(weight_dtype))}  "
          f"= {spec.params/1e9:.2f}B × {DTYPE_BYTES[weight_dtype]} 字节")
    per_tok = spec.kv_bytes_per_token(kv_dtype)
    print(f"  每 token KV({kv_dtype:>4})       {per_tok/1024:6.1f} KB  "
          f"= 2 × {spec.num_layers} × {spec.num_kv_heads} × {spec.head_dim} × {DTYPE_BYTES[kv_dtype]}")
    print(f"  KV Cache @ seq={seq}×batch={batch}   {_fmt_gb(spec.kv_bytes(seq, batch, kv_dtype))}")
    print(f"  合计                   {_fmt_gb(spec.total_bytes(seq, batch, weight_dtype, kv_dtype))}")


def main() -> None:
    p = argparse.ArgumentParser(description="LLM 显存 / KV Cache 估算器")
    p.add_argument("--model", choices=list(PRESETS), help="预设模型；不填则全部列出")
    p.add_argument("--seq", type=int, default=4096, help="上下文长度（prompt+生成）")
    p.add_argument("--batch", type=int, default=1, help="并发序列条数")
    p.add_argument("--weight-dtype", choices=list(DTYPE_BYTES), default="fp16")
    p.add_argument("--kv-dtype", choices=list(DTYPE_BYTES), default="fp16")
    p.add_argument("--vram", type=float, help="给定显存(GB)，反算能放多少并发")
    args = p.parse_args()

    specs = [PRESETS[args.model]] if args.model else list(PRESETS.values())
    for spec in specs:
        report(spec, args.seq, args.batch, args.weight_dtype, args.kv_dtype)
        if args.vram:
            n = spec.max_concurrency(args.vram, args.seq, args.weight_dtype, args.kv_dtype)
            print(f"  → {args.vram:g}GB 显存，扣掉权重后约可并发 "
                  f"{n} 条长度 {args.seq} 的序列")
    print()


if __name__ == "__main__":
    main()
