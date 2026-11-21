# 推理性能报告（第3节作业）

> 模板。把 `serving/vllm_server.ipynb` 在 Kaggle/Colab 上跑出来的数字填进下面的表，并写结论。
> 下面的示例数字来自一次 Colab **T4(16GB)** 上的参考运行，**仅供对照格式**——以你自己机器的实测为准。

## 1. 环境

| 项 | 值 |
| --- | --- |
| GPU | 例：Tesla T4, 16 GB（填你的） |
| 推理引擎 | vLLM 0.11.0 |
| 模型 | Qwen/Qwen2.5-1.5B-Instruct（FP16） |
| max-model-len | 4096 |
| 压测脚本 | `serving/benchmark.py`，每档 64 请求，max_tokens=128 |

## 2. 吞吐 / 延迟 随并发变化（Qwen2.5-1.5B，FP16）

| 并发 | 吞吐(tok/s) | TTFT p50(ms) | TTFT p90(ms) | TPOT(ms) | 失败 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | _填_ | _填_ | _填_ | _填_ | 0 |
| 2 | | | | | |
| 4 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 32 | | | | | |

**结论（照着写）：**
- 并发从 1→N，系统总吞吐__先升后平__（连续批处理把 GPU 喂满后到顶）。
- 单请求 TTFT/TPOT 随并发__变差__（排队 + 批内竞争）。
- 在 TTFT p90 还能接受（例如 < 1s）的前提下，最高可用并发约为 __ __。

## 3. 量化与显存：7B 怎么塞进 16GB

| 模型 | 精度 | 权重显存 | 服务占用显存(nvidia-smi) | 并发8 吞吐(tok/s) |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | FP16 | ~14.2 GB | 例：T4 上会 OOM / 极勉强 | — |
| Qwen2.5-7B-Instruct-AWQ | INT4 | ~3.8 GB | _填_ | _填_ |

**结论：** 量化把权重显存降下来，省出的显存给 KV Cache → 支持更长上下文 / 更高并发；
代价是极小的精度损失（AWQ/GPTQ 都是后训练量化，几乎无需重训）。

## 4. 显存估算 vs 实测（`serving/kvcache_calc.py`）

```
# 例：uv run python -m serving.kvcache_calc --model qwen2.5-7b --seq 4096 --vram 16 --weight-dtype int4
权重(int4)   3.55 GB
每 token KV  56.0 KB
→ 16GB 显存，扣掉权重后约可并发 N 条长度 4096 的序列
```

把估算出的"可并发条数"和第 2 节压测里吞吐见顶的并发档__对照__，解释差异（估算没算
激活、碎片、框架预留；vLLM 的 PagedAttention 能把碎片降到很低，所以实际往往更接近估算上限）。

## 5. 一句话总结
（例：在 T4 上，1.5B-FP16 单请求 TTFT ~__ms、峰值吞吐 ~__tok/s@并发__；7B 必须量化到 INT4 才跑得起来。）
