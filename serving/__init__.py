"""第3节：大模型推理服务（serving）。

- kvcache_calc.py  —— 显存/KV Cache 估算器（纯 Python，课上现场算）
- benchmark.py     —— OpenAI 兼容服务压测（纯标准库，Kaggle/Colab/本地都能跑）
- kv_cache_demo.py —— HF Transformers 手动生成，对比开/关 KV Cache 的速度
- openai_client_demo.py —— 一段最小客户端，证明 vLLM / Ollama 的 /v1 接口可用
"""
