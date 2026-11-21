"""最小客户端：证明 vLLM / Ollama 的 OpenAI 兼容接口能用（第3节项目实战）。

只用标准库。vLLM 和 Ollama 都暴露同一套 /v1/chat/completions，所以同一段代码、
只换 base_url 和模型名就能在两者之间切——这正是我们统一调用层（第4节）的基础。

跑法：
    # vLLM（Kaggle/Colab 上起的服务，端口转发到本地或直接用其 URL）：
    uv run python -m serving.openai_client_demo \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-1.5B-Instruct
    # Ollama（本地 ollama serve，先 ollama pull qwen2.5:1.5b）：
    uv run python -m serving.openai_client_demo \
        --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def stream_chat(base_url: str, model: str, prompt: str,
                api_key: str = "EMPTY", max_tokens: int = 256) -> None:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            for choice in json.loads(data).get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    sys.stdout.write(piece)
                    sys.stdout.flush()
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI 兼容接口最小客户端")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--prompt", default="一句话解释 vLLM 为什么比朴素推理快。")
    args = p.parse_args()
    stream_chat(args.base_url, args.model, args.prompt, args.api_key)


if __name__ == "__main__":
    main()
