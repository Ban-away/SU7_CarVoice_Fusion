#!/usr/bin/env python
"""vLLM 性能压测 — TTFT + 吞吐量。

Ported from XIAOMI_SU7_RAG/deploy/benchmark.py。
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def single_query(client: OpenAI, model: str, prompt: str, max_tokens: int = 128) -> dict:
    t0 = time.time()
    first_token_time = None
    total_tokens = 0

    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0, stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            total_tokens += 1
            if first_token_time is None:
                first_token_time = time.time()
    elapsed = time.time() - t0
    ttft = (first_token_time - t0) * 1000 if first_token_time else 0
    return {"ttft_ms": ttft, "elapsed_s": elapsed, "tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser(description="vLLM benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--prompt", default="小米SU7的续航里程是多少？请详细回答。")
    args = parser.parse_args()

    client = OpenAI(base_url=args.url, api_key="not-needed")
    print(f"Benchmark: {args.num_requests} requests @ concurrency {args.concurrency}")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(single_query, client, args.model, args.prompt) for _ in range(args.num_requests)]
        for f in as_completed(futures):
            results.append(f.result())

    elapsed = time.time() - t0
    total_tokens = sum(r["tokens"] for r in results)
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    throughput = total_tokens / elapsed

    print(f"\n{'='*50}")
    print(f"Requests:    {len(results)}")
    print(f"Total time:  {elapsed:.1f}s")
    print(f"TTFT avg:    {avg_ttft:.0f}ms")
    print(f"Throughput:  {throughput:.0f} token/s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
