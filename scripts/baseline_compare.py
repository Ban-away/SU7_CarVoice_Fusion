#!/usr/bin/env python
"""基线对比测试 — 本地 Qwen3-8B vs OpenAI GPT-4o。

Ported from XIAOMI_SU7_RAG/deploy/baseline_gpt4o.py。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

TEST_QUESTIONS = [
    "小米SU7的续航里程是多少？",
    "小米SU7支持哪些类型的钥匙？",
    "如何开启SU7的空调？",
    "小米SU7 Pro版和Max版有什么区别？",
    "SU7的保养周期是多久？",
]


def score_answer(pred: str, ref: str = "") -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, pred, ref).ratio() if ref else 0.5


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison")
    parser.add_argument("--model", choices=["local", "openai", "both"], default="both")
    parser.add_argument("--local-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--local-model", default="models/Qwen3-8B")
    parser.add_argument("--openai-model", default="gpt-4o")
    args = parser.parse_args()

    results: dict[str, dict] = {}

    if args.model in ("local", "both"):
        local_client = OpenAI(base_url=args.local_url, api_key="not-needed")
        print(f"[LOCAL] Testing {args.local_model}...")
        scores = []
        for q in TEST_QUESTIONS:
            t0 = time.time()
            resp = local_client.chat.completions.create(
                model=args.local_model, messages=[{"role": "user", "content": q}],
                max_tokens=128, temperature=0.0,
            )
            elapsed = time.time() - t0
            answer = resp.choices[0].message.content or ""
            scores.append({"question": q, "answer": answer, "elapsed_s": elapsed})
        results["local"] = {"avg_time": sum(s["elapsed_s"] for s in scores) / len(scores), "samples": scores}
        print(f"  Avg time: {results['local']['avg_time']:.2f}s")

    if args.model in ("openai", "both"):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("[OPENAI] OPENAI_API_KEY not set — skipping")
        else:
            openai_client = OpenAI(api_key=api_key)
            print(f"[OPENAI] Testing {args.openai_model}...")
            scores = []
            for q in TEST_QUESTIONS:
                t0 = time.time()
                resp = openai_client.chat.completions.create(
                    model=args.openai_model, messages=[{"role": "user", "content": q}],
                    max_tokens=128, temperature=0.0,
                )
                elapsed = time.time() - t0
                answer = resp.choices[0].message.content or ""
                scores.append({"question": q, "answer": answer, "elapsed_s": elapsed})
            results["openai"] = {"avg_time": sum(s["elapsed_s"] for s in scores) / len(scores), "samples": scores}
            print(f"  Avg time: {results['openai']['avg_time']:.2f}s")

    if len(results) == 2:
        local_avg = results["local"]["avg_time"]
        openai_avg = results["openai"]["avg_time"]
        print(f"\n{'='*50}")
        print(f"Local:    {local_avg:.2f}s avg")
        print(f"OpenAI:   {openai_avg:.2f}s avg")
        print(f"{'='*50}")

    # Save results
    out_path = Path("data/training/benchmark/baseline_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
