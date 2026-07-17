#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent 流水线测试脚本。

对应 CarVoice_Agent 的 test.py + intent_client.py + reject_client.py + nlu_client.py。
验证仲裁、NLU、拒识、改写、关联判断、技能执行等核心链路。

用法:
  # 单条测试
  python scripts/test_agent.py --query "请导航到公司"

  # 批量测试（从文件读取，每行一条）
  python scripts/test_agent.py --file data/nlu/single_test.txt

  # 交互模式
  python scripts/test_agent.py --interactive

  # 评测模式（有标准答案的测试集）
  python scripts/test_agent.py --eval --file data/nlu/multi_test.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("test_agent")


def test_single(query: str, session_id: str | None = None) -> dict:
    from app.core.orchestrator import ChatOrchestrator

    t0 = time.perf_counter()
    orch = ChatOrchestrator()
    resp = orch.handle(query, session_id=session_id)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "query": query,
        "type": resp.type,
        "text": resp.text,
        "route": resp.trace.route,
        "confidence": resp.trace.classifier_confidence,
        "fallback_reason": resp.trace.fallback_reason,
        "citations": [c.model_dump() for c in resp.citations],
        "session_id": resp.session_id,
        "latency_ms": latency_ms,
    }


def test_batch(filepath: str) -> list[dict]:
    results = []
    with open(filepath, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    logger.info("批量测试 %d 条...", len(lines))
    for i, line in enumerate(lines):
        result = test_single(line)
        results.append(result)
        if (i + 1) % 50 == 0:
            logger.info("进度: %d/%d", i + 1, len(lines))
    return results


def test_interactive() -> None:
    session_id: str | None = None
    print("SU7_CarVoice_Fusion Agent 交互测试")
    print("输入 query 开始，输入 :q 退出，输入 :new 新建会话\n")
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query in (":q", ":quit", ":exit"):
            break
        if query == ":new":
            session_id = None
            print("[新会话]")
            continue
        if not query:
            continue

        result = test_single(query, session_id=session_id)
        session_id = result["session_id"]
        print(f"  type:     {result['type']}")
        print(f"  route:    {result['route']} ({result['confidence']})")
        print(f"  text:     {result['text'][:80]}")
        if result["citations"]:
            print(f"  cite:     {result['citations']}")
        if result["fallback_reason"]:
            print(f"  fallback: {result['fallback_reason']}")
        print(f"  latency:  {result['latency_ms']}ms")
        print()


def test_eval(filepath: str) -> None:
    results = test_batch(filepath)
    total = len(results)
    type_counts: dict[str, int] = {}
    for r in results:
        t = r["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0
    task_count = type_counts.get("task_result", 0)
    faq_count = type_counts.get("faq_answer", 0)
    chat_count = type_counts.get("chitchat", 0)
    clar_count = type_counts.get("clarification", 0)

    print(f"\n{'='*60}")
    print(f"📊 Agent 批量测试报告")
    print(f"{'='*60}")
    print(f"  总数:         {total}")
    print(f"  task_result:  {task_count} ({task_count/total*100:.0f}%)")
    print(f"  faq_answer:   {faq_count} ({faq_count/total*100:.0f}%)")
    print(f"  chitchat:     {chat_count} ({chat_count/total*100:.0f}%)")
    print(f"  clarification:{clar_count} ({clar_count/total*100:.0f}%)")
    print(f"  平均延迟:     {avg_latency:.0f} ms")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 流水线测试")
    parser.add_argument("--query", help="单条测试")
    parser.add_argument("--file", help="批量测试文件（每行一条 query）")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--eval", action="store_true", help="评测模式（统计分类分布）")
    args = parser.parse_args()

    if args.interactive:
        test_interactive()
    elif args.eval and args.file:
        test_eval(args.file)
    elif args.file:
        results = test_batch(args.file)
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
    elif args.query:
        result = test_single(args.query)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
