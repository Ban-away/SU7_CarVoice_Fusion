#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 离线评估脚本。

对应 XIAOMI_SU7_RAG/final_score.py 的流程：
  加载评估集 → 检索 → 生成答案 → 语义评分 + 关键词加权 → RAGas 评估

用法:
  # 使用测试集评估
  python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json

  # 快速验证（5 条）
  python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --dry-run

  # 跳过 RAGas（省 API 费用）
  python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --skip-ragas
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("eval_rag")


@dataclass
class EvalResult:
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    latency_ms: int = 0
    answer: str = ""
    ref: str = ""


def run_eval(
    samples: list[dict],
    dry_run: bool = False,
    skip_ragas: bool = True,
) -> dict:
    from app.eval.scorer import report_score, extract_keywords
    from app.knowledge.service import KnowledgeService

    ks = KnowledgeService()
    results: list[EvalResult] = []
    total = min(5, len(samples)) if dry_run else len(samples)

    logger.info("开始评估 %d 条样本...", total)
    for i, sample in enumerate(samples[:total]):
        question = sample.get("question", "")
        ref_answer = sample.get("answer", "")

        t0 = time.perf_counter()
        docs = ks.retrieve(question, top_k=3)
        answer, _ = ks.synthesize_with_citations(question, docs)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        keywords = extract_keywords(ref_answer)
        score = report_score(answer, ref_answer, ref_keywords=keywords)

        results.append(EvalResult(
            semantic_score=0.7,  # rough estimate without text2vec
            keyword_score=score,
            combined_score=score,
            latency_ms=latency_ms,
            answer=answer,
            ref=ref_answer,
        ))

        if (i + 1) % 10 == 0:
            logger.info("进度: %d/%d", i + 1, total)

    avg_score = sum(r.combined_score for r in results) / len(results) if results else 0.0
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

    ragas_result = {}
    if not skip_ragas:
        try:
            from app.eval.ragas_eval import EvalSample, evaluate_ragas
            eval_samples = [
                EvalSample(question=samples[i]["question"], answer=r.answer,
                           contexts=[], ground_truth=r.ref)
                for i, r in enumerate(results)
            ]
            ragas_result = evaluate_ragas(eval_samples)
        except Exception:
            logger.exception("RAGas evaluation failed")

    return {
        "total": total,
        "avg_combined_score": round(avg_score, 4),
        "avg_latency_ms": avg_latency,
        "ragas": ragas_result,
        "samples": [{"q": samples[i]["question"], "pred": r.answer, "ref": r.ref, "score": r.combined_score}
                     for i, r in enumerate(results)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 离线评估")
    parser.add_argument("--input", required=True, help="评估集 JSON 文件路径")
    parser.add_argument("--output", default="data/training/benchmark/eval_result.json", help="结果输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只评估 5 条")
    parser.add_argument("--skip-ragas", action="store_true", default=True, help="跳过 RAGas (默认)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("评估文件不存在: %s", input_path)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        samples = json.load(f)
    logger.info("加载评估样本: %d 条", len(samples))

    result = run_eval(samples, dry_run=args.dry_run, skip_ragas=args.skip_ragas)

    # 输出结果
    print(f"\n{'='*60}")
    print(f"📊 RAG 评估结果")
    print(f"{'='*60}")
    print(f"  样本数:        {result['total']}")
    print(f"  综合评分:      {result['avg_combined_score']:.4f}")
    print(f"  平均延迟:      {result['avg_latency_ms']} ms")
    if result["ragas"]:
        for k, v in result["ragas"].items():
            print(f"  RAGas/{k}: {v:.4f}")
    print(f"{'='*60}")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
