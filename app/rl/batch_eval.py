"""RL 模型批量评测 — 语义+关键词评分 + RAGas + 6维奖励。

Ported from XIAOMI_SU7_RAG/src/rl/batch_eval.py。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS = "data/training/qa_pairs/test_qa_pair_verify.json"


def main():
    parser = argparse.ArgumentParser(description="RL model batch evaluation")
    parser.add_argument("--model", default="su7_rl")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--input", default=DEFAULT_QUESTIONS, help="Test questions file")
    parser.add_argument("--output", default="data/training/benchmark/rl_eval_results.json")
    parser.add_argument("--dry-run", action="store_true", help="Only 5 items")
    parser.add_argument("--skip-ragas", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    # Load questions
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return

    with open(input_path, encoding="utf-8") as f:
        samples = json.load(f)
    questions = [(s.get("question", ""), s.get("answer", "")) for s in samples if isinstance(s, dict)]
    if args.dry_run:
        questions = questions[:5]

    # Checkpoint
    checkpoint_path = Path(args.output).with_suffix(".ckpt.jsonl")
    done_ids = set()
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                done_ids.add(item.get("question", ""))

    # Load scorer
    from app.eval.scorer import report_score, extract_keywords
    from app.rl.reward_model import compute_reward
    from app.rl.infer_rl import RLInferenceEngine

    engine = RLInferenceEngine(vllm_url=args.vllm_url, model=args.model)
    results = []
    scores = []

    for i, (q, ref) in enumerate(questions):
        if q in done_ids:
            continue
        logger.info(f"[{i+1}/{len(questions)}] {q[:60]}")
        try:
            result = engine.run(q)
            score = report_score(result["answer"], ref, ref_keywords=extract_keywords(ref))
            reward = compute_reward(q, result["trajectory"])
            scores.append(score)
            entry = {"question": q, "answer": result["answer"], "ref": ref,
                     "score": score, "reward": reward,
                     "trajectory": result["trajectory"], "elapsed_ms": result["elapsed_ms"]}
            results.append(entry)

            # Checkpoint
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception(f"Failed: {q[:60]}")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\n{'='*60}")
    print(f"RL Model Evaluation ({len(results)} items)")
    print(f"  Avg score:  {avg_score:.4f}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results → {args.output}")


if __name__ == "__main__":
    main()
