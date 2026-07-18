"""生成本地可答轨迹 — 仅 <search_local> + <answer>，教模型"本地够用不联网"。

Ported from XIAOMI_SU7_RAG/src/rl/build_local_trajectories.py。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

QA_SOURCES = [
    "data/training/test_qa_pair_verify.json",
    "data/training/qa_pairs/train_qa_pair.json",
]


def main():
    parser = argparse.ArgumentParser(description="Build local-only trajectories")
    parser.add_argument("--sample", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Only process 5 items")
    parser.add_argument("--output", default="data/training/rl/local_trajectories.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    from app.knowledge.service import KnowledgeService
    ks = KnowledgeService(web_search_enabled=False)

    # Load questions from QA sources
    questions: list[str] = []
    seen = set()
    for path_str in QA_SOURCES:
        path = Path(path_str)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    q = item.get("question", "") if isinstance(item, dict) else str(item)
                    h = hashlib.md5(q.encode()).hexdigest()
                    if q.strip() and h not in seen:
                        seen.add(h)
                        questions.append(q.strip())

    logger.info(f"Loaded {len(questions)} unique questions from QA sources")

    if args.dry_run:
        questions = questions[:5]
    elif args.sample > 0:
        questions = questions[:args.sample]

    trajectories = []
    for i, q in enumerate(questions):
        logger.info(f"[{i+1}/{len(questions)}] {q[:60]}")
        docs = ks.search_local_docs(q, top_k=5)
        info_parts = [f"【{j}】{d.content}" for j, d in enumerate(docs, 1)] if docs else ["本地知识库未检索到相关内容。"]

        trajectory = (
            f"<search_local>{q}</search_local>\n"
            f"<information>{chr(10).join(info_parts)}</information>\n"
            f"<answer>{docs[0].content if docs else '暂未检索到相关信息，请尝试更具体的问题。'}</answer>"
        )
        trajectories.append({"question": q, "trajectory": trajectory})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(trajectories)} trajectories → {args.output}")


if __name__ == "__main__":
    main()
