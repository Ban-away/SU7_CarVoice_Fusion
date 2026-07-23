#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SFT 训练数据构造脚本 — Summary & Rerank 数据集生成。

对应 XIAOMI_SU7_RAG/generate_sft_data.py 的功能：
  1. 从 QA 对生成 Summary 训练数据（instruction→output 格式）
  2. 从检索结果生成 Rerank 训练数据（query→positive/negative docs）

用法:
  # 默认：生成 Summary + Rerank 数据
  python scripts/generate_sft_data.py

  # 仅 Summary
  python scripts/generate_sft_data.py --step summary

  # 仅 Rerank
  python scripts/generate_sft_data.py --step rerank

  # 自定义路径
  python scripts/generate_sft_data.py \
    --qa-file data/training/qa_pairs/qa_pair.json \
    --output-dir data/training/sft_data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("generate_sft_data")

DEFAULT_QA_FILE = "data/training/qa_pairs/qa_pair.json"
DEFAULT_OUTPUT_DIR = "data/training/sft_data"


def _load_qa_pairs(path: str) -> list[dict]:
    """加载 QA 对文件。"""
    if not os.path.exists(path):
        logger.warning("QA 文件不存在: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("加载 %d 条 QA 对 → %s", len(data), path)
    return data


def _build_summary_data(qa_pairs: list[dict], output_dir: Path) -> None:
    """构建 Summary 训练数据集 (instruction → output)。"""
    if not qa_pairs:
        logger.warning("没有 QA 对，跳过 Summary 数据构造")
        return

    SYSTEM_PROMPT = "你是小米 SU7 用户手册问答专家，请根据参考文档准确简洁地回答用户问题。"

    dataset = []
    for i, qa in enumerate(qa_pairs):
        question = qa.get("question") or qa.get("instruction") or qa.get("query", "")
        answer = qa.get("answer") or qa.get("output", "")
        context = qa.get("context") or qa.get("content", "")

        if not question or not answer:
            continue

        entry = {
            "id": f"summary_{i:05d}",
            "instruction": question,
            "input": context[:500] if context else "",
            "output": answer,
            "system": SYSTEM_PROMPT,
        }
        dataset.append(entry)

    if not dataset:
        logger.warning("构造后无有效 Summary 数据")
        return

    # Train/test split (80/20)
    split_idx = int(len(dataset) * 0.8)
    train = dataset[:split_idx]
    test = dataset[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(train, output_dir / "summary_train.json")
    _save_json(test, output_dir / "summary_test.json")
    logger.info("Summary 数据集: %d train, %d test → %s", len(train), len(test), output_dir)


def _build_rerank_data(qa_pairs: list[dict], output_dir: Path) -> None:
    """构建 Rerank 训练数据集 (query → positive/negative docs)。

    如果有 retrieved_docs 字段则使用真实检索结果，
    否则使用简单的正负例构造。
    """
    if not qa_pairs:
        logger.warning("没有 QA 对，跳过 Rerank 数据构造")
        return

    dataset = []
    for i, qa in enumerate(qa_pairs):
        question = qa.get("question") or qa.get("instruction") or qa.get("query", "")
        answer = qa.get("answer") or qa.get("output", "")
        retrieved = qa.get("retrieved_docs") or qa.get("context") or []

        if not question:
            continue

        # Use answer as positive doc if no retrieved docs
        if isinstance(retrieved, list) and len(retrieved) > 0:
            positive = retrieved[0] if isinstance(retrieved[0], str) else retrieved[0].get("content", str(retrieved[0]))
        else:
            positive = answer

        # Create negatives from other answers
        neg_candidates = [
            qa2.get("answer", "")
            for j, qa2 in enumerate(qa_pairs)
            if j != i and qa2.get("answer")
        ]
        negatives = neg_candidates[:3] if neg_candidates else [""]  # top-3 negatives

        entry = {
            "id": f"rerank_{i:05d}",
            "query": question,
            "positive": [positive] if positive else [],
            "negative": negatives,
        }
        dataset.append(entry)

    if not dataset:
        logger.warning("构造后无有效 Rerank 数据")
        return

    # Train/test split (80/20)
    split_idx = int(len(dataset) * 0.8)
    train = dataset[:split_idx]
    test = dataset[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(train, output_dir / "rerank_train.json")
    _save_json(test, output_dir / "rerank_test.json")
    logger.info("Rerank 数据集: %d train, %d test → %s", len(train), len(test), output_dir)


def _save_json(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("  saved: %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 训练数据构造 (Summary + Rerank)")
    parser.add_argument(
        "--step", choices=["summary", "rerank", "all"], default="all",
        help="生成哪个数据集 (default: all)",
    )
    parser.add_argument(
        "--qa-file", default=DEFAULT_QA_FILE,
        help="QA 对 JSON 文件路径",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    args = parser.parse_args()

    qa_pairs = _load_qa_pairs(args.qa_file)
    output_dir = Path(args.output_dir)

    if args.step in ("summary", "all"):
        _build_summary_data(qa_pairs, output_dir)

    if args.step in ("rerank", "all"):
        _build_rerank_data(qa_pairs, output_dir)

    logger.info("SFT 数据构造完成。")


if __name__ == "__main__":
    main()
