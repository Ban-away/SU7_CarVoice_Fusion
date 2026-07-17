#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练数据生成脚本。

对应 XIAOMI_SU7_RAG/generate_all_data.py + generate_sft_data.py 的流程：
  QA 对生成 → 问题扩展 → 质量过滤 → Summary/Rerank 训练集构建

用法:
  # 生成 QA 对（需配置 LLM_PROVIDER）
  python scripts/generate_data.py --step qa

  # 过滤已有 QA 对
  python scripts/generate_data.py --step filter --input data/training/qa_pairs/qa_pair.json

  # 构建训练数据集
  python scripts/generate_data.py --step dataset --input data/training/qa_pairs/qa_filtered.json

  # 全流程
  python scripts/generate_data.py --step all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("generate_data")


def step_qa(output_dir: Path) -> None:
    from app.knowledge.service import KnowledgeService
    from app.data_pipeline.qa_generator import generate_qa_pairs

    ks = KnowledgeService()
    docs = [d.content for d in ks._documents]
    logger.info("从 %d 条文档生成 QA 对...", len(docs))
    pairs = generate_qa_pairs(docs)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "qa_pair.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info("生成 %d 条 QA 对 → %s", len(pairs), out_path)


def step_filter(input_path: str, output_dir: Path) -> None:
    from app.data_pipeline.qa_filter import filter_qa_pairs

    with open(input_path, encoding="utf-8") as f:
        pairs = json.load(f)
    logger.info("加载 %d 条 QA 对，开始过滤...", len(pairs))
    clean = filter_qa_pairs(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(input_path).stem}_filtered.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    logger.info("过滤完成: %d → %d → %s", len(pairs), len(clean), out_path)


def step_dataset(input_path: str, output_dir: Path) -> None:
    from app.data_pipeline.dataset_builder import build_summary_dataset, build_rerank_dataset

    with open(input_path, encoding="utf-8") as f:
        pairs = json.load(f)
    logger.info("加载 %d 条 QA 对，构建训练数据集...", len(pairs))

    summary_dir = output_dir / "summary"
    build_summary_dataset(pairs, output_dir=str(summary_dir))
    logger.info("Summary 数据集 → %s", summary_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练数据生成")
    parser.add_argument(
        "--step", choices=["qa", "filter", "dataset", "all"], default="qa",
        help="执行步骤 (默认 qa)",
    )
    parser.add_argument("--input", default="", help="输入文件路径")
    parser.add_argument("--output", default="data/training/qa_pairs", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    step = args.step

    if step in ("qa", "all"):
        step_qa(output_dir)

    if step in ("filter", "all"):
        input_path = args.input or str(output_dir / "qa_pair.json")
        step_filter(input_path, output_dir)

    if step in ("dataset", "all"):
        input_path = args.input or str(output_dir / "qa_pair_filtered.json")
        step_dataset(input_path, output_dir)

    logger.info("数据生成完成。")


if __name__ == "__main__":
    main()
