#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档解析质量评估脚本。

对应 XIAOMI_SU7_RAG/evaluate_parse_quality.py 的功能：
  评估 PDF 解析质量，输出文本保留率、切分质量等指标。

用法:
  python scripts/evaluate_parse_quality.py
  python scripts/evaluate_parse_quality.py --pdf data/knowledge/Xiaomi_SU7_Manual.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def _evaluate_text_quality(pages: list[dict]) -> dict:
    """评估文本解析质量。"""
    if not pages:
        return {"text_retention": 0.0, "blank_ratio": 0.0, "anomaly_ratio": 0.0}

    total_chars = 0
    blank_lines = 0
    total_lines = 0
    anomaly_chars = 0
    has_title = 0
    has_list = 0
    total_pages = len(pages)

    for page in pages:
        text = page.get("text", "")
        total_chars += len(text)

        lines = text.split("\n")
        for line in lines:
            total_lines += 1
            if not line.strip():
                blank_lines += 1
            # Count anomalous characters
            anomaly_chars += sum(1 for c in line if ord(c) < 32 and c not in "\n\r\t")
            # Check for title patterns
            if line.strip().startswith("#") or line.strip().startswith("第") or "章" in line[:20]:
                has_title += 1
            # Check for list patterns
            if line.strip().startswith(("•", "-", "·", "1.", "2.", "3.", "（")):
                has_list += 1

    text_retention = min(100.0, total_chars / max(len(pages) * 500, 1) * 100)  # estimate
    blank_ratio = (blank_lines / max(total_lines, 1)) * 100
    anomaly_ratio = (anomaly_chars / max(total_chars, 1)) * 100
    title_retention = min(100.0, has_title / max(total_pages, 1) * 100)
    list_retention = min(100.0, has_list / max(total_pages, 1) * 100)

    # Composite score
    composite = (
        min(100, text_retention) * 0.4
        + max(0, 100 - blank_ratio * 2) * 0.3
        + max(0, 100 - anomaly_ratio * 10) * 0.3
    )

    return {
        "text_retention": round(text_retention, 2),
        "blank_ratio": round(blank_ratio, 2),
        "title_retention": round(title_retention, 2),
        "list_retention": round(list_retention, 2),
        "anomaly_ratio": round(anomaly_ratio, 2),
        "composite": round(composite, 2),
    }


def _evaluate_chunk_quality(chunks: list[str]) -> dict:
    """评估切分质量。"""
    if not chunks:
        return {"chunk_count": 0, "avg_length": 0, "std_length": 0, "quality_score": 0.0}

    lengths = [len(c) for c in chunks]
    avg_len = sum(lengths) / len(lengths)
    variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
    std_len = variance ** 0.5

    # Quality score: target around 500-2000 chars per chunk
    too_short = sum(1 for l in lengths if l < 100)
    too_long = sum(1 for l in lengths if l > 3000)
    good = len(lengths) - too_short - too_long

    quality = (good / len(lengths) * 0.6 + max(0, 1 - std_len / avg_len) * 0.4) * 100

    return {
        "chunk_count": len(chunks),
        "avg_length": round(avg_len, 1),
        "std_length": round(std_len, 1),
        "quality_score": round(quality, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="文档解析与切分质量评估")
    parser.add_argument(
        "--pdf", default="data/knowledge/Xiaomi_SU7_Manual.pdf",
        help="PDF 文件路径",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("PDF 文件不存在: %s", pdf_path)
        sys.exit(1)

    # Parse PDF
    logger.info("解析 PDF: %s", pdf_path)
    try:
        from app.knowledge.parser.pdf_parser import PDFParser
        parser_obj = PDFParser()
        pages = parser_obj.parse(str(pdf_path))
        logger.info("PDF 解析完成: %d 页", len(pages))
    except Exception as exc:
        logger.error("PDF 解析失败: %s", exc)
        sys.exit(1)

    # Evaluate text quality
    text_quality = _evaluate_text_quality(pages)

    # Chunk
    logger.info("执行语义切分...")
    try:
        from app.knowledge.chunker import SemanticChunker
        texts = [p.get("text", "") for p in pages if p.get("text", "").strip()]
        full_text = "\n".join(texts)
        chunker = SemanticChunker(chunk_size=1024, chunk_overlap=100)
        chunk_docs = chunker.chunk_text(full_text, source=str(pdf_path))
        chunks = [doc.content for doc in chunk_docs]
        logger.info("切分完成: %d 个块", len(chunks))
    except Exception as exc:
        logger.error("切分失败: %s", exc)
        chunks = []

    chunk_quality = _evaluate_chunk_quality(chunks)

    # Print report
    print()
    print("=" * 60)
    print("  📄 文档解析质量评估报告")
    print("=" * 60)
    print(f"  文件: {pdf_path.name}")
    print(f"  总页数: {len(pages)}")
    print()
    print("  【文本解析质量】")
    print(f"    ├─ 文本保留率: {text_quality['text_retention']:.2f}%")
    print(f"    ├─ 空白行比例: {text_quality['blank_ratio']:.2f}%")
    print(f"    ├─ 标题保留率: {text_quality['title_retention']:.2f}%")
    print(f"    ├─ 列表保留率: {text_quality['list_retention']:.2f}%")
    print(f"    ├─ 异常字符率: {text_quality['anomaly_ratio']:.2f}%")
    print(f"    └─ 综合评分:   {text_quality['composite']:.2f}%")
    print()
    print("  【文档切分质量】")
    print(f"    ├─ 切分数量:   {chunk_quality['chunk_count']}")
    print(f"    ├─ 平均长度:   {chunk_quality['avg_length']} 字符")
    print(f"    ├─ 长度标准差: {chunk_quality['std_length']}")
    print(f"    └─ 切分质量:   {chunk_quality['quality_score']:.2f}%")
    print("=" * 60)

    final_score = (
        text_quality["composite"] * 0.7
        + chunk_quality["quality_score"] * 0.3
    )
    print(f"  📊 最终解析准确率: {final_score:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
