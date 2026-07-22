#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线知识库索引构建脚本。

对应 XIAOMI_SU7_RAG/build_index.py 的完整流程：
  PDF 解析 → 文本清洗 → 语义分块 → BM25 索引 → FAISS/Milvus 索引

用法:
  # 默认：构建 BM25 索引
  python scripts/build_index.py

  # 构建全部索引（含 Milvus，需 GPU）
  python scripts/build_index.py --backend all

  # 只构建 BM25
  python scripts/build_index.py --backend bm25

  # 指定 PDF 路径
  python scripts/build_index.py --pdf data/knowledge/Xiaomi_SU7_Manual.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("build_index")


def build_bm25(chunks: list[str], output_dir: Path) -> None:
    from app.knowledge.retriever.bm25 import BM25Retriever

    logger.info("构建 BM25 索引 (%d 条文档)...", len(chunks))
    retriever = BM25Retriever(chunks)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "bm25retriever.pkl", "wb") as f:
        pickle.dump(retriever, f)
    logger.info("BM25 索引已保存到 %s", output_dir / "bm25retriever.pkl")


def build_faiss(chunks: list[str], output_dir: Path) -> None:
    from app.knowledge.retriever.faiss import FAISSRetriever

    logger.info("构建 FAISS 索引 (%d 条文档)...", len(chunks))
    retriever = FAISSRetriever(chunks)
    retriever.save(str(output_dir / "faiss.db"))
    logger.info("FAISS 索引已保存到 %s", output_dir / "faiss.db")


def build_milvus(chunks: list[str], output_dir: Path) -> None:
    from app.knowledge.retriever.milvus import MilvusRetriever

    logger.info("构建 Milvus 混合索引 (%d 条文档, BGE+SPLADE)...", len(chunks))
    retriever = MilvusRetriever(chunks)
    logger.info("Milvus 索引已保存到 %s", output_dir / "milvus.db")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建知识库索引")
    parser.add_argument("--pdf", default="data/knowledge/Xiaomi_SU7_Manual.pdf", help="PDF 路径")
    parser.add_argument("--docs-json", default="data/knowledge/su7_docs.json", help="JSON 文档路径（备选）")
    parser.add_argument(
        "--backend", choices=["bm25", "faiss", "milvus", "all"], default="bm25",
        help="索引后端 (默认 bm25)",
    )
    parser.add_argument("--output", default="data/knowledge/saved_index", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    chunks: list[str] = []

    # 1. 尝试解析 PDF
    pdf_path = Path(args.pdf)
    if pdf_path.exists():
        logger.info("解析 PDF: %s", pdf_path)
        from app.knowledge.parser.pdf_parser import PDFParser
        from app.knowledge.chunker import SemanticChunker

        parser_obj = PDFParser()
        pages = parser_obj.parse(str(pdf_path))
        texts = [p.get("text", "") for p in pages if p.get("text", "").strip()]
        logger.info("PDF 解析完成: %d 页有文本", len(texts))

        chunker = SemanticChunker(chunk_size=1024, chunk_overlap=100)
        full_text = "\n".join(texts)
        chunk_docs = chunker.chunk_text(full_text, source=str(pdf_path))
        chunks = [doc.content for doc in chunk_docs]
        logger.info("语义分块完成: %d 个块", len(chunks))

    # 2. 备选：从 JSON 加载
    if not chunks:
        json_path = Path(args.docs_json)
        if json_path.exists():
            logger.info("从 JSON 加载文档: %s", json_path)
            with open(json_path, encoding="utf-8") as f:
                docs = json.load(f)
            chunks = [d.get("content", "") for d in docs if d.get("content", "").strip()]
            logger.info("JSON 加载完成: %d 条文档", len(chunks))

    if not chunks:
        logger.error("没有可用文档，请检查 PDF 或 JSON 路径")
        sys.exit(1)

    # 3. 构建索引
    backend = args.backend
    if backend in ("bm25", "all"):
        build_bm25(chunks, output_dir)
    if backend in ("faiss", "all"):
        build_faiss(chunks, output_dir)
    if backend in ("milvus", "all"):
        build_milvus(chunks, output_dir)

    logger.info("索引构建完成。")


if __name__ == "__main__":
    main()
