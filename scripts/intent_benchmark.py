#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""意图识别模型精度评测脚本。

对应 CarVoice_Agent/test/intent_client.py 的功能：
  使用 test.txt + class.txt 评测 BERT 意图模型的 Top1/Top5 准确率。

用法:
  python scripts/intent_benchmark.py
  python scripts/intent_benchmark.py --top-k 3
  python scripts/intent_benchmark.py --data data/training/intent/test.txt --ckpt models/saved/intent/bert.ckpt
"""

from __future__ import annotations

import os
import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_class_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_test_data(path: str) -> list[tuple[str, int]]:
    """Load test data: each line is 'text\\tlabel'."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    # format: text \t label
                    label = int(parts[-1])
                    text = "\t".join(parts[:-1])
                    samples.append((text, label))
                except ValueError:
                    continue
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT 意图模型精度评测")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K 准确率 (default: 5)")
    parser.add_argument(
        "--data", default="data/training/intent/test.txt",
        help="测试数据路径",
    )
    parser.add_argument(
        "--class-file", default="data/training/intent/class.txt",
        help="类别列表路径",
    )
    parser.add_argument(
        "--ckpt", default="models/saved/intent/bert.ckpt",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--model-path", default="models/chinese_roberta_wwm_ext",
        help="预训练 BERT 路径",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="批量大小")
    parser.add_argument("--max-len", type=int, default=32, help="最大序列长度")
    parser.add_argument("--verbose", action="store_true", help="逐条打印错误")
    args = parser.parse_args()

    # Load data
    class_list = _load_class_list(args.class_file)
    logger.info("加载 %d 个意图类别 → %s", len(class_list), args.class_file)

    samples = _load_test_data(args.data)
    logger.info("加载 %d 条测试样本 → %s", len(samples), args.data)

    if not samples:
        logger.error("没有测试样本，请检查数据路径和格式")
        sys.exit(1)

    # Load model
    from app.train.models.bert import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(len(class_list), model_path=args.model_path).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    logger.info("模型加载完成: device=%s, classes=%d", device, len(class_list))

    # Load tokenizer
    from app.train.core import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # Evaluate
    top1_correct = 0
    topk_correct = 0
    total = 0
    errors = []
    t0 = time.time()

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i + args.batch_size]
        batch_texts = [s[0] for s in batch]
        batch_labels = [s[1] for s in batch]

        # Tokenize
        all_ids, all_masks = [], []
        for text in batch_texts:
            tokens = tokenizer.tokenize(text)
            tokens = (["[CLS]"] + tokens[:args.max_len - 2] + ["[SEP]"])[:args.max_len]
            ids = tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(ids) + [0] * (args.max_len - len(ids))
            ids = ids + [0] * (args.max_len - len(ids))
            all_ids.append(ids)
            all_masks.append(mask)

        with torch.no_grad():
            input_ids = torch.tensor(all_ids).to(device)
            attention_mask = torch.tensor(all_masks).to(device)
            logits = model(input_ids, attention_mask)

        # Top-K accuracy
        _, topk_indices = torch.topk(logits, k=min(args.top_k, len(class_list)), dim=1)
        top1_preds = torch.argmax(logits, dim=1).cpu().tolist()

        for j, (pred1, topk, true_label) in enumerate(zip(top1_preds, topk_indices.cpu().tolist(), batch_labels)):
            total += 1
            if pred1 == true_label:
                top1_correct += 1
            if true_label in topk:
                topk_correct += 1
            elif args.verbose:
                text = batch_texts[j][:60]
                errors.append(
                    f"[TOP{args.top_k}_MISS] true={class_list[true_label] if true_label < len(class_list) else true_label} "
                    f"pred1={class_list[pred1] if pred1 < len(class_list) else pred1} "
                    f"text='{text}'"
                )

    elapsed = time.time() - t0

    # Print results
    print()
    print("=" * 60)
    print(f"  意图识别模型精度评测")
    print(f"  模型: {args.ckpt}")
    print(f"  测试样本: {total} 条")
    print(f"  耗时: {elapsed:.1f}s ({total / elapsed:.1f} samples/s)")
    print("=" * 60)
    print(f"  Top-1 准确率:  {top1_correct / total * 100:.2f}% ({top1_correct}/{total})")
    print(f"  Top-{args.top_k} 准确率: {topk_correct / total * 100:.2f}% ({topk_correct}/{total})")
    print("=" * 60)

    if errors:
        print(f"\n  错误详情 (Top {len(errors)}):")
        for err in errors[:20]:
            print(f"    {err}")
        if len(errors) > 20:
            print(f"    ... 共 {len(errors)} 条错误")

    # Reference from CarVoice_Agent:
    #   Top-1: 85.2%, Top-5: 97.6%


if __name__ == "__main__":
    main()
