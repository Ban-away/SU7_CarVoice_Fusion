#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拒识模型精度评测脚本。

对应 CarVoice_Agent/test/reject_client.py 的功能：
  使用 test.txt + class.txt 评测 BERT-Tiny 拒识模型的 Accuracy/Precision/Recall/F1。

用法:
  python scripts/reject_benchmark.py
  python scripts/reject_benchmark.py --data data/training/reject/test.txt --ckpt models/saved/reject/bert_tiny.ckpt
  python scripts/reject_benchmark.py --verbose  # 逐条打印错误
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


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
                    label = int(parts[-1])
                    text = "\t".join(parts[:-1])
                    samples.append((text, label))
                except ValueError:
                    continue
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT-Tiny 拒识模型精度评测")
    parser.add_argument(
        "--data", default="data/training/reject/test.txt",
        help="测试数据路径",
    )
    parser.add_argument(
        "--ckpt", default="models/saved/reject/bert_tiny.ckpt",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--model-path", default="models/roberta_tiny_clue",
        help="预训练 BERT-Tiny 路径",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="批量大小")
    parser.add_argument("--max-len", type=int, default=32, help="最大序列长度")
    parser.add_argument("--verbose", action="store_true", help="逐条打印错误")
    args = parser.parse_args()

    # Load data
    samples = _load_test_data(args.data)
    logger.info("加载 %d 条测试样本 → %s", len(samples), args.data)

    if not samples:
        logger.error("没有测试样本")
        sys.exit(1)

    # Load model
    from app.train.models.bert_tiny import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model_path=args.model_path).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    logger.info("拒识模型加载完成: device=%s", device)

    # Load tokenizer
    from app.train.core import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # Evaluate
    all_preds, all_labels = [], []
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
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(batch_labels)

        if args.verbose:
            for j, (p, t) in enumerate(zip(preds, batch_labels)):
                if p != t:
                    errors.append(f"[MISS] true={t} pred={p} text='{batch_texts[j][:80]}'")

    elapsed = time.time() - t0

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)

    print()
    print("=" * 60)
    print(f"  拒识模型精度评测")
    print(f"  模型: {args.ckpt}")
    print(f"  测试样本: {len(samples)} 条")
    print(f"  耗时: {elapsed:.1f}s ({len(samples) / elapsed:.1f} samples/s)")
    print("=" * 60)
    print(f"  Accuracy:   {acc * 100:.2f}%")
    print(f"  Precision:  {p * 100:.2f}%")
    print(f"  Recall:     {r * 100:.2f}%")
    print(f"  F1:         {f1 * 100:.2f}%")
    print("=" * 60)

    # Reference from CarVoice_Agent:
    #   Accuracy=89.71%, Precision=89.65%, Recall=89.74%, F1=89.69%

    if errors:
        print(f"\n  错误详情 (前 20 条):")
        for err in errors[:20]:
            print(f"    {err}")
        if len(errors) > 20:
            print(f"    ... 共 {len(errors)} 条错误")


if __name__ == "__main__":
    main()
