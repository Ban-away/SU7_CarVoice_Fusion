#!/usr/bin/env python
"""
三分类模型训练：Task / FAQ / Chitchat。

训练数据自动构建：
  1. 从 intent/train.txt 抽样 → Task 样本（已有31w条技能标注）
  2. 从 FAQ 关键词 + reject/train.txt 提取 → FAQ 样本
  3. 从 raw_general_chats.txt + chats.txt 提取 → Chitchat 样本

用法:
  python scripts/train_3class.py              # 训练
  python scripts/train_3class.py --predict "怎么打开空调"  # 单条预测
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train_3class")

# ── 三分类标签 ──────────────────────────────────────────────────
LABELS = ["Task", "FAQ", "Chitchat"]

# FAQ 标记词（用于从 reject 数据中筛选 FAQ 样本）
FAQ_MARKERS = [
    "怎么", "如何", "是什么", "什么是", "为什么", "能不能", "可以",
    "有没有", "是否", "支持", "说明书", "手册", "操作", "方法",
    "续航", "充电", "胎压", "故障", "保养", "设置", "调整",
    "参数", "规格", "什么意思", "亮了", "显示",
]

# Chitchat 标记词（用于从通用对话中筛选 Chitchat 样本）
CHITCHAT_MARKERS = [
    "天气", "新闻", "你好", "谢谢", "再见", "你是谁", "笑话",
    "诗", "歌", "电影", "故事", "推荐", "介绍", "翻译",
    "多少", "什么", "哪", "谁", "什么时候", "怎么", "为什么",
]


def build_3class_dataset(
    output_dir: str = "data/training/3class",
    samples_per_class: int = 5000,
) -> None:
    """从现有数据构建三分类训练集。"""
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Task 样本：从 intent/train.txt 抽样 ──
    intent_path = Path("data/training/intent/train.txt")
    task_samples = []
    if intent_path.exists():
        with open(intent_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rsplit("\t", 1)
                text = parts[0] if len(parts) >= 2 else line.strip()
                if len(text) >= 3:
                    task_samples.append((text, "Task"))
    random.shuffle(task_samples)
    task_samples = task_samples[:samples_per_class]
    logger.info("Task samples: %d", len(task_samples))

    # ── FAQ 样本：从 reject/train.txt 筛选车辆相关问题 ──
    reject_path = Path("data/training/reject/train.txt")
    faq_samples = []
    if reject_path.exists():
        with open(reject_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rsplit("\t", 1)
                text = parts[0] if len(parts) >= 2 else line.strip()
                if any(m in text for m in FAQ_MARKERS):
                    faq_samples.append((text, "FAQ"))
    random.shuffle(faq_samples)
    faq_samples = faq_samples[:samples_per_class]
    logger.info("FAQ samples: %d", len(faq_samples))

    # ── Chitchat 样本：从闲聊数据 + 通用问句 ──
    chat_paths = [
        "data/training/chats.txt",
        "data/training/raw_general_chats.txt",
    ]
    chitchat_samples = []
    for path_str in chat_paths:
        path = Path(path_str)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text and len(text) >= 2 and not any(m in text for m in FAQ_MARKERS):
                        chitchat_samples.append((text, "Chitchat"))
    # 也从未标记为 FAQ 的 reject 样本中补充
    if reject_path.exists():
        with open(reject_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rsplit("\t", 1)
                text = parts[0] if len(parts) >= 2 else line.strip()
                if text and not any(m in text for m in FAQ_MARKERS):
                    if any(m in text for m in CHITCHAT_MARKERS):
                        chitchat_samples.append((text, "Chitchat"))
    random.shuffle(chitchat_samples)
    chitchat_samples = chitchat_samples[:samples_per_class]
    logger.info("Chitchat samples: %d", len(chitchat_samples))

    # ── 合并 + 写文件 ──
    all_samples = task_samples + faq_samples + chitchat_samples
    random.shuffle(all_samples)

    # 8:1:1 分割
    n = len(all_samples)
    train_n = int(n * 0.8)
    dev_n = int(n * 0.1)

    for split, data in [("train", all_samples[:train_n]),
                         ("dev", all_samples[train_n:train_n + dev_n]),
                         ("test", all_samples[train_n + dev_n:])]:
        with open(out / f"{split}.txt", "w", encoding="utf-8") as f:
            for text, label in data:
                f.write(f"{text}\t{LABELS.index(label)}\n")

    # 写 class.txt
    with open(out / "class.txt", "w", encoding="utf-8") as f:
        for label in LABELS:
            f.write(f"{label}\n")

    logger.info("Dataset built: %s (train=%d dev=%d test=%d)", output_dir, train_n, dev_n, n - train_n - dev_n)

    # 分布统计
    for split_name in ("train", "dev", "test"):
        path = out / f"{split_name}.txt"
        counts = {"Task": 0, "FAQ": 0, "Chitchat": 0}
        with open(path, encoding="utf-8") as f:
            for line in f:
                idx = int(line.strip().rsplit("\t", 1)[1])
                counts[LABELS[idx]] += 1
        logger.info("  %s: Task=%d FAQ=%d Chitchat=%d", split_name, counts["Task"], counts["FAQ"], counts["Chitchat"])


def train_3class_model(data_dir: str = "data/training/3class") -> None:
    """用 BERT 训练三分类模型。"""
    from app.train.core import BertModel, BertTokenizer
    from app.train.train_eval import train, test
    from app.train.data_helper import build_dataset, build_iterator

    import torch
    import numpy as np

    np.random.seed(1)
    torch.manual_seed(1)

    # 加载数据
    train_data, dev_data, test_data = build_dataset(_make_config(data_dir, "train"))
    train_iter = build_iterator(train_data, _make_config(data_dir, "train"))
    dev_iter = build_iterator(dev_data, _make_config(data_dir, "dev"))
    test_iter = build_iterator(test_data, _make_config(data_dir, "test"))

    # 训练
    model = _build_bert_model(3).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    config = _make_config(data_dir, "train")
    train(config, model, train_iter, dev_iter, test_iter)


def predict_3class(text: str, model_path: str = "models/saved/3class/bert.ckpt") -> str:
    """单条预测。"""
    import torch
    from app.train.core import BertTokenizer

    model = _build_bert_model(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("models/chinese_roberta_wwm_ext")
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens[:30] + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(ids) + [0] * (32 - len(ids))
    ids = ids + [0] * (32 - len(ids))

    with torch.no_grad():
        logits = model((torch.tensor([ids]).to(device),
                         torch.tensor([len(tokens)]).to(device),
                         torch.tensor([mask]).to(device)))
    idx = torch.argmax(logits, dim=1).item()
    return LABELS[idx]


def _build_bert_model(num_classes: int):
    import torch.nn as nn
    from app.train.core import BertModel

    class ThreeClassModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained("models/chinese_roberta_wwm_ext")
            for p in self.bert.parameters():
                p.requires_grad = True
            self.fc = nn.Linear(768, num_classes)

        def forward(self, x):
            _, pooled = self.bert(x[0], attention_mask=x[2], output_all_encoded_layers=False)
            return self.fc(pooled)
    return ThreeClassModel()


class _Config:
    def __init__(self, data_dir, split):
        self.dataset = "3class"
        self.train_path = f"{data_dir}/train.txt"
        self.dev_path = f"{data_dir}/dev.txt"
        self.test_path = f"{data_dir}/test.txt"
        self.class_list = LABELS
        self.save_path = f"models/saved/3class/bert.ckpt"
        self.device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
        self.num_classes = 3
        self.num_epochs = 3
        self.batch_size = 64
        self.pad_size = 32
        self.learning_rate = 2e-5
        self.bert_path = "models/chinese_roberta_wwm_ext"
        self.require_improvement = 1000


def _make_config(data_dir, split):
    return _Config(data_dir, split)


def main():
    parser = argparse.ArgumentParser(description="3-class classifier training")
    parser.add_argument("--build-data", action="store_true", help="Build training dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--predict", help="Predict a single query")
    args = parser.parse_args()

    if args.build_data:
        build_3class_dataset()
    elif args.predict:
        result = predict_3class(args.predict)
        print(f"{args.predict} -> {result}")
    elif args.train:
        build_3class_dataset()
        train_3class_model()
    else:
        # Default: build + train
        build_3class_dataset()
        train_3class_model()


if __name__ == "__main__":
    main()
