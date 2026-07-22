#!/usr/bin/env python
"""BERT 分类模型训练入口。

Ported from CarVoice_Agent/train/run.py。

用法:
  python -m app.train.run --model bert --data intent      # 意图分类 (RoBERTa-wwm-ext)
  python -m app.train.run --model bert_tiny --data reject  # 拒识分类 (3-layer BERT Tiny)
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from app.train.data_loader import build_dataloaders
from app.train.models.bert import Model as BertClassifier
from app.train.models.bert_tiny import Model as BertTinyClassifier

logger = logging.getLogger(__name__)

# Data paths relative to project root
DATA_BASE = "data/training"

# Model configs (matching CarVoice_Agent original)
MODEL_CONFIGS = {
    "bert": {
        "model_path": "models/chinese_roberta_wwm_ext",
        "num_classes_fn": lambda ds: len(_load_class_list(f"{DATA_BASE}/{ds}/class.txt")),
        "num_epochs": 3,
        "batch_size": 128,
        "max_len": 32,
        "lr": 5e-5,
    },
    "bert_tiny": {
        "model_path": "models/roberta_tiny_clue",
        "num_classes_fn": lambda ds: 2,
        "num_epochs": 3,
        "batch_size": 128,
        "max_len": 32,
        "lr": 5e-5,
    },
}


def _load_class_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def _eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / len(dataloader), acc, p, r, f1


def run_training(model_name: str, dataset_name: str) -> None:
    cfg = MODEL_CONFIGS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {model_name} on {dataset_name}, device={device}")

    # Data paths
    train_path = f"{DATA_BASE}/{dataset_name}/train.txt"
    dev_path   = f"{DATA_BASE}/{dataset_name}/dev.txt"
    test_path  = f"{DATA_BASE}/{dataset_name}/test.txt"

    # Build dataloaders
    train_loader, dev_loader, test_loader = build_dataloaders(
        train_path, dev_path, test_path,
        tokenizer_name=cfg["model_path"],
        batch_size=cfg["batch_size"], max_len=cfg["max_len"],
    )

    # Build model
    num_classes = cfg["num_classes_fn"](dataset_name)
    if model_name == "bert":
        model = BertClassifier(num_classes, model_path=cfg["model_path"]).to(device)
    else:
        model = BertTinyClassifier(model_path=cfg["model_path"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])

    # Save path
    save_dir = f"models/saved/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{model_name}.ckpt"

    best_dev_acc = 0.0
    t0 = time.time()

    for epoch in range(cfg["num_epochs"]):
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_acc, dev_p, dev_r, dev_f1 = _eval_epoch(model, dev_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{cfg['num_epochs']}: "
                     f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                     f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} "
                     f"P={dev_p:.4f} R={dev_r:.4f} F1={dev_f1:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

    elapsed = time.time() - t0
    logger.info(f"Training finished in {elapsed:.1f}s. Best dev acc: {best_dev_acc:.4f}")

    # Final test evaluation
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc, test_p, test_r, test_f1 = _eval_epoch(model, test_loader, criterion, device)
    logger.info(f"Test: loss={test_loss:.4f} acc={test_acc:.4f} P={test_p:.4f} R={test_r:.4f} F1={test_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="BERT classifier training")
    parser.add_argument("--model", required=True, choices=["bert", "bert_tiny"])
    parser.add_argument("--data", required=True, choices=["intent", "reject"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    run_training(args.model, args.data)


if __name__ == "__main__":
    main()
