"""训练数据加载器。

Ported from CarVoice_Agent/train/data_helper.py。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from app.train.core import BertTokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, filepath: str, tokenizer_name: str, max_len: int = 32):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {filepath}")

        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        self.texts: list[str] = []
        self.labels: list[int] = []
        for line in lines:
            parts = line.rsplit("\t", 1)
            if len(parts) >= 2:
                self.texts.append(parts[0])
                try:
                    self.labels.append(int(parts[1]))
                except ValueError:
                    self.labels.append(0)
            else:
                self.texts.append(line)
                self.labels.append(0)

        logger.info("Loaded %d samples from %s", len(self.texts), filepath)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.texts[idx])
        tokens = (["[CLS]"] + tokens[:self.max_len-2] + ["[SEP]"])[:self.max_len]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(ids) + [0] * (self.max_len - len(ids))
        ids = ids + [0] * (self.max_len - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_dataloaders(
    train_path: str,
    dev_path: str,
    test_path: str,
    tokenizer_name: str,
    batch_size: int = 128,
    max_len: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TextDataset(train_path, tokenizer_name, max_len)
    dev_ds = TextDataset(dev_path, tokenizer_name, max_len)
    test_ds = TextDataset(test_path, tokenizer_name, max_len)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(dev_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
