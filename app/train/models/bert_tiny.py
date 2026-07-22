"""BERT-Tiny (3-layer) reject classifier model.

Ported from CarVoice_Agent/train/models/bert_tiny/.
Uses the custom BERT core with a 3-layer config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from app.train.core import BertModel, BertTokenizer


class Model(nn.Module):
    """3-layer BERT Tiny + Linear → binary reject classification."""

    def __init__(self, model_path: str = "models/roberta_tiny_clue", dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        # Binary classification: reject (0) vs non-reject (1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, pooled = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        return self.fc(self.dropout(pooled))
