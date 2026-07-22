"""RoBERTa-wwm-ext intent classifier model.

Ported from CarVoice_Agent/train/models/, adapted to match run.py calling convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from app.train.core import BertModel, BertTokenizer


class Model(nn.Module):
    """RoBERTa-wwm-ext + Linear → intent classification."""

    def __init__(self, num_classes: int, model_path: str = "models/chinese_roberta_wwm_ext", dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, pooled = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        return self.fc(self.dropout(pooled))
