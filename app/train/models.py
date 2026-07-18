"""BERT 分类模型 — Intent (RoBERTa-wwm-ext) classifier.

Ported from CarVoice_Agent/train/models/。使用自定义 BERT 实现。
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

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        return self.fc(self.dropout(pooled))
