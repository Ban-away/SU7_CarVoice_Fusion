#!/usr/bin/env python
"""拒识模型训练 — python scripts/train_reject.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.train.run import run_training
if __name__ == "__main__":
    run_training("bert_tiny", "reject")
