#!/usr/bin/env python
"""意图分类模型训练 — python scripts/train_intent.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.train.run import run_training
if __name__ == "__main__":
    run_training("bert", "intent")
