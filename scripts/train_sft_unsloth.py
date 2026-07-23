#!/usr/bin/env python
"""Unsloth Qwen3-8B SFT — local model, FP16, limited steps."""
import os, sys, json
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path.cwd()))

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Load training data
with open("data/training/summary/train.json") as f:
    data = json.load(f)
train_data = Dataset.from_list(data[:3000])
print(f"Training samples: {len(train_data)}")

# 2. Load model from local path (FP16, skip 4-bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/Qwen3-8B",
    max_seq_length=1024,
    load_in_4bit=False,
    dtype=torch.float16,
)
# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model, r=8, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

# 3. Format data
SYSTEM_PROMPT = "你是小米SU7用户手册问答专家，请根据参考文档准确简洁地回答用户问题。"

def fmt(examples):
    texts = []
    for ins, out in zip(examples["instruction"], examples["output"]):
        texts.append(tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ins or out[:50]},
            {"role": "assistant", "content": out},
        ], tokenize=False))
    return {"text": texts}

train_data = train_data.map(fmt, batched=True)
print(f"Formatted {len(train_data)} samples")

# 4. Train (limited steps)
out_dir = "models/qwen3_lora_sft_unsloth"
os.makedirs(out_dir, exist_ok=True)
trainer = SFTTrainer(
    model=model, processing_class=tokenizer, train_dataset=train_data,
    dataset_text_field="text",
    args=TrainingArguments(
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        max_steps=15, learning_rate=2e-4, fp16=True,
        logging_steps=5, save_strategy="no", output_dir=out_dir, report_to="none",
    ),
)
print("Starting Unsloth training...")
trainer.train()
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print(f"Saved to {out_dir}")
