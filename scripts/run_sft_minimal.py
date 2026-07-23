#!/usr/bin/env python
"""Minimal Qwen3-8B QLoRA SFT — transformers + peft + bitsandbytes, no unsloth."""
import json, os, torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

os.chdir(Path(__file__).resolve().parent.parent)

# 1. Load data
with open("data/training/summary/train.json") as f:
    raw = json.load(f)
data = Dataset.from_list(raw[:500] * 20)  # 100 samples → repeat for meaningful batches
print(f"Training samples: {len(data)}")

# 2. Load model FP16 (no bitsandbytes — 48GB VRAM is enough for 8B model)
model = AutoModelForCausalLM.from_pretrained("models/Qwen3-8B",
    torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-8B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 3. LoRA (no kbit prepare needed for FP16)
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 4. Format
def fmt(ex):
    ins = ex.get("instruction","") or ex.get("input","")
    out = ex.get("output","")
    return {"text": f"<|im_start|>system\n你是小米SU7手册问答助手。<|im_end|>\n<|im_start|>user\n{ins}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"}
data = data.map(fmt)

# 5. Train
class TimeCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= 50:  # 3 min = ~50 steps
            control.should_training_stop = True

trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=data,
    args=TrainingArguments(
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=1, learning_rate=2e-4, fp16=True,
        logging_steps=5, save_steps=999999,
        output_dir="models/sft_test", report_to="none"),
    callbacks=[TimeCallback()])
trainer.train()

# 6. Save
out = "models/qwen3_lora_sft"
os.makedirs(out, exist_ok=True)
model.save_pretrained(out); tokenizer.save_pretrained(out)
print(f"Saved to {out}")
