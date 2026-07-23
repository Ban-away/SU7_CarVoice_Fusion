#!/usr/bin/env python
"""Self-contained GRPO training — loads SFT adapter, trains with GRPOTrainer."""
import json, os, sys, torch
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path.cwd()))

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from app.rl.reward_model import reward_fn

# 1. Minimal GRPO training data (inline, no file deps)
samples = [
    {"prompt": "小米SU7续航多少", "answer": "CLTC续航约700km"},
    {"prompt": "如何启动语音助手", "answer": "说'你好小爱'唤醒"},
    {"prompt": "SU7充电需要多久", "answer": "快充30分钟可达80%"},
]
data = Dataset.from_list([{"prompt": s["prompt"], "completion": s["answer"]} for s in samples * 30])
print(f"GRPO samples: {len(data)}")

# 2. Load base model + merge SFT adapter
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained("models/Qwen3-8B",
    torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-8B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Merge SFT adapter
sft_path = "models/qwen3_lora_sft"
if os.path.exists(sft_path):
    print(f"Merging SFT adapter: {sft_path}")
    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()

# 3. Apply new LoRA for GRPO
print("Applying GRPO LoRA...")
peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. Train with GRPO (limited steps)
print("Starting GRPO training...")
from trl import GRPOConfig
trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(
        output_dir="models/qwen3_lora_grpo_test",
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
        num_train_epochs=1, max_steps=5,
        learning_rate=2e-5, fp16=True,
        logging_steps=2, save_steps=999, report_to="none",
        num_generations=2, max_completion_length=64,
    ),
    train_dataset=data,
    reward_funcs=[reward_fn],
    processing_class=tokenizer,
)
trainer.train()

# 5. Save
out = "models/qwen3_lora_grpo"
os.makedirs(out, exist_ok=True)
model.save_pretrained(out); tokenizer.save_pretrained(out)
print(f"Saved to {out}")
