#!/usr/bin/env python
"""Unsloth 加速 Qwen3-8B QLoRA SFT 微调。

适用: 单卡 ≥ 12GB (RTX 3060/4060)
速度: 比 LLaMA-Factory QLoRA 快 2-5x，显存降 50%+

用法:
  pip install unsloth
  python scripts/train_sft_unsloth.py

输出: models/qwen3_lora_sft_unsloth/
"""

from __future__ import annotations

import os, sys, json, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    from unsloth import FastLanguageModel
    from datasets import Dataset
    import torch

    # 1. 加载训练数据
    train_path = Path("data/training/summary/train.json")
    if not train_path.exists():
        logger.error("训练数据不存在: %s。请先运行 scripts/generate_data.py", train_path)
        return

    with open(train_path, encoding="utf-8") as f:
        data = json.load(f)

    # 转为 HuggingFace Dataset
    train_data = Dataset.from_list(data[:5000])  # 用前5000条示例

    # 2. 加载模型 (4-bit QLoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-8B",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # 3. 添加 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,            # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
    )

    # 4. 格式化数据
    SYSTEM_PROMPT = "你是小米 SU7 用户手册问答专家，请根据参考文档准确简洁地回答用户问题。"

    def format_fn(examples):
        texts = []
        for ins, out in zip(examples["instruction"], examples["output"]):
            texts.append(tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ins or out[:50]},
                {"role": "assistant", "content": out},
            ], tokenize=False))
        return {"text": texts}

    train_data = train_data.map(format_fn, batched=True)

    # 5. 训练
    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = "models/qwen3_lora_sft_unsloth"
    os.makedirs(output_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            max_steps=500,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_steps=100,
            output_dir=output_dir,
        ),
    )

    logger.info("Starting Unsloth training...")
    trainer.train()

    # 6. 保存
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
