"""GRPO 强化学习训练入口。

Ported from XIAOMI_SU7_RAG/src/rl/train_grpo.py。
整合数据构建 → 格式转换 → SFT warm-up → GRPO 训练 → 导出全流程。

用法:
  python -m app.rl.train_grpo --stage all      # 全流程
  python -m app.rl.train_grpo --stage sft      # 仅 SFT warm-up
  python -m app.rl.train_grpo --stage grpo     # 仅 GRPO 训练
  python -m app.rl.train_grpo --stage export   # 仅导出
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

SFT_CONFIG = "configs/original_sft.yaml"
GRPO_CONFIG = "configs/original_grpo.yaml"


def run_sft():
    logger.info("Starting SFT warm-up...")
    cmd = ["llamafactory-cli", "train", SFT_CONFIG]
    subprocess.run(cmd, check=True)


def run_grpo():
    logger.info("Starting GRPO training via TRL GRPOTrainer...")
    try:
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from app.rl.reward_model import reward_fn
        import torch, json

        # Load base model
        base_model = os.getenv("GRPO_BASE_MODEL", "models/Qwen3-8B")
        sft_adapter = os.getenv("GRPO_SFT_ADAPTER", "models/qwen3_lora_sft")

        logger.info(f"Loading base: {base_model}, adapter: {sft_adapter}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load SFT adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()

        # Apply LoRA for GRPO
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
        model = get_peft_model(model, lora_config)

        # Load training data
        data_path = "data/training/rl/combined_trajectories_grpo.jsonl"
        if not os.path.exists(data_path):
            logger.warning(f"GRPO data not found: {data_path} — run format_converter first")
            return

        logger.info("GRPO training started...")
        training_args = GRPOConfig(
            output_dir="models/qwen3_lora_grpo",
            num_train_epochs=1,
            max_steps=5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-6,
            logging_steps=1,
            save_steps=999,
            fp16=True,
            report_to="none",
            num_generations=4,
        )
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=data_path,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model("models/qwen3_lora_grpo_final")
        logger.info("GRPO training complete")

    except ImportError as e:
        logger.error("Missing dependencies: %s — install: pip install trl peft bitsandbytes", e)
    except Exception:
        logger.exception("GRPO training failed")


def run_export():
    logger.info("Exporting merged model...")
    cmd = ["llamafactory-cli", "export", SFT_CONFIG]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="GRPO RL training pipeline")
    parser.add_argument("--stage", choices=["data", "sft", "grpo", "export", "all"], default="all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    stage = args.stage
    if stage in ("data", "all"):
        from app.rl.data_builder import TrajectoryBuilder
        builder = TrajectoryBuilder()
        logger.info("Run data generation separately: python -m app.rl.data_builder")

    if stage in ("sft", "all"):
        run_sft()

    if stage in ("grpo", "all"):
        run_grpo()

    if stage in ("export", "all"):
        run_export()


if __name__ == "__main__":
    main()
