"""VeRL GRPO 强化学习训练 — 全量 21K+ 样本，多卡并行。

适用: 多卡 ≥ 2×24GB (RTX 3090/4090) 或 A100
对比 TRL: 样本量 80→21K+, 吞吐 3-5x

前置条件:
  pip install verl
  python app/rl/data_builder.py          # 生成轨迹
  python app/rl/format_converter.py       # 转换格式
  python app/rl/rebalance_sft_data.py    # 再平衡

用法:
  # 单机 4 卡
  python app/rl/train_grpo_verl.py --n_gpus 4

  # 单机 2 卡
  python app/rl/train_grpo_verl.py --n_gpus 2

  # 指定 SFT 模型
  python app/rl/train_grpo_verl.py --sft_model models/qwen3_lora_sft
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)


def train_verl_grpo(
    data_path: str = "data/training/rl/trajectories_grpo.jsonl",
    sft_model: str = "models/qwen3_lora_sft",
    base_model: str = "Qwen/Qwen3-8B",
    output_dir: str = "models/qwen3_lora_verl_grpo",
    n_gpus: int = 4,
    max_prompt_length: int = 2048,
    max_response_length: int = 1024,
    learning_rate: float = 1e-6,
    total_epochs: int = 3,
    micro_batch_size: int = 4,
) -> None:
    """使用 VeRL 运行 GRPO 训练。

    VeRL 通过 HybridEngine 实现训练时模型并行 + 推理时 vLLM，
    零开销切换，显存效率远高于 TRL。

    Args:
        data_path: GRPO JSONL 训练数据路径（由 format_converter.py 生成）。
        sft_model: SFT 微调后的模型路径（作为 GRPO 初始权重）。
        base_model: 基座模型名称或路径。
        output_dir: 训练输出目录。
        n_gpus: GPU 数量。
    """
    if not Path(data_path).exists():
        logger.error("GRPO 数据不存在: %s", data_path)
        logger.error("请先运行: python app/rl/data_builder.py && python app/rl/format_converter.py")
        return

    # VeRL CLI 参数
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"data.train_files={data_path}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        f"actor_rollout_ref.model.path={base_model}",
        f"actor_rollout_ref.model.lora_adapter_path={sft_model}",
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={micro_batch_size}",
        f"actor_rollout_ref.actor.ppo_epochs={total_epochs}",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.n={n_gpus}",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.top_p=0.9",
        "reward_model.reward_manager=naive",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.project_name=su7_carvoice_grpo",
        f"trainer.experiment_name=verl_grpo",
        f"trainer.default_local_dir={output_dir}",
        "trainer.logger=['console']",
    ]

    logger.info("Starting VeRL GRPO training with %d GPUs...", n_gpus)
    logger.info("Command: verl.trainer.main_ppo %s", " ".join(cmd[3:]))

    import subprocess
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="VeRL GRPO 训练")
    parser.add_argument("--data", default="data/training/rl/trajectories_grpo.jsonl")
    parser.add_argument("--sft-model", default="models/qwen3_lora_sft")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", default="models/qwen3_lora_verl_grpo")
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--max-response-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    train_verl_grpo(
        data_path=args.data,
        sft_model=args.sft_model,
        base_model=args.base_model,
        output_dir=args.output,
        n_gpus=args.n_gpus,
        max_prompt_length=args.max_prompt_len,
        max_response_length=args.max_response_len,
        learning_rate=args.lr,
        total_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
