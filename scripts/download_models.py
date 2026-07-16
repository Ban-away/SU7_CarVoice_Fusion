#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SU7_CarVoice_Fusion 模型下载脚本。

合并了 CarVoice_Agent/download_models.py 与
XIAOMI_SU7_RAG/deploy/download_models.py 的全部模型清单。

用法:
  # 下载核心模型（主流程必需）
  python scripts/download_models.py

  # 下载全部模型（含备选重排器）
  python scripts/download_models.py --preset all

  # 只下载 Agent 模型
  python scripts/download_models.py --preset agent

  # 只下载 RAG 模型
  python scripts/download_models.py --preset rag

  # 指定国内镜像 + Token
  HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --hf-token hf_xxx
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# ── 可选依赖 ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise ImportError(
        "huggingface_hub 未安装，请执行: pip install huggingface_hub"
    )


# ── 模型定义 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    target_rel_path: str
    required: bool = True
    source: str = ""           # "carvoice" | "su7_rag"


# ---- Agent 模型（CarVoice_Agent 训练用） ----
AGENT_MODELS: List[ModelSpec] = [
    # 意图分类模型（RoBERTa-wwm-ext，平衡精度与效率）
    ModelSpec("chinese_roberta_wwm_ext", "hfl/chinese-roberta-wwm-ext",
              "models/chinese_roberta_wwm_ext", source="carvoice"),
    # 拒识模型（3 层 BERT Tiny，极致效率）
    ModelSpec("roberta_tiny_clue", "clue/roberta_chinese_3L312_clue_tiny",
              "models/roberta_tiny_clue", source="carvoice"),
]

# ---- RAG 核心模型（XIAOMI_SU7_RAG 主流程必需） ----
RAG_CORE_MODELS: List[ModelSpec] = [
    # 语义切分
    ModelSpec("m3e-small", "moka-ai/m3e-small",
              "models/moka-ai/m3e-small", source="su7_rag"),
    # Dense 检索
    ModelSpec("bge-large-zh-v1.5", "BAAI/bge-large-zh-v1.5",
              "models/BAAI/bge-large-zh-v1.5", source="su7_rag"),
    # Sparse 检索
    ModelSpec("splade-v2", "naver/splade-cocondenser-ensembledistil",
              "models/naver/splade-cocondenser-ensembledistil", source="su7_rag"),
    # 精排（在线推理 + 离线评估共用）
    ModelSpec("bge-reranker-v2-minicpm-layerwise",
              "BAAI/bge-reranker-v2-minicpm-layerwise",
              "models/BAAI/bge-reranker-v2-minicpm-layerwise", source="su7_rag"),
    # 生成模型基座（微调用）
    ModelSpec("Qwen3-8B", "Qwen/Qwen3-8B",
              "models/Qwen3-8B", source="su7_rag"),
    # 评测相似度
    ModelSpec("text2vec-base-chinese", "shibing624/text2vec-base-chinese",
              "models/text2vec-base-chinese", source="su7_rag"),
]

# ---- RAG 可选模型（备选重排器 / 额外检索） ----
RAG_EXTRA_MODELS: List[ModelSpec] = [
    ModelSpec("bge-m3", "BAAI/bge-m3",
              "models/BAAI/bge-m3", required=False, source="su7_rag"),
    ModelSpec("bce-embedding-base_v1", "maidalun/bce-embedding-base_v1",
              "models/maidalun/bce-embedding-base_v1", required=False, source="su7_rag"),
    ModelSpec("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B",
              "models/Qwen3-Embedding-0.6B", required=False, source="su7_rag"),
    ModelSpec("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3",
              "models/BAAI/bge-reranker-v2-m3", required=False, source="su7_rag"),
    ModelSpec("Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-0.6B",
              "models/Qwen3-Reranker-0.6B", required=False, source="su7_rag"),
    ModelSpec("Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-4B",
              "models/Qwen3-Reranker-4B", required=False, source="su7_rag"),
    ModelSpec("jina-reranker-v2-base-multilingual",
              "jinaai/jina-reranker-v2-base-multilingual",
              "models/jinaai/jina-reranker-v2-base-multilingual",
              required=False, source="su7_rag"),
]

# ── 预设组合 ────────────────────────────────────────────────────────────────

MODEL_PRESETS: Dict[str, List[ModelSpec]] = {
    # 仅 Agent 模型
    "agent": AGENT_MODELS,
    # 仅 RAG 核心模型
    "rag": RAG_CORE_MODELS,
    # 核心模型（Agent + RAG，默认）
    "core": AGENT_MODELS + RAG_CORE_MODELS,
    # 全部模型（含备选重排器）
    "all": AGENT_MODELS + RAG_CORE_MODELS + RAG_EXTRA_MODELS,
}

# ── 未公开下载的产物说明 ────────────────────────────────────────────────────

MANUAL_ARTIFACTS = [
    "train/saved/intent/bert.ckpt          (意图分类模型，需本地训练)",
    "train/saved/reject/bert_tiny.ckpt      (拒识模型，需本地训练)",
    "models/qwen3_lora_sft/                 (SFT 微调产物，需 LLaMA-Factory)",
    "models/qwen3_lora_sft_int4/            (INT4 量化产物)",
    "models/qwen3_lora_rl/                  (GRPO RL 产物)",
    "LLaMA-Factory-main/output/             (所有训练输出)",
]


# ── 核心逻辑 ────────────────────────────────────────────────────────────────


def resolve_base_dir(user_base_dir: str = "") -> Path:
    """解析项目根目录。

    优先级: --base-dir > FUSION_BASE_DIR > RAG_BASE_DIR > CARVOICE_BASE_DIR > 当前目录。
    """
    if user_base_dir:
        return Path(user_base_dir).resolve()
    for env_var in ("FUSION_BASE_DIR", "RAG_BASE_DIR", "CARVOICE_BASE_DIR", "PROJECT_HOME"):
        val = os.getenv(env_var, "")
        if val:
            return Path(val).resolve()
    # 默认：脚本所在目录的上一级（即项目根目录）
    return Path(__file__).resolve().parent.parent


def download_one(spec: ModelSpec, base_dir: Path, hf_token: str = "") -> None:
    target_dir = base_dir / spec.target_rel_path
    target_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{spec.source}]" if spec.source else ""
    print(f"[INFO]{tag} downloading {spec.name} -> {target_dir}")
    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=hf_token or None,
    )
    print(f"[DONE]{tag} {spec.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SU7_CarVoice_Fusion — 模型下载（合并 CarVoice + SU7_RAG）"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS.keys()),
        default="core",
        help="模型预设: agent | rag | core (默认) | all",
    )
    parser.add_argument(
        "--base-dir",
        default="",
        help="项目根目录。默认自动检测或使用当前目录。",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", ""),
        help="HuggingFace Token（用于门控/私有模型）。",
    )
    args = parser.parse_args()

    base_dir = resolve_base_dir(args.base_dir)
    print(f"[INFO] base_dir  = {base_dir}")
    print(f"[INFO] preset    = {args.preset}")
    print(f"[INFO] HF_ENDPOINT = {os.getenv('HF_ENDPOINT', '(未设置，用官方源)')}")
    print()

    failed: List[str] = []
    for spec in MODEL_PRESETS[args.preset]:
        try:
            download_one(spec, base_dir, args.hf_token)
        except Exception as exc:
            level = "ERROR" if spec.required else "WARN"
            print(f"[{level}] {spec.name} download failed: {exc}")
            if spec.required:
                failed.append(spec.name)

    print()
    print("[NOTE] 以下产物无法通过公开下载获取，需本地训练或手动准备：")
    for artifact in MANUAL_ARTIFACTS:
        print(f"       {artifact}")

    if failed:
        raise SystemExit(
            f"[FAILED] 必需模型下载失败: {', '.join(failed)}"
        )
    print("[SUCCESS] 模型下载完成。")


if __name__ == "__main__":
    main()
