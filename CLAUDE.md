# CLAUDE.md — SU7_CarVoice_Fusion

> 车载智能语音助手融合项目，合并自 CarVoice_Agent + XIAOMI_SU7_RAG

## 项目概述

以 CarVoice_Agent 为主控框架，按需调用 XIAOMI_SU7_RAG 知识检索，实现三路路由（Task/FAQ/Chitchat）统一调度。

## 关键路径

| 路径 | 说明 |
|------|------|
| `app/main.py` | FastAPI 入口 (端口 8080) |
| `app/core/` | 主控编排 (orchestrator, classifier) |
| `app/nlp/` | NLP 管线 (intent, reject, NLU, NLG) |
| `app/knowledge/` | RAG (retriever, reranker, chunker) |
| `app/llm/` | LLM 客户端 (doubao, vllm, openai, mock) |
| `app/train/` | BERT 训练 (core/models/servers) |
| `app/rl/` | Search-R1 强化学习 |
| `app/skills/` | 技能定义 + DM |
| `scripts/` | 30+ 执行脚本 |
| `configs/` | sft.yaml, grpo.yaml, ds_z3_config.json |
| `models/` | 预训练模型 + BERT checkpoints |
| `LLaMA-Factory-main/` | LLaMA-Factory 框架 + 训练产出 |

## 快速启动

```bash
# Mock 模式（零依赖）
uvicorn app.main:app --host 0.0.0.0 --port 8080

# 生产模式（需先设置 .env 中 LLM_PROVIDER=doubao）
bash scripts/start_all_services.sh
```

## 常用命令

```bash
# 测试
pytest -q -v                                           # 63 tests
python scripts/run_agent.py --query "导航到公司"        # 单条测试
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 批量评测

# 精度评测
python scripts/intent_benchmark.py                     # 意图精度
python scripts/reject_benchmark.py                     # 拒识精度

# 训练
python scripts/train_intent.py                         # BERT 意图训练
python scripts/train_reject.py                         # BERT 拒识训练
python scripts/train_sft_unsloth.py                    # Unsloth SFT

# vLLM
python scripts/run_vllm.py                             # 启动 vLLM
python scripts/benchmark_vllm.py                       # vLLM 压测
python scripts/rl_infer.py --model su7_rl              # RL 推理

# 数据
python scripts/build_index.py --backend bm25           # 构建索引
python scripts/generate_data.py --step all             # 生成 QA
python scripts/check_training_data.py                  # 数据检查
python scripts/evaluate_parse_quality.py               # 解析质量
```

## 已知注意事项

1. **LLaMA-Factory SFT**: configs/sft.yaml 需 `do_train: true`，去掉 bitsandbytes 量化
2. **bitsandbytes**: CUDA 13 无 libnvJitLink.so，FP16 替代 4-bit
3. **模型下载**: `scripts/download_models.py` 支持 ModelScope → HF snapshot → 逐文件 三级兜底
4. **numpy**: LLaMA-Factory 需要 numpy<2.0，与 vLLM 冲突。降级后需补丁 `np.long`/`np.ulong`
5. **路径**: 所有脚本已添加 `os.chdir()`，从任意目录运行均可
6. **LLaMA-Factory 产出**: `LLaMA-Factory-main/output/` (SFT/INT4/RL 完整模型)
7. **BERT checkpoints**: `models/saved/intent/bert.ckpt` + `models/saved/reject/bert_tiny.ckpt`
