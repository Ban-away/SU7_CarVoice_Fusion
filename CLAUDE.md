# CLAUDE.md — SU7_CarVoice_Fusion

> 车载智能语音助手融合项目：CarVoice_Agent（主控框架）+ XIAOMI_SU7_RAG（知识检索）

## 快速导航

| 路径 | 说明 |
|------|------|
| `app/main.py` | FastAPI 入口 (端口 8080) |
| `app/core/` | 主控编排 (orchestrator, classifier, session) |
| `app/nlp/` | NLP 管线 (intent, reject, NLU, NLG, arbitration, rewrite, correlation) |
| `app/knowledge/` | RAG 检索 (8 retriever + 5 reranker + chunker + parser) |
| `app/llm/` | LLM 客户端 (doubao, vllm, openai, mock) |
| `app/train/` | BERT 训练 (core/models/servers) |
| `app/rl/` | Search-R1 强化学习 (data_builder, train_grpo, infer_rl, batch_eval) |
| `app/skills/` | 技能定义 + DM |
| `app/mcp/` | MCP 扩展工具 (高德地图 13工具 + QQ音乐) |
| `app/prompts/` | 7 System Prompts |
| `scripts/` | 30+ 执行脚本 |
| `configs/` | sft.yaml, grpo.yaml, ds_z3_config.json, export.yaml |
| `models/` | 预训练模型 (8个) + BERT checkpoints |
| `LLaMA-Factory-main/` | LLaMA-Factory 框架 + output/ (SFT/INT4/RL) |

## 快速启动

```bash
# Mock 模式
uvicorn app.main:app --host 0.0.0.0 --port 8080

# 生产模式（5服务一键启动）
bash scripts/start_all_services.sh

# vLLM 推理
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000
```

## 常用命令

```bash
# 测试
pytest -q -v                              # 63 tests
python scripts/run_agent.py --query "导航"  # 单条
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 批量

# 精度评测
python scripts/intent_benchmark.py        # 意图 Top-1 86.07%
python scripts/reject_benchmark.py        # 拒识 Acc 89.56%
python scripts/nlu_benchmark.py           # NLU 联合

# 训练
python scripts/train_intent.py            # BERT 意图
python scripts/train_reject.py            # BERT 拒识
python scripts/train_sft_unsloth.py       # Unsloth SFT
llamafactory-cli train configs/sft.yaml   # LLaMA-Factory SFT

# vLLM
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000
python scripts/benchmark_vllm.py
python scripts/rl_infer.py --model su7_rl

# 数据
python scripts/build_index.py --backend bm25
python scripts/generate_data.py --step all
python scripts/check_training_data.py
python scripts/evaluate_parse_quality.py
```

## 关键注意事项

1. **LLaMA-Factory**: `configs/sft.yaml` 需 `do_train: true`，注释掉 `quantization_bit` 和 `quantization_method`（CUDA 13 无 libnvJitLink）
2. **numpy**: 升级到 2.x 后 NP alias 补丁（sitecustomize.py）已自动处理
3. **模型下载**: `scripts/download_models.py` ModelScope → HF snapshot → 逐文件 三级兜底
4. **路径**: 所有脚本已添加 `os.chdir()`，从任意目录可直接运行
5. **产出位置**: `LLaMA-Factory-main/output/` (SFT 16GB/INT4 5.7GB/RL 16GB)
6. **BERT checkpoints**: `models/saved/intent/bert.ckpt` + `models/saved/reject/bert_tiny.ckpt`
7. **openai 版本**: 锁定 `openai==2.47.0`（vLLM + ragas 兼容窗口），`requirements.txt` 中 `openai>=1.0.0` 太宽导致版本漂移
8. **transformers 5.x 补丁**: LLaMA-Factory 的 `is_torch_sdpa_available`/`is_safetensors_available` 已打补丁

## 验证结果速览

| 指标 | 值 |
|------|-----|
| 意图 Top-1 | 86.07% |
| 意图 Top-5 | 97.64% |
| 拒识 Acc | 89.56% |
| vLLM 吞吐 | 1832 tok/s |
| TTFT | 133ms |
| 文档解析 | 98.31% |
| QPS 拒识 | 89ms |
| E2E 批量 | 242条 (task 79%) |
| RAG context_recall | 0.8847 |
| RAG context_precision | 0.9764 |
| PDF LLM 清洗 | ✅ |
| 语义切分 (m3e-small) | ✅ |
| HyDE 扩写 | ✅ |
