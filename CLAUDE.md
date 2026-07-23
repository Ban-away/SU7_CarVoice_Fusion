# CLAUDE.md — SU7_CarVoice_Fusion

> CarVoice_Agent + XIAOMI_SU7_RAG 融合项目 · 14/14 RAG + 9/9 Agent 全部验证通过

## 快速导航

| 路径 | 说明 |
|------|------|
| `app/main.py` | FastAPI 入口 (8080) |
| `app/core/` | 主控编排 (orchestrator, classifier, session) |
| `app/nlp/` | NLP (intent, reject, NLU, NLG, arbitration, rewrite, correlation) |
| `app/knowledge/` | RAG (BM25/FAISS/Milvus retriever + MiniCPM reranker + chunker + web_search) |
| `app/llm/` | LLM (doubao/vllm/openai/mock + llm_clean + llm_hyde) |
| `app/train/` | BERT 训练 (core/models/servers) |
| `app/rl/` | Search-R1 RL (data_builder, train_grpo, infer_rl, batch_eval, web_reader) |
| `app/skills/` | 技能定义 + DM (maps/music/weather) |
| `app/mcp/` | MCP (Amap 13工具+QQ音乐) |
| `scripts/` | 30+ 执行脚本 |
| `configs/` | sft.yaml, grpo.yaml, ds_z3_config.json, export.yaml |
| `LLaMA-Factory-main/output/` | SFT(16GB)/INT4(5.7GB)/RL(16GB) |

## 常用命令

```bash
# 测试
pytest -q -v                              # 63 tests
python scripts/run_agent.py --query "导航"  # 单条
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 批量

# 精度
python scripts/intent_benchmark.py        # Top-1 86.07%, Top-5 97.64%
python scripts/reject_benchmark.py        # Acc 89.56%

# 训练
python scripts/train_intent.py            # BERT 意图, 392MB
python scripts/train_reject.py            # BERT 拒识, 24MB
python scripts/train_sft_unsloth.py       # Unsloth SFT
llamafactory-cli train configs/sft.yaml   # LLaMA-Factory (需 do_train:true + 无量化)

# 索引
python scripts/build_index.py --backend all --pdf data/knowledge/Xiaomi_SU7_Manual.pdf

# vLLM
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000
python scripts/benchmark_vllm.py          # 1832 tok/s, TTFT 133ms
python scripts/rl_infer.py --model su7_rl # RL 推理

# GRPO
python app/rl/data_builder.py --dry-run   # 网络轨迹
python app/rl/train_grpo.py --stage grpo  # GRPO 训练

# 数据
python scripts/generate_data.py --step all
python scripts/evaluate_parse_quality.py  # 98.31%
python scripts/check_training_data.py     # 7/7

# QPS
locust -f scripts/reject_qps.py --host http://127.0.0.1:8007 --headless -u 10 -r 10 -t 5s
```

## 关键注意事项

1. **LLaMA-Factory**: `do_train: true`, 注释掉 `quantization_bit`/`quantization_method`
2. **openai**: 锁定 `2.47.0`（vLLM 0.25.1 + ragas 兼容窗口）
3. **模型下载**: ModelScope → HF snapshot → 逐文件 三级策略
4. **MongoDB**: 需捆绑 OpenSSL 1.1 (dpkg -x libssl1.1.deb)
5. **Milvus**: 需启动 milvus-lite 本地服务 `localhost:19530`
6. **transformers 5.x**: LLaMA-Factory + MiniCPM 需要函数补丁
7. **numpy≥2.0**: sitecustomize.py 别名补丁
8. **BERT checkpoints**: `models/saved/intent/bert.ckpt` + `models/saved/reject/bert_tiny.ckpt`

## 验证结果

| 指标 | 值 |
|------|-----|
| 意图 Top-1 | 86.07% |
| 意图 Top-5 | 97.64% |
| 拒识 Acc | 89.56% |
| vLLM 吞吐 | 1832 tok/s |
| TTFT | 133ms |
| 文档解析 | 98.31% |
| QPS 拒识 | 89ms |
| E2E 批量 | 242条 |
| RAG recall | 0.88 |
| RAG precision | 0.98 |
