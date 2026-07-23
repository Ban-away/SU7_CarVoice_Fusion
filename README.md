# SU7_CarVoice_Fusion

车载智能语音助手融合架构：以 CarVoice_Agent 为主控框架，按需调用 XIAOMI_SU7_RAG 知识检索。实现任务技能执行、用户手册问答、百科闲聊三路路由统一调度，支持 BERT 意图识别、LLM Function Calling 槽位提取、RAG 可溯源检索、Search-R1 动态工具调用与 GRPO 强化学习。

> **验证环境**：RTX 4090 (48GB), Python 3.12, PyTorch 2.13, CUDA 13.2
> **验证日期**：2026-07-10 ~ 2026-07-23

---

## 目录

1. [整体架构](#整体架构)
2. [快速开始](#快速开始)
3. [推理流程](#推理流程)
4. [训练流程](#训练流程)
5. [生产验证](#生产验证)
6. [API 接口](#api-接口)
7. [配置项](#配置项)
8. [项目验证](#项目验证)
9. [脚本对照](#脚本对照)
10. [项目结构](#项目结构)
11. [验证报告](#验证报告)
12. [代码修复记录](#代码修复记录)

---

## 整体架构

```
用户输入 → 三级分类器（BERT粗召回→规则→LLM仲裁）→ 路由分发

Task（技能执行）              FAQ（手册问答）             Chitchat（百科闲聊）
  BERT意图粗召回                RAG检索管线                 拒识 + 关联判断
  LLM Function Calling          BM25+Milvus→WRRF           → 联网搜索
  → 槽位归一化                   →MiniCPM→LLM               (SerpAPI/Serper
  → DM (maps/music/weather)     →citations[{source,page}]   /Bing/Doubao)
  → MCP (Amap 13工具+QQ音乐)                                → LLM整合
  → NLG友好回复                                          → 时效性问答
```

### 意图分类

| 级别 | 方案 | 准确率 | 说明 |
|------|------|--------|------|
| 1 | BERT 400+ 类 | 86.07% top-1 | RoBERTa-wwm-ext，31w 训练 |
| 2 | 启发式规则 | ~90% 常见场景 | 疑问语气 vs 祈使指令 |
| 3 | LLM 仲裁 | ~98% | Doubao 182 行 Prompt |

### 路由决策

| 分类 | 触发条件 | 管线 | 输出 |
|------|---------|------|------|
| Task | 技能指令/疑问+技能域 | LLM→DM→MCP→NLG | task_result |
| FAQ | 用户手册提问 | RAG→引用拼装 | faq_answer+citations |
| Chitchat | 百科闲聊 | 拒识+联网→LLM | chitchat |

---

## 快速开始

### Mock 模式（30 秒启动，零依赖）

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

> **注意**：部分依赖（`huggingface_hub`、`transformers`、`PyMuPDF`、`python-dotenv`）未列入 `requirements.txt`，首次运行需手动安装：
> ```bash
> pip install huggingface_hub transformers PyMuPDF python-dotenv
> ```

**验证：**

```bash
curl http://127.0.0.1:8080/healthz
# → {"status":"ok"}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"请导航到公司"}'
# → {"type":"task_result","text":"已开始导航到公司。","trace":{"route":"Task",...}}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"SU7 续航是多少"}'
# → {"type":"faq_answer","citations":[{"source":"su7_manual.pdf","page":12},...]}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"你好"}'
# → {"type":"chitchat","trace":{"route":"Chitchat",...}}
```

### 切生产模式

`.env` 中修改：

```bash
LLM_PROVIDER=doubao
DOUBAO_API_KEY=ark-xxx
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL_NAME=doubao-1-5-lite-32k-250115
```

### Mock 模式组件行为

| 组件 | 行为 |
|------|------|
| 意图分类 | 疑问/祈使语气规则，<1ms |
| 技能执行 | 7 个模板 |
| RAG 检索 | su7_docs.json (5条) TF 打分 |
| 闲聊 | 固定模板 |
| 联网搜索 | mock 预设 hint |
| 拒识 | 默认通过 |
| NLG | 原样返回 |
| 会话 | 内存 dict |

---

## 推理流程

### Task — 技能执行

```
"导航到公司"
  → BERT 意图粗召回 (400+ 类 → Top-5)
  → LLM Function Calling (从 Top-5 选最佳 + 提取槽位)
  → 槽位归一化 (位置映射、极值提取、百分比转换)
  → DM (maps.py → 调用 MCP Amap 搜索地点)
  → NLG ("好的，已为您找到公司附近的路线")
```

### FAQ — 手册问答

```
"SU7 续航多少"
  → BM25 召回 (top-15, jieba+停用词)
  → Milvus 召回 (top-40, BGE-Large Dense + SPLADE Sparse)
  → WRRF 融合 (weights=[0.7,0.7], k=60)
  → MiniCPM 重排 (top-12)
  → LLM 答案生成 (Qwen3-8B, 引用标记)
  → citations[{source, page}]
```

### Chitchat — 百科闲聊

```
"周杰伦是谁"
  → 拒识模型 + 多轮关联判断
  → 通过 → 联网搜索 (SerpAPI→Serper→Bing→Doubao)
  → LLM 整合 → 回复
  → 拒识 → clarification
```

### Mock vs 生产

| 组件 | Mock | 生产 |
|------|------|------|
| 意图分类 | 规则 | BERT (Top-1 86.07%) |
| NLU | 关键词 | LLM Function Calling |
| RAG 检索 | su7_docs.json TF | BM25+Milvus→WRRF→MiniCPM |
| 闲聊 | 模板 | Doubao/vLLM 流式 |
| 联网搜索 | mock | SerpAPI/Serper/Bing/Doubao |
| 拒识 | 默认通过 | BERT-Tiny (Acc=89.56%) |
| NLG | 原样 | LLM 生成 |

---

## 训练流程

### SFT 微调（三种硬件）

| 方式 | 框架 | 显存 | 验证结果 |
|------|------|------|------|
| peft + trl | 自定义脚本 | ≥16GB | ✅ loss 3.39→1.73 |
| Unsloth 2x | unsloth 2026.7 | ≥12GB | ✅ loss 2.39→2.08, 1.5s/step |
| LLaMA-Factory | llamafactory 0.9.3 | ≥24GB | ⚠️ 版本冲突 |

```bash
# peft + trl (FP16, 最简)
python scripts/run_sft_minimal.py

# Unsloth QLoRA (最快)
pip install unsloth
python scripts/train_sft_unsloth.py

# LLaMA-Factory QLoRA (默认, 最稳定)
llamafactory-cli train configs/sft.yaml
```

### Agent 训练

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 1. 下载 BERT 预训练模型
python scripts/download_models.py --preset agent

# 2. 意图分类训练 (31w 语料)
python scripts/train_intent.py

# 3. 拒识模型训练 (32w 语料)
python scripts/train_reject.py

# 4. 启动推理服务
python -m uvicorn app.train.servers:intent_app --port 8008
python -m uvicorn app.train.servers:reject_app --port 8007
```

**实测结果：**

| 指标 | 值 |
|------|-----|
| 意图 Top-1 | **86.07%** |
| 意图 Top-5 | **97.64%** |
| 拒识 Accuracy | **89.56%** |
| 拒识 F1 | **89.57%** |

### RAG 微调

```bash
# 下载模型 (BGE+SPLADE+MiniCPM+Qwen3-8B)
python scripts/download_models.py --preset rag

# 构建索引
python scripts/build_index.py --backend bm25

# 生成 QA 训练数据 (需 LLM_PROVIDER=doubao)
python scripts/generate_data.py --step all

# 构造 SFT 数据集
python scripts/generate_sft_data.py

# LLaMA-Factory (需先安装 LLaMA-Factory)
cp data/training/summary/train.json LLaMA-Factory-main/data/summary_train.json
llamafactory-cli train configs/sft.yaml
llamafactory-cli export configs/sft.yaml

# vLLM 部署
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --host 0.0.0.0 --port 8000
```

### GRPO 强化学习

```bash
# 1. 生成轨迹
python app/rl/data_builder.py          # 网络兜底轨迹 (需 SerpAPI)
python app/rl/build_local_trajectories.py  # 本地可答轨迹

# 2. 格式转换 + 再平衡
python app/rl/format_converter.py
python app/rl/rebalance_sft_data.py

# 3. 训练（两种框架）
python app/rl/train_grpo.py --stage grpo       # TRL: 单卡快速验证
python app/rl/train_grpo_verl.py --n-gpus 4    # VeRL: 多卡生产

# 4. 验证导出
python app/rl/verify_export.py
```

### RL 推理

```bash
vllm serve LLaMA-Factory-main/output/qwen3_lora_rl --host 0.0.0.0 --port 8000
python scripts/rl_infer.py --model su7_rl --show-trajectory --show-reward
```

---

## 生产验证

### 硬件要求

| 场景 | 显卡 | 显存 | 存储 |
|------|------|------|------|
| Mock 开发 | 无需 | 0 | ~35MB |
| 推理(INT4) | RTX 3060 | ≥12GB | ~6GB |
| 推理(FP16) | RTX 3090/4090 | ≥24GB | ~16GB |
| Unsloth SFT | RTX 3060 | ≥12GB | ~50GB |
| QLoRA SFT | RTX 3090/4090 | ≥24GB | ~50GB |
| DeepSpeed 全量 | 2×RTX 3090 | 48GB+ | ~100GB |
| TRL GRPO | RTX 3090 | ≥24GB | ~50GB |
| VeRL GRPO | 2-4×RTX 3090 | 48GB+ | ~100GB |

### 验证清单

```bash
# 1. 硬件确认
nvidia-smi && curl http://127.0.0.1:8000/v1/models

# 2. RAG 检索
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" -d '{"query":"SU7 续航","top_k":3}'

# 3. LLM 生成（确认非模板）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" -d '{"message":"你好"}'

# 4. 数据质量检查
python scripts/check_training_data.py

# 5. 文档解析质量
python scripts/evaluate_parse_quality.py
# → 解析准确率 98.31%

# 6. RAG 评估
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --skip-ragas

# 7. vLLM 压测
python scripts/benchmark_vllm.py --url http://127.0.0.1:8000/v1

# 8. 基线对比
python scripts/baseline_compare.py --model local --local-url http://127.0.0.1:8000/v1
```

**实测结果：**

| 指标 | 值 |
|------|-----|
| vLLM 吞吐率 | **1832 tok/s** |
| TTFT 均值 | **133ms** |
| 文档解析准确率 | **98.31%** |
| RAG 综合评分 | 0.42 (mock) / 0.90 (生产预期) |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话 |
| `GET` | `/api/v1/skills` | 技能白名单 (7个) |
| `GET` | `/api/v1/functions` | 函数定义 (400+) |
| `POST` | `/api/v1/knowledge/retrieve` | 知识检索调试 |
| `WS` | `/ws/chat` | WebSocket 实时会话 |

### 请求/响应

```json
POST /api/v1/chat
{"message":"请导航到公司","confirm":false,"session_id":null}

{
  "type": "task_result",
  "text": "已开始导航到公司。",
  "citations": [],
  "trace": {
    "route": "Task", "classifier_confidence": 0.9, "latency_ms": 1,
    "fallback_reason": null, "risk_level": "medium", "session_id": "uuid"
  },
  "session_id": "uuid"
}
```

`type`: task_result | faq_answer | chitchat | clarification | error

---

## 配置项

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_PROVIDER` | mock | mock / doubao / vllm / openai |
| `DOUBAO_API_KEY` | — | 豆包 API Key |
| `DOUBAO_BASE_URL` | https://ark.cn-beijing.volces.com/api/v3 | 豆包地址 |
| `DOUBAO_MODEL_NAME` | — | 豆包模型名 |
| `VLLM_BASE_URL` | http://127.0.0.1:8000/v1 | vLLM 推理地址 |
| `RETRIEVER_BACKEND` | mock | mock / bm25 / milvus / hybrid |
| `RERANKER_BACKEND` | mock | mock / minicpm |
| `WEB_SEARCH_ENABLED` | false | 联网搜索开关 |
| `SERPAPI_KEY` | — | SerpAPI Key |
| `NLU_URL` | http://127.0.0.1:8009/chatnlu-server/v1 | NLU 服务 |
| `REJECT_URL` | http://127.0.0.1:8007/reject-server/v1 | 拒识服务 |
| `INTENT_URL` | http://127.0.0.1:8008/intent-server/v1 | 意图服务 |
| `REDIS_HOST` | 127.0.0.1 | Redis 地址 |
| `AMAP_API_KEY` | — | 高德地图 |
| `SERPAPI_KEY` / `SERPER_API_KEY` | — | 网络搜索 |

SFT/RL 模型路径（LLaMA-Factory 产出）：

```
LLaMA-Factory-main/output/
├── qwen3_lora_sft/          (16GB)  基础 SFT
├── qwen3_lora_sft_int4/     (5.7GB) INT4 量化 → vLLM 推理
└── qwen3_lora_rl/           (16GB)  GRPO 强化学习
```

完整见 `.env.example`。

---

## 项目验证

```bash
pytest -q -v                              # 63 tests
python scripts/run_agent.py -i             # 交互测试
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 批量评测
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --skip-ragas
python scripts/intent_benchmark.py         # 意图精度评测
python scripts/reject_benchmark.py         # 拒识精度评测
python scripts/check_training_data.py      # 训练数据检查
python scripts/evaluate_parse_quality.py   # 文档解析质量
```

---

## 脚本对照

| 原始 | 融合 |
|------|------|
| CarVoice download_models.py | scripts/download_models.py |
| CarVoice train/run.py | scripts/train_intent.py / train_reject.py |
| CarVoice dialog.py / test.py | scripts/run_agent.py |
| CarVoice server.sh | scripts/start_all_services.sh |
| CarVoice test/reject_client.py | scripts/reject_benchmark.py |
| CarVoice test/intent_client.py | scripts/intent_benchmark.py |
| CarVoice test/nlu_client.py | scripts/nlu_benchmark.py |
| CarVoice test/*benchmark*.py (locust) | scripts/intent_qps.py / reject_qps.py / nlu_qps.py |
| CarVoice e2e_score.py | scripts/e2e_score.py |
| SU7_RAG build_index.py | scripts/build_index.py |
| SU7_RAG generate_all_data.py | scripts/generate_data.py |
| SU7_RAG generate_sft_data.py | scripts/generate_sft_data.py |
| SU7_RAG final_score.py | scripts/eval_rag.py |
| SU7_RAG infer.py / infer_rl.py | POST /api/v1/chat / scripts/rl_infer.py |
| SU7_RAG deploy/auto_vllm_server.py | scripts/run_vllm.py |
| SU7_RAG deploy/benchmark.py | scripts/benchmark_vllm.py |
| SU7_RAG deploy/baseline_gpt4o.py | scripts/baseline_compare.py |
| SU7_RAG evaluate_parse_quality.py | scripts/evaluate_parse_quality.py |
| SU7_RAG check_training_data.py | scripts/check_training_data.py |

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/
│   ├── main.py                    # FastAPI 入口
│   ├── api/                       # HTTP + WebSocket 网关
│   ├── core/                      # 主控编排 (orchestrator, classifier, session)
│   ├── nlp/                       # NLP (arbitration, intent, rewrite, NLU, NLG, reject, correlation)
│   ├── skills/                    # 技能 (definitions, registry, slot_processor, dm/)
│   ├── knowledge/                 # RAG (8 retriever, 5 reranker, generator, chunker, parser)
│   ├── llm/                       # LLM (doubao, vllm, openai, mock)
│   ├── mcp/                       # MCP (client, amap 13工具, qq音乐)
│   ├── prompts/                   # 7 System Prompt
│   ├── train/                     # BERT 训练 (core, models, train_eval, servers, run)
│   ├── rl/                        # RL (data_builder, format_converter, reward_model, train_grpo, infer_rl, ...)
│   ├── data_pipeline/             # QA生成/过滤/训练集
│   ├── eval/                      # scorer + RAGas
│   └── shared/                    # schemas, config, logging, redis, utils(WRRF)
├── scripts/                       # 30+ 执行脚本
│   ├── autodl_start.sh            # AutoDL 一键
│   ├── start_all_services.sh      # 5 服务一键启动
│   ├── download_models.py         # 模型下载 (Agent + RAG 合并, ModelScope优先)
│   ├── train_intent.py / train_reject.py / train_3class.py    # BERT 训练
│   ├── train_sft_unsloth.py       # Unsloth SFT
│   ├── build_index.py             # RAG 索引构建
│   ├── generate_data.py           # QA 数据生成
│   ├── generate_sft_data.py       # SFT 数据构造 (Summary+Rerank)
│   ├── eval_rag.py                # RAG 离线评估
│   ├── evaluate_parse_quality.py  # 文档解析质量
│   ├── check_training_data.py     # 训练数据质量检查
│   ├── intent_benchmark.py        # 意图精度评测
│   ├── reject_benchmark.py        # 拒识精度评测
│   ├── nlu_benchmark.py           # NLU 联合评测
│   ├── intent_qps.py / reject_qps.py / nlu_qps.py    # QPS 压测 (locust)
│   ├── e2e_score.py               # 端到端准确率统计
│   ├── run_agent.py               # Agent 测试 (单轮/批量/交互/评测)
│   ├── run_vllm.py                # vLLM 自动部署
│   ├── benchmark_vllm.py          # vLLM 性能压测
│   ├── baseline_compare.py        # 基线对比
│   ├── rl_infer.py                # RL 推理
│   ├── predict.py                 # LLaMA-Factory 批量预测
│   └── run_sft_minimal.py / run_grpo_minimal.py  # 快速测试
├── configs/                       # sft.yaml, grpo.yaml, ds_z3_config.json
├── data/                          # 训练数据 + 知识库
│   ├── training/                  # intent/, reject/, summary/, qa_pairs/
│   ├── knowledge/                 # PDF + su7_docs.json + saved_index/
│   └── nlu/                       # intent_map, slot_intent, test files
├── LLaMA-Factory-main/            # LLaMA-Factory 框架
│   └── output/                    # qwen3_lora_sft / int4 / rl
├── models/                        # 预训练模型 + BERT checkpoints
│   ├── Qwen3-8B/                  # 15.5GB
│   ├── chinese_roberta_wwm_ext/   # ~400MB
│   ├── roberta_tiny_clue/         # 25MB
│   ├── BAAI/                      # bge-large, bge-reranker
│   ├── moka-ai/  naver/           # m3e, splade
│   ├── text2vec-base-chinese/     # 391MB
│   ├── saved/                     # intent/bert.ckpt, reject/bert_tiny.ckpt
│   └── qwen3_lora_sft_*/         # 测试产物
├── mongodb-7.0.20/                # MongoDB 二进制 (可选)
├── tests/                         # 63 测试用例
├── log/                           # 运行日志
├── docs/                          # 文档
├── .env.example                   # 环境变量模板
├── docker-compose.yml / Dockerfile
└── requirements.txt
```

---

## 验证报告

### 全部已验证项

| 模块 | 状态 | 关键指标 |
|------|------|------|
| Mock 推理 | ✅ | 5/5 curl 正确 |
| pytest | ✅ | 63/63 passed |
| BERT 意图训练 + 评测 | ✅ | Top-1 86.07%, Top-5 97.64% |
| BERT 拒识训练 + 评测 | ✅ | Accuracy 89.56%, F1 89.57% |
| BM25 索引 | ✅ | 278页 → 144 chunks |
| Doubao QA 生成 | ✅ | 15 QA pairs, 5×200 OK |
| SerpAPI 网络轨迹 | ✅ | 3 trajectories |
| SFT (peft+trl) | ✅ | loss 3.39→1.73 |
| SFT (Unsloth 2x) | ✅ | loss 2.39→2.08 |
| GRPO 训练 | ✅ | 5 steps, TRL GRPOTrainer |
| vLLM 部署 + API | ✅ | SFT INT4, 回答正确 |
| vLLM 压测 | ✅ | 1832 tok/s, TTFT 133ms |
| 基线对比 | ✅ | 0.88s/请求 |
| RL 推理 | ✅ | 检索+生成正确 |
| RAG 评估 | ✅ | 流程跑通 |
| 文档解析质量 | ✅ | **98.31%** |
| 训练数据检查 | ✅ | 意图31w+拒识32w, 7/7通过 |
| SFT 数据构造 | ✅ | Summary+Rerank 12/3 |
| start_all_services | ✅ | 5/5 服务启动 |
| Redis | ✅ | v6.0.16 |
| VeRL | ✅ | v0.8.0 |
| 全部模型下载 | ✅ | 8/8 (ModelScope+HF) |

### 模型下载方案

| 策略 | 适用模型 |
|------|------|
| ModelScope（国内优先） | Qwen3-8B, bge-large-zh-v1.5 |
| HF mirror snapshot | chinese_roberta, m3e, splade, miniCPM |
| HF 逐文件下载（绕过 Xet） | text2vec-base-chinese |
| 手动上传 | roberta_tiny_clue |

### 待完成项

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| 1 | LLaMA-Factory SFT | ⚠️ | transformers 5.6 与 llamafactory 0.9.3 版本冲突, 产出已由用户从原始项目提供 |
| 2 | NLU 联合评测 | ⏳ | 脚本就绪 |
| 3 | E2E 评测 | ✅ | **批量 242条: task 79%, faq 3%, chat 17%, 平均 4138ms** |
| 4 | QPS 压测 (locust) | ⚠️ | 框架跑通, API 格式需适配 |
| 5 | GRPO 全量链路 | ⏳ | 需完整轨迹数据 |
| 6 | RL batch_eval | ⏳ | 需 RL vLLM + 测试数据 |
| 7 | Docker 部署 | ⏳ | 未测试 |

---

## 代码修复记录

验证过程中发现并修复的代码缺陷：

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `app/train/models/` | 单文件非包，bert_tiny 缺失 | 创建 bert.py + bert_tiny.py 包 |
| 2 | `app/train/run.py` | 缺 os.chdir + 导入路径错误 | 添加 chdir + 统一导入 |
| 3 | `scripts/build_index.py` | 调用不存在方法 `split()` | 改用 `chunk_text()` |
| 4 | `app/knowledge/chunker.py` | `_split()` 递归溢出 | 添加前向进度 guard |
| 5 | `app/nlp/intent.py` | 模型调用接口不匹配 | 改为 `forward(input_ids, mask)` |
| 6 | `app/llm/base.py` | Doubao API Key 未注入 | 添加 settings 自动读取 |
| 7 | `app/llm/doubao.py` | 参数命名不规范 | 统一 `base_url`/`model_name` |
| 8 | `app/shared/config.py` | `.env` 未自动加载 | 添加 `python-dotenv` 加载 |
| 9 | `app/train/data_loader.py` | HF tokenizer 与自定义 BERT 不兼容 | 改用自定义 `BertTokenizer` |
| 10 | `models/roberta_tiny_clue/` | config 与实际权重不匹配 | 修正 hidden_size=312, vocab=8021 |
| 11 | `scripts/download_models.py` | 单一 HF 源，Xet 存储 401 | ModelScope → snapshot → 逐文件 三级兜底 |
| 12 | `scripts/*.py` (多个) | 相对路径导致 FileNotFoundError | 统一添加 `os.chdir()` |

---

## 许可证

MIT License
