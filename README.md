# SU7_CarVoice_Fusion

车载智能语音助手融合架构：以 CarVoice_Agent 为主控框架，按需调用 XIAOMI_SU7_RAG 知识检索。实现任务技能执行、用户手册问答、百科闲聊三路路由统一调度，支持 BERT 意图识别、LLM Function Calling 槽位提取、RAG 可溯源检索、Search-R1 动态工具调用与 GRPO 强化学习。

---

## 目录

1. [整体架构](#整体架构)
2. [快速开始](#快速开始)
3. [推理流程](#推理流程)
4. [训练流程](#训练流程)
5. [生产验证](#生产验证)
6. [API 接口](#api-接口)
7. [配置项](#配置项)
8. [项目结构](#项目结构)

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
| 1 | BERT 400+ 类 | 85% top-1 | RoBERTa-wwm-ext，31w 训练，训练后自动启用 |
| 2 | 启发式规则 | ~90% 常见场景 | 疑问语气 vs 祈使指令，Mock 默认 |
| 3 | LLM 仲裁 | ~98% | Doubao 182 行 Prompt，含隐含指令 |

### 路由决策

| 分类 | 触发条件 | 管线 | 输出 |
|------|---------|------|------|
| Task | 技能指令/疑问+技能域 | LLM→DM→MCP→NLG | task_result |
| FAQ | 用户手册提问 | RAG→引用拼装 | faq_answer+citations |
| Chitchat | 百科闲聊 | 拒识+联网→LLM | chitchat |

---

## 快速开始

### Mock 模式（30 秒启动，零依赖）

答案来自模板和本地规则，用于验证路由逻辑。

**Linux / macOS：**

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

**Windows：**

```powershell
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

**验证（Windows CMD）：**

```cmd
curl http://127.0.0.1:8080/healthz
# → {"status":"ok"}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"请导航到公司\"}"
# → {"type":"task_result","text":"已开始导航到公司。","trace":{"route":"Task",...}}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"SU7 续航是多少\"}"
# → {"type":"faq_answer","citations":[{"source":"su7_manual.pdf","page":12},{"source":"su7_quick_start.pdf","page":5}],...}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"今天天气怎么样\"}"
# → {"type":"chitchat","trace":{"route":"Task",...}}  # 天气→Task正确, Mock回退闲聊

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"你好\"}"
# → {"type":"chitchat","trace":{"route":"Chitchat",...}}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"我饿了\"}"
# → {"type":"clarification",...}  # Mock→Unknown, 生产→LLM仲裁→Task
```

### Mock 模式组件行为

| 组件 | 行为 |
|------|------|
| 意图分类 | 疑问/祈使语气规则，<1ms |
| 技能执行 | 7 个模板 |
| RAG 检索 | su7_docs.json (5条) TF 打分，返回 source+page |
| 闲聊 | 固定模板 |
| 联网搜索 | mock 预设 hint |
| 拒识 | 默认通过 |
| NLG | 原样返回 |
| 会话 | 内存 dict |

切生产：`.env` 中改 `LLM_PROVIDER=doubao`。

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
| 意图分类 | 规则 | BERT / Doubao 仲裁 |
| NLU | 关键词 | LLM Function Calling |
| RAG 检索 | su7_docs.json TF | BM25+Milvus→WRRF→MiniCPM |
| 闲聊 | 模板 | Doubao/vLLM 流式 |
| 联网搜索 | mock | SerpAPI/Serper/Bing/Doubao |
| 拒识 | 默认通过 | BERT-Tiny (Acc=89.7%) |
| NLG | 原样 | LLM 生成 |

---

## 训练流程

### SFT 微调（三种硬件）

```bash
# 12GB 单卡 → Unsloth QLoRA (最快, 显存最省)
pip install unsloth
python scripts/train_sft_unsloth.py

# 24GB 单卡 → LLaMA-Factory QLoRA (默认, 最稳定)
llamafactory-cli train configs/sft.yaml

# 2-4 卡 → LLaMA-Factory + DeepSpeed ZeRO-3 (全量微调)
llamafactory-cli train configs/sft.yaml --deepspeed configs/ds_z3_config.json
```

### Agent 训练

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 1. 下载 BERT 预训练模型
python scripts/download_models.py --preset agent

# 2. 意图分类训练 (31w 语料, Acc@5=97.6%)
python scripts/train_intent.py

# 3. 拒识模型训练 (32w 语料, Acc=89.7%)
python scripts/train_reject.py

# 4. 启动推理服务
python -m uvicorn app.train.servers:intent_app --port 8008
python -m uvicorn app.train.servers:reject_app --port 8007

# 配置 .env: NLU_URL / REJECT_URL 指向对应端口
```

### RAG 微调

```bash
# 下载模型 (BGE+SPLADE+MiniCPM+Qwen3-8B)
python scripts/download_models.py --preset rag

# 构建索引
python scripts/build_index.py --backend all

# 生成 QA 训练数据
python scripts/generate_data.py --step all

# SFT 微调 (需 LLaMA-Factory)
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory-main
cd LLaMA-Factory-main && pip install -r requirements.txt && pip install -e . && cd ..
cp data/training/summary/train.json LLaMA-Factory-main/data/summary_train.json
llamafactory-cli train configs/sft.yaml
llamafactory-cli export configs/sft.yaml

# vLLM 部署
vllm serve models/qwen3_lora_sft_int4 --host 0.0.0.0 --port 8000
```

### GRPO 强化学习

```bash
# 1. 生成轨迹
python app/rl/data_builder.py
python app/rl/build_local_trajectories.py

# 2. 格式转换 + 再平衡
python app/rl/format_converter.py
python app/rl/rebalance_sft_data.py

# 3. 训练（两种框架）

# TRL: 单卡快速验证 (80 条)
python app/rl/train_grpo.py --stage grpo

# VeRL: 多卡生产 (全量 21K+, 吞吐 3-5x)
pip install verl
python app/rl/train_grpo_verl.py --n-gpus 4

# 4. 验证导出
python app/rl/verify_export.py
```

### RL 推理

```bash
vllm serve models/qwen3_lora_rl --host 0.0.0.0 --port 8000
python scripts/rl_infer.py --model su7_rl --show-trajectory --show-reward
```

---

## 生产验证

### 硬件要求

| 场景 | 显卡 | 显存 | 存储 |
|------|------|------|------|
| Mock 开发 | 无需 | 0 | ~35MB |
| 推理(INT4) | RTX 3060 | ≥12GB | ~30GB |
| 推理(FP16) | RTX 3090/4090 | ≥24GB | ~50GB |
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

# 4. RAG 评估
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --dry-run

# 5. vLLM 压测
python scripts/benchmark_vllm.py --url http://127.0.0.1:8000/v1

# 6. 基线对比
python scripts/baseline_compare.py --model local
```

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

### WebSocket

```
ws://127.0.0.1:8080/ws/chat
→ {"message":"请关闭安全系统"}
← {"type":"clarification","trace":{"fallback_reason":"high_risk_needs_confirmation"}}
→ {"message":"确认执行","confirm":true,"session_id":"xxx"}
← {"type":"task_result"}
```

---

## 配置项

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_PROVIDER` | mock | mock / doubao / vllm / openai |
| `DOUBAO_API_KEY` | — | 豆包 API Key |
| `VLLM_BASE_URL` | http://127.0.0.1:8000/v1 | vLLM 推理地址 |
| `RETRIEVER_BACKEND` | mock | mock / bm25 / milvus / hybrid |
| `RERANKER_BACKEND` | mock | mock / minicpm |
| `HYBRID_DENSE_BACKEND` | milvus | milvus / faiss |
| `WEB_SEARCH_ENABLED` | false | 联网搜索 |
| `NLU_URL` | — | NLU 服务 (8009) |
| `REJECT_URL` | — | 拒识服务 (8007) |
| `REDIS_URL` | — | Redis (无则内存) |
| `AMAP_API_KEY` | — | 高德地图 |

完整见 `.env.example`。

---

## 项目验证

```bash
pytest -q -v                  # 63 tests
python scripts/run_agent.py -i                # 交互测试
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 评测
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --dry-run
```

---

## 脚本对照

| 原始 | 融合 |
|------|------|
| CarVoice download_models.py | scripts/download_models.py |
| CarVoice train/run.py | scripts/train_intent.py / train_reject.py |
| CarVoice dialog.py / test.py | scripts/run_agent.py |
| CarVoice server.sh | scripts/start_all_services.sh |
| SU7_RAG build_index.py | scripts/build_index.py |
| SU7_RAG generate_all_data.py | scripts/generate_data.py |
| SU7_RAG final_score.py | scripts/eval_rag.py |
| SU7_RAG infer.py / infer_rl.py | POST /api/v1/chat / scripts/rl_infer.py |
| SU7_RAG deploy/ | scripts/run_vllm.py / benchmark_vllm.py / baseline_compare.py |

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
│   ├── rl/                        # RL (web_reader, reward_model, environment, data_builder,
│   │                              #     format_converter, train_grpo, train_grpo_verl, infer_rl, ...)
│   ├── data_pipeline/             # QA生成/过滤/缩写/训练集
│   ├── eval/                      # scorer + RAGas
│   └── shared/                    # schemas, config, logging, redis, utils(WRRF)
├── scripts/                       # 17 执行脚本
│   ├── autodl_start.sh            # AutoDL 一键
│   ├── build_index.py             # RAG 索引
│   ├── generate_data.py           # 数据生成
│   ├── eval_rag.py               # RAG 评估
│   ├── run_agent.py               # Agent 测试
│   ├── download_models.py         # 模型下载
│   ├── run_vllm.py / benchmark_vllm.py / baseline_compare.py  # vLLM
│   ├── rl_infer.py               # RL 推理
│   ├── train_intent.py / train_reject.py / train_3class.py    # BERT 训练
│   ├── train_sft_unsloth.py       # Unsloth 加速训练
│   └── start_all_services.sh      # 多服务启动
├── configs/                       # sft.yaml, grpo.yaml, ds_z3_config.json
├── data/                          # 手工整理源数据 (21 文件)
├── tests/                         # 63 测试用例
├── .env.example                   # 环境变量模板 (~40 项)
├── docker-compose.yml / Dockerfile
└── requirements.txt
```
