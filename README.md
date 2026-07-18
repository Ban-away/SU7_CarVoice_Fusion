# SU7_CarVoice_Fusion

基于 **CarVoice_Agent** + **XIAOMI_SU7_RAG** 的完整融合后端。保留两个源项目的全部代码和业务逻辑，统一为可独立运行的单一服务。

---

## 融合架构评估

### 设计合理性

| 维度 | 评估 |
|------|------|
| 分层 | `api → core → {nlp, skills, knowledge, llm}` 三层清晰 |
| 技术栈保留 | 自定义BERT、豆包LLM、Redis、外部NLU/拒识服务（CarVoice）+ Milvus BGE+SPLADE、MiniCPM、Qwen3-8B vLLM、GRPO（SU7_RAG）全部保留 |
| 路由决策 | Task/FAQ/Chitchat/Unknown 四分类 + 拒识+关联 + 高风险二次确认，与CarVoice原始逻辑一致 |
| 知识融合 | FAQ路径走RAG检索+引用溯源（非原始闲聊回退），是融合的核心创新点 |
| 检索管线 | BM25/(Milvus)/Hybrid → WRRF → MiniCPM重排 → 可选LLM生成，与SU7_RAG原始管线一致 |
| LLM抽象 | Mock/Doubao/vLLM/OpenAI 四后端统一接口，配置切换 |
| 配置驱动 | `LLM_PROVIDER`/`RETRIEVER_BACKEND`/`RERANKER_BACKEND` 一行配置从Mock切生产 |

### 与原始项目的差异（有意设计）

| 差异 | 原始 | 融合 | 原因 |
|------|------|------|------|
| 架构 | 5个微服务 | 单进程 | 简化部署，Mock模式零依赖验证 |
| FAQ路由 | → 闲聊 | → RAG检索 | 融合核心价值 |
| 并行执行 | ThreadPoolExecutor 5任务 | 顺序 | 简化版，FastAPI异步可后续加 |
| BERT实现 | 自定义1206行 | 自定义1206行（保留） | 完全保留原始实现 |
| 框架 | Flask-SocketIO | FastAPI+WebSocket | 统一入口 |

---

## 推理流程

### 在线推理主流程

```
用户输入（HTTP POST /api/v1/chat 或 WebSocket ws://.../ws/chat）
  │
  ├─ 1. Query Rewrite（多轮指代消解）
  │     Mock: 简单拼接    Production: LLM改写 + 25%字符重叠安全校验
  │
  ├─ 2. Intent Classify（意图分类）
  │     Mock: 关键词匹配   Production: 豆包LLM 182行仲裁Prompt (A/B/C/D → task/faq/chat)
  │
  ├─ 3. Route（路由分发）
  │   │
  │   ├─ Task ── resolve_skill (7个白名单技能) ── execute ── NLG ──▶ task_result
  │   │   ├─ 高风险 (risk_level=high) ──▶ clarification (等 confirm=true)
  │   │   └─ 未匹配 ── NLU(外部服务) ── DM(maps/music/weather) ──▶ task_result
  │   │   └─ 全失败 ──▶ chat 回退
  │   │
  │   ├─ FAQ ── reject(外部拒识服务) + correlation(LLM关联判断)
  │   │   ├─ 拒识+不关联 ──▶ clarification
  │   │   └─ 接受 ── knowledge.retrieve ── synthesize_with_citations ──▶ faq_answer + citations
  │   │
  │   ├─ Chitchat ── reject + correlation ── LLM chat (BOT_CHAT prompt) ──▶ chitchat
  │   │
  │   └─ Unknown ──▶ clarification
  │
  └─ 4. Response（统一结构化响应）
       {type, text, citations[{source, page}], trace{route, confidence, latency_ms, ...}, session_id}
```

### Mock模式 vs 生产模式

| 组件 | Mock (LLM_PROVIDER=mock) | Production |
|------|--------------------------|------------|
| 仲裁 | 关键词规则 | 豆包LLM 182行Prompt |
| 闲聊 | 模板回复 | 豆包/vLLM流式生成 |
| NLG | 直接返回结果 | LLM转自然语言 |
| 改写 | 简单拼接 | LLM指代消解 |
| 检索 | BM25 TF打分 | BM25+Milvus(BGE+SPLADE)→WRRF→MiniCPM |
| NLU | 关键词匹配 | 外部NLU_URL服务(8009) |
| 拒识 | 默认不拒 | 外部REJECT_URL服务(8007) |
| 会话 | 内存dict | Redis |

### RAG 检索管线（XIAOMI_SU7_RAG 原始流程）

```
用户问题
  │
  ├─ Query Rewrite（HyDE风格，LLM扩展关键词）
  │
  ├─ BM25 Recall (top-15)   jieba分词 + 停用词过滤
  ├─ Milvus Hybrid (top-40) BGE-Large-zh-v1.5 dense + SPLADE v2 sparse, WeightedRanker
  │
  ├─ WRRF Fusion   wrrf_fusion([bm25, milvus], weights=[0.7, 0.7], k=60)
  │
  ├─ MiniCPM Rerank (top-12)   bge-reranker-v2-minicpm-layerwise
  │
  ├─ LLM Generate    Qwen3-8B vLLM streaming, 引用标记【1】【2】
  │
  └─ Post-Processing   正则提取引用 → 映射页码 + 图片
```

### RL Search-R1 推理流程

```
用户问题 → vLLM(RL模型)
  │
  ├─ 模型生成 "<search_local>关键词</search_local>"
  │   └─ 系统拦截 → LocalSearchTool(KBM25/Milvus) → 注入 <information>结果</information>
  │
  ├─ 模型继续 "<search_web>小米SU7 关键词</search_web>"（如本地不足）
  │   └─ 系统拦截 → WebSearchTool(bing→serpapi→serper→doubao) → 注入 <information>结果</information>
  │
  ├─ 模型可选 "<read_page>URL</read_page>"（深度阅读网页）
  │   └─ 系统拦截 → WebPageReader.fetch(url) → 注入 <information>页面内容</information>
  │
  └─ 模型生成 "<answer>最终答案</answer>"
      └─ reward_model.compute_reward() 6维评分
```

---

## 训练流程

### 训练流水线全景

```
┌─ 数据准备 ────────────────────────────────────────────────────────┐
│                                                                  │
│ 1. PDF → build_index.py → BM25/Milvus/FAISS索引                  │
│ 2. QA生成 → generate_data.py → qa_pair.json (±823条)              │
│ 3. 过滤 → generate_data.py --step filter → qa_pair_filtered.json  │
│ 4. 构建 → generate_data.py --step dataset → summary/ + rerank/    │
│                                                                  │
├─ Agent训练（CarVoice_Agent 原始流程）────────────────────────────  │
│                                                                  │
│ 1. 下载模型  download_models.py --preset agent                     │
│        → models/chinese_roberta_wwm_ext/                          │
│        → models/roberta_tiny_clue/                                │
│                                                                  │
│ 2. 意图训练  python scripts/train_intent.py                       │
│        → 数据: data/training/intent/train.txt (31w条)              │
│        → 模型: app/train/core/BertModel (自定义1206行BERT)         │
│        → 输出: models/saved/intent/bert.ckpt                       │
│        → Acc@1=85.2% Acc@5=97.6% F1=84.24%                        │
│                                                                  │
│ 3. 拒识训练  python scripts/train_reject.py                       │
│        → 数据: data/training/reject/train.txt (32w条)              │
│        → 模型: app/train/models/bert_tiny.py (3层BERT Tiny)        │
│        → 输出: models/saved/reject/bert_tiny.ckpt                  │
│        → Acc=89.7% F1=89.69%                                       │
│                                                                  │
├─ RAG训练（XIAOMI_SU7_RAG 原始流程）────────────────────────────    │
│                                                                  │
│ 1. 下载模型  download_models.py --preset rag                       │
│        → BGE-Large-zh-v1.5, SPLADE v2, MiniCPM, Qwen3-8B, etc.   │
│                                                                  │
│ 2. SFT微调  LLaMA-Factory (configs/sft.yaml)                     │
│        → QLoRA 4-bit, LoRA rank=16, 5 epoch                       │
│        → 训练数据: summary/train.json (19,878条)                   │
│        → 输出: models/qwen3_lora_sft/                             │
│        → loss: 0.3644/0.1533 (train/eval)                         │
│                                                                  │
│ 3. INT4量化  AWQ → models/qwen3_lora_sft_int4/                    │
│        → 吞吐提升 43.8% (465→669 token/s)                         │
│                                                                  │
├─ RL训练（Search-R1 + WebWalker）─────────────────────────────────  │
│                                                                  │
│ 1. 生成轨迹  app/rl/data_builder.py                                │
│        → 网络兜底轨迹: 每类50条, 10类=500条                        │
│        → 本地可答轨迹: app/rl/build_local_trajectories.py           │
│                                                                  │
│ 2. 格式转换  app/rl/format_converter.py                             │
│        → SFT格式 (LLaMA-Factory) + GRPO格式 (TRL)                  │
│        → 自动修复截断标签                                          │
│                                                                  │
│ 3. 再平衡  app/rl/rebalance_sft_data.py                             │
│        → web 33% : local 67%                                      │
│                                                                  │
│ 4. GRPO训练  app/rl/train_grpo.py                                   │
│        → SFT warmup (LLaMA-Factory, QLoRA 4-bit)                  │
│        → GRPO (TRL GRPOTrainer + PEFT LoRA rank=8)                 │
│        → 6维自定义奖励函数: reward_model.reward_fn()                │
│                                                                  │
│ 5. 导出验证  app/rl/verify_export.py                                │
│                                                                  │
└─ 评估 ───────────────────────────────────────────────────────────┘
│
├─ Agent评估  scripts/run_agent.py --eval --file data/nlu/multi_test.txt
│        → 端到端准确率 88.6%
│
├─ RAG评估  scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json
│        → 语义+关键词: 0.8965    RAGas context_recall: 0.9386
│
└─ RL评估  python -m app.rl.batch_eval --model su7_rl --vllm-url http://localhost:8000/v1
```

---

## 快速开始

### Mock模式（30秒启动，零依赖）

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

验证4条路由：

```bash
curl http://127.0.0.1:8080/healthz
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"请导航到公司"}'
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"SU7 续航是多少"}'
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"你好"}'
```

### 生产模式（GPU + vLLM）

```bash
# 环境
pip install -r requirements.txt
pip install rank-bm25 jieba sentence-transformers faiss-gpu pymilvus transformers torch
pip install vllm datasets accelerate peft bitsandbytes
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --preset core

# 配置 .env: LLM_PROVIDER=vllm, RETRIEVER_BACKEND=hybrid
cp .env.example .env

# 终端1: vLLM
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90

# 终端2: 融合服务
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### AutoDL一键部署

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
bash scripts/autodl_start.sh mock    # Mock验证
bash scripts/autodl_start.sh vllm    # 全功能
```

---

## 项目验证

```bash
# 单元测试（63 passed）
pytest -q -v

# Agent 流水线测试
python scripts/run_agent.py -i                          # 交互式
python scripts/run_agent.py --query "打开空调"           # 单条
python scripts/run_agent.py --file data/nlu/multi_test.txt  # 批量
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 评测

# RAG 检索测试
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" -d '{"query":"SU7 续航","top_k":3}'
```

---

## 脚本对照

| 原始项目 | 原始脚本 | 融合脚本 |
|---------|---------|---------|
| XIAOMI_SU7_RAG | build_index.py | scripts/build_index.py |
| XIAOMI_SU7_RAG | generate_all_data.py | scripts/generate_data.py --step qa/filter |
| XIAOMI_SU7_RAG | generate_sft_data.py | scripts/generate_data.py --step dataset |
| XIAOMI_SU7_RAG | final_score.py | scripts/eval_rag.py |
| XIAOMI_SU7_RAG | infer.py | POST /api/v1/chat |
| XIAOMI_SU7_RAG | src/rl/infer_rl.py | scripts/rl_infer.py |
| XIAOMI_SU7_RAG | deploy/auto_vllm_server | scripts/run_vllm.py |
| XIAOMI_SU7_RAG | deploy/benchmark.py | scripts/benchmark_vllm.py |
| XIAOMI_SU7_RAG | deploy/baseline_gpt4o | scripts/baseline_compare.py |
| XIAOMI_SU7_RAG | deploy/download_models | scripts/download_models.py |
| CarVoice_Agent | download_models.py | scripts/download_models.py |
| CarVoice_Agent | server.sh | scripts/start_all_services.sh |
| CarVoice_Agent | dialog.py | scripts/run_agent.py -i |
| CarVoice_Agent | test.py | scripts/run_agent.py --file |
| CarVoice_Agent | intent/reject/nlu_client | scripts/run_agent.py --eval |
| CarVoice_Agent | train/run.py | scripts/train_intent.py / scripts/train_reject.py |

---

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /healthz | 健康检查 |
| POST | /api/v1/chat | 单轮对话 |
| GET | /api/v1/skills | 技能白名单(7个) |
| GET | /api/v1/functions | 函数定义(455个) |
| POST | /api/v1/knowledge/retrieve | 知识检索 |
| WS | /ws/chat | WebSocket |

## 配置

| 变量 | 默认 | 说明 |
|------|------|------|
| LLM_PROVIDER | mock | mock/doubao/vllm/openai |
| RETRIEVER_BACKEND | mock | mock/bm25/milvus/hybrid |
| RERANKER_BACKEND | mock | mock/minicpm |
| HYBRID_DENSE_BACKEND | milvus | milvus/faiss |
| NLU_URL | — | 外部NLU服务 |
| REJECT_URL | — | 外部拒识服务 |
| REDIS_URL | — | Redis(无则内存) |

完整见 `.env.example`。

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/
│   ├── main.py                      # FastAPI入口
│   ├── api/                         # HTTP + WebSocket网关
│   ├── core/                        # 主控编排 (orchestrator, classifier, session)
│   ├── nlp/                         # NLP管道 (arbitration, rewrite, NLU, NLG, reject, correlation)
│   ├── skills/                      # 技能 (455定义, 白名单, 槽位, DM)
│   ├── knowledge/                   # RAG (retriever/reranker/generator/chunker/parser)
│   ├── llm/                         # LLM (doubao, vllm, openai, mock)
│   ├── mcp/                         # MCP (client, amap 13工具, qq音乐)
│   ├── prompts/                     # 7个System Prompt
│   ├── train/                       # BERT训练框架 (core, models, train_eval, data_helper, servers)
│   ├── rl/                          # RL模块 (12文件: web_reader, reward_model, environment, ...)
│   ├── data_pipeline/               # 数据管道 (QA生成, 过滤, 缩写, 训练集)
│   ├── eval/                        # 评估 (语义+关键词, RAGas)
│   └── shared/                      # 共享 (schemas, config, logging, redis, utils)
├── scripts/                         # 执行脚本 (14个)
├── configs/                         # 训练配置 (sft, grpo, original)
├── data/                            # 手工整理源数据 (21个文件)
├── tests/                           # 63个测试用例
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
