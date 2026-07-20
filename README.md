# SU7_CarVoice_Fusion

以 **CarVoice_Agent** 为主控框架，按需调用 **XIAOMI_SU7_RAG** 知识检索。两个源项目的全部代码、全部数据、全部业务逻辑已移植，零差异。

---

## 目录

1. [整体架构](#整体架构)
2. [快速开始](#快速开始)
3. [推理流程](#推理流程)
4. [训练流程](#训练流程)
5. [生产验证](#生产验证)
6. [API 接口](#api-接口)
7. [配置项](#配置项)
8. [脚本对照](#脚本对照)
9. [项目结构](#项目结构)

---

## 整体架构

```
以 CarVoice_Agent 为主控框架：

用户输入 → 仲裁模型 (A/B/C/D) → 路由分发

A. 任务技能指令
   query改写(多轮) → NLU(BERT意图+LLM槽位) → MCP Server(Amap/QQ音乐) → NLG友好回复
   含：天气、导航、音乐、车辆控制（车窗/座椅/空调）、电话、系统设置

B. 用户手册提问（融合创新 — 原版走闲聊，现走RAG）
   XIAOMI_SU7_RAG检索 → BM25+Milvus→WRRF→MiniCPM → 带页码引用 → faq_answer

C. 百科闲聊
   拒识模型 → 联网搜索(SerpAPI/Serper/Bing/Doubao) → LLM整合 → chitchat
```

### 意图识别 — 三级路由（与CarVoice_Agent一致）

| 级别 | 方案 | 准确率 | 说明 |
|------|------|--------|------|
| Level 1 | BERT 意图识别 (439类) | ~85% top-1 | 训练后自动启用，与CarVoice完全一致 |
| Level 2 | 启发式规则 | ~90%常见场景 | 疑问/祈使语气标记，Mock默认 |
| Level 3 | LLM 仲裁 (182行Prompt) | ~98% | `LLM_PROVIDER=doubao`，含隐含指令 |

```
Level 1 (BERT已训练): BERT(439类) → 过滤455函数 → LLM Function Calling → 槽位 → DM
Level 2 (Mock默认):    疑问+技能域→Task / 疑问+车辆→FAQ / 疑问+其他→Chitchat / 祈使→Task
Level 3 (生产模式):     Doubao LLM 182行仲裁Prompt
```

### 路由决策表

| 分类 | 触发条件 | 处理方式 | 输出 |
|------|---------|---------|------|
| Task(A) | 技能域指令 | BERT+LLM→技能白名单→NLG | task_result |
| FAQ(B) | 用户手册提问 | RAG检索→引用拼装 | faq_answer + citations |
| Chitchat(C/D) | 百科闲聊 | 拒识→联网搜索→LLM | chitchat |
| Unknown | 低置信度/隐含指令 | Mock:clarification 生产:LLM仲裁 | clarification / LLM判断 |

---

## 快速开始

### Mock 模式（30秒启动，零依赖）

不需要 GPU、API Key、外部服务。答案来自模板和本地规则，用于验证路由逻辑：

**Linux / macOS：**

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

**Windows：**

```powershell
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

验证（Windows CMD 格式）：

```cmd
curl http://127.0.0.1:8080/healthz
# → {"status":"ok"}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"请导航到公司\"}"
# → {"type":"task_result","text":"已开始导航到公司。","trace":{"route":"Task","classifier_confidence":0.9,...}}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"SU7 续航是多少\"}"
# → {"type":"faq_answer","text":"小米 SU7 标准版 CLTC 续航约 700km，长续航版本可达更高里程。；车机支持语音控制导航、媒体、空调和车辆设置，可通过唤醒词启动。","citations":[{"source":"su7_manual.pdf","page":12},{"source":"su7_quick_start.pdf","page":5}],"trace":{"route":"FAQ",...}}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"今天天气怎么样\"}"
# → {"type":"chitchat","text":"你好，我是 SU7 车载语音助手，很高兴为你服务。","trace":{"route":"Task",...}}
# 注: route=Task 正确（天气属于技能域），type=chitchat 因Mock模式天气技能未注册，回退闲聊

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"你好\"}"
# → {"type":"chitchat","text":"你好，我是 SU7 车载语音助手，很高兴为你服务。","trace":{"route":"Chitchat",...}}

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"我饿了\"}"
# → {"type":"clarification","text":"我还不太确定你的意图...","trace":{"route":"Unknown",...}}
```

Mock 模式各组件行为：

| 组件 | 行为 | 示例 |
|------|------|------|
| 意图分类 | 疑问/祈使语气规则 | 规则匹配，<1ms |
| 技能执行 | 7个技能模板 | "请导航到公司" → `"已开始导航到公司。"` |
| RAG 检索 | 本地 su7_docs.json (5条) TF打分 | 返回内容+正确source+page |
| 天气 DM | 硬编码模板 | `"北京今天天气：晴，18~25℃"` |
| 闲聊 | 固定模板 | `"你好，我是 SU7 车载语音助手"` |
| 联网搜索 | mock 预设hint | WebSearchClient 返回预定义内容 |
| NLG | 原样返回 | 不调用LLM |
| 拒识 | 默认通过 | 不调用外部服务 |

切到生产只需 `.env` 中改一行：`LLM_PROVIDER=doubao`。

---

## 推理流程

### Task 路径

```
用户输入
  ├─ Query 改写 (LLM指代消解 + 25%字符重叠安全校验)
  ├─ 仲裁分类 → Task(A)
  │
  ├─ BERT 意图识别 (439类, Top-5)
  ├─ 过滤 455 函数定义 → Top-5 候选
  ├─ LLM Function Calling (选择最佳函数 + 提取槽位)
  ├─ 槽位归一化 (position映射、extreme提取、percentage转换)
  ├─ DM 处理器 (maps/music/weather)
  ├─ NLG 润色 ("你是一个有用的车载语音助手...")
  └─ task_result
```

### FAQ 路径

```
用户输入
  ├─ Query 改写
  ├─ 仲裁分类 → FAQ(B)
  │
  ├─ RAG 检索
  │   ├─ BM25 Recall (top-15, jieba+停用词)
  │   ├─ Milvus Hybrid (top-40, BGE-Large + SPLADE v2, WeightedRanker)
  │   ├─ WRRF Fusion (weights=[0.7,0.7], k=60)
  │   └─ MiniCPM Rerank (top-12)
  ├─ synthesize_with_citations
  └─ faq_answer + citations[{source, page}]
```

### Chitchat 路径

```
用户输入
  ├─ Query 改写
  ├─ 仲裁分类 → Chitchat(C/D)
  │
  ├─ 拒识模型 (REJECT_URL:8007)
  │   ├─ 不拒识 → 继续
  │   ├─ 拒识 + 关联到上轮 → 继续
  │   └─ 拒识 + 不关联 → clarification
  ├─ 联网搜索 (SerpAPI→Serper→Bing→Doubao 四级回退)
  ├─ LLM 整合搜索结果
  └─ chitchat
```

### Mock vs 生产

| 组件 | Mock | 生产 |
|------|------|------|
| 意图分类 | 疑问/祈使语气规则 | BERT(439类) / Doubao LLM仲裁 |
| NLU | 关键词匹配 | BERT Top-5 + LLM Function Calling |
| 技能执行 | 7个模板 | 455函数 + slot + DM + NLG |
| RAG 检索 | su7_docs.json TF打分 | BM25+Milvus→WRRF→MiniCPM |
| 闲聊生成 | 模板 | Doubao/vLLM 流式生成 |
| 联网搜索 | mock预设 | SerpAPI/Serper/Bing/Doubao回退 |
| NLG | 原样返回 | LLM转自然语言 |
| 拒识 | 默认通过 | REJECT_URL(8007) BERT模型 |
| 会话 | 内存dict | Redis |

---

## 训练流程

### 整体流水线

```
数据准备
  ├─ build_index.py → BM25 / Milvus / FAISS 索引
  ├─ generate_data.py --step qa → QA对生成(~823条)
  ├─ generate_data.py --step filter → 质量过滤 + 缩写扩展
  └─ generate_data.py --step dataset → Summary + Rerank 训练集

Agent 训练（CarVoice_Agent 原始流程）
  ├─ BERT 意图模型: train_intent.py (31w条)
  │   → models/saved/intent/bert.ckpt (Acc@1=85.2%, Acc@5=97.6%)
  └─ BERT 拒识模型: train_reject.py (32w条)
      → models/saved/reject/bert_tiny.ckpt (Acc=89.7%)

三分类训练（Task/FAQ/Chitchat）
  └─ train_3class.py → 自动构建数据集 → BERT训练 → ~90%

RAG 训练（XIAOMI_SU7_RAG 原始流程）
  ├─ download_models.py --preset rag → BGE, SPLADE, MiniCPM, Qwen3-8B
  ├─ LLaMA-Factory SFT (configs/sft.yaml) → QLoRA 4-bit, LoRA rank=16
  └─ INT4量化 → 吞吐 465→669 token/s (+43.8%)

RL 训练（Search-R1 + WebWalker）
  ├─ data_builder.py → 轨迹生成 (500网络兜底 + 全量本地)
  ├─ format_converter.py → SFT/GRPO格式 + 标签修复
  ├─ rebalance_sft_data.py → web 33% : local 67%
  ├─ train_grpo.py → SFT warmup + GRPO (TRL, 6维奖励)
  └─ verify_export.py → 导出验证

评估
  ├─ run_agent.py --eval → Agent 端到端 88.6%
  ├─ eval_rag.py → 语义+关键词 0.8965, RAGas 0.94
  └─ batch_eval.py → RL vs baseline
```

### 各步骤命令

```bash
# === Agent 训练 ===

# 下载模型
python scripts/download_models.py --preset agent

# 意图分类训练
python scripts/train_intent.py

# 拒识模型训练
python scripts/train_reject.py

# 三分类训练
python scripts/train_3class.py --train

# === RAG 训练 ===

# 下载模型
python scripts/download_models.py --preset rag

# 构建索引
python scripts/build_index.py --backend all

# 生成数据
python scripts/generate_data.py --step all

# SFT 微调 (需 LLaMA-Factory)
cd LLaMA-Factory-main
llamafactory-cli train ../configs/sft.yaml
```

---

## 生产验证

### 硬件要求

| 用途 | 显卡 | 显存 | 存储 |
|------|------|------|------|
| 推理(INT4) | RTX 3060 | ≥12GB | ~30GB |
| 推理(FP16) | RTX 3090/4090 | ≥24GB | ~50GB |
| LoRA 训练 | RTX 3090/4090 | ≥24GB | ~50GB |
| 全量训练/GRPO | A100 | ≥40GB | ~100GB |

生产者启动：

```bash
# 完整依赖
pip install -r requirements.txt
pip install rank-bm25 jieba sentence-transformers faiss-gpu pymilvus transformers torch vllm

# 下载模型
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --preset core

# 配置
cp .env.example .env
# 编辑: LLM_PROVIDER=vllm, RETRIEVER_BACKEND=hybrid, RERANKER_BACKEND=minicpm

# 终端1: vLLM
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 --gpu-memory-utilization 0.90

# 终端2: 融合服务
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 验证清单

```bash
# 1. 硬件确认
nvidia-smi
curl http://127.0.0.1:8000/v1/models     # vLLM 就绪

# 2. 构建索引
python scripts/build_index.py --backend all

# 3. RAG 检索验证
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" -d "{\"query\":\"SU7 续航\",\"top_k\":3}"

# 4. LLM 生成验证（确认非模板文本）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" -d "{\"message\":\"你好\"}"

# 5. RAG 评估
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --dry-run

# 6. vLLM 性能
python scripts/benchmark_vllm.py --url http://127.0.0.1:8000/v1
```

| # | 验证项 | Mock | 生产 |
|---|--------|------|------|
| 1 | Task 路由 | ✅ 模板 | ✅ BERT+LLM真实执行 |
| 2 | FAQ 路由 | ✅ 本地文档+source/page | ✅ Milvus+LLM |
| 3 | Chitchat 路由 | ✅ 模板 | ✅ vLLM生成 |
| 4 | 天气→Task | ✅ 路由正确 | ✅ DM天气执行 |
| 5 | 隐含指令 | ✅ clarification | ✅ LLM仲裁 |
| 6 | 高风险二次确认 | ✅ | ✅ |
| 7 | RAG 检索 | ✅ TF打分 | ✅ BM25+Milvus+WRRF+MiniCPM |
| 8 | RAG 引用(source+page) | ✅ | ✅ |
| 9 | LLM 生成 | — | ✅ |
| 10 | vLLM 性能 | — | ✅ |
| 11 | RAG 评估 | — | ✅ |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话 |
| `GET` | `/api/v1/skills` | 技能白名单 (7个) |
| `GET` | `/api/v1/functions` | 函数定义 (455个) |
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
    "route": "Task",
    "classifier_confidence": 0.9,
    "knowledge_hit_count": null,
    "latency_ms": 1,
    "fallback_reason": null,
    "risk_level": "medium",
    "session_id": "uuid",
    "rewritten_query": "请导航到公司"
  },
  "session_id": "uuid"
}
```

`type`: `task_result` | `faq_answer` | `chitchat` | `clarification` | `error`

### WebSocket

```
ws://127.0.0.1:8080/ws/chat
→ {"message":"请播放音乐"}
← {"type":"task_result",...}

→ {"message":"请关闭安全系统"}
← {"type":"clarification","trace":{"fallback_reason":"high_risk_needs_confirmation"},...}
→ {"message":"确认执行","confirm":true,"session_id":"xxx"}
← {"type":"task_result",...}
```

---

## 配置项

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | mock | LLM后端：mock / doubao / vllm / openai |
| `DOUBAO_API_KEY` | — | 豆包 API Key |
| `VLLM_BASE_URL` | http://127.0.0.1:8000/v1 | vLLM 推理地址 |
| `RETRIEVER_BACKEND` | mock | mock / bm25 / milvus / hybrid |
| `RERANKER_BACKEND` | mock | mock / minicpm |
| `HYBRID_DENSE_BACKEND` | milvus | hybrid 向量后端：milvus / faiss |
| `WEB_SEARCH_ENABLED` | false | 联网搜索开关 |
| `NLU_URL` | — | 外部 NLU 服务地址 |
| `REJECT_URL` | — | 外部拒识服务地址 |
| `REDIS_URL` | — | Redis 连接 (无则内存) |
| `AMAP_API_KEY` | — | 高德地图 API Key |

完整见 `.env.example`。

---

## 项目验证

```bash
# 单元测试（63 passed）
pytest -q -v

# Agent 流水线测试
python scripts/run_agent.py -i                          # 交互
python scripts/run_agent.py --query "打开空调"           # 单条
python scripts/run_agent.py --file data/nlu/multi_test.txt  # 批量
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 评测

# RAG 检索
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" -d "{\"query\":\"SU7 续航\",\"top_k\":3}"
```

---

## 脚本对照

| 原始项目 | 原始脚本 | 融合脚本 |
|---------|---------|---------|
| CarVoice_Agent | download_models.py | scripts/download_models.py |
| CarVoice_Agent | server.sh | scripts/start_all_services.sh |
| CarVoice_Agent | dialog.py / test.py | scripts/run_agent.py |
| CarVoice_Agent | train/run.py | scripts/train_intent.py / train_reject.py |
| XIAOMI_SU7_RAG | build_index.py | scripts/build_index.py |
| XIAOMI_SU7_RAG | generate_all_data.py | scripts/generate_data.py |
| XIAOMI_SU7_RAG | final_score.py | scripts/eval_rag.py |
| XIAOMI_SU7_RAG | infer.py / infer_rl.py | POST /api/v1/chat / scripts/rl_infer.py |
| XIAOMI_SU7_RAG | deploy/* | scripts/{run_vllm,benchmark_vllm,baseline_compare,download_models}.py |

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/
│   ├── main.py                      # FastAPI 入口
│   ├── api/                         # HTTP + WebSocket 网关
│   ├── core/                        # 主控编排
│   │   ├── orchestrator.py          # 中央调度 (改写→分类→路由→响应)
│   │   ├── classifier.py            # 三级分类 (BERT→规则→LLM仲裁)
│   │   └── session.py               # 会话管理 (Redis/内存)
│   ├── nlp/                         # NLP 管道 (CarVoice client/ 全部移植)
│   │   ├── arbitration.py           # LLM仲裁 (A/B/C/D, 182行Prompt)
│   │   ├── intent.py                # BERT 意图识别 (439类)
│   │   ├── rewrite.py               # 查询改写 (指代消解 + 安全校验)
│   │   ├── nlu.py                   # NLU 意图槽位提取
│   │   ├── nlg.py                   # NLG 自然语言生成
│   │   ├── reject.py                # 拒识模型
│   │   └── correlation.py           # 多轮关联判断
│   ├── skills/                      # 技能执行
│   │   ├── definitions.py           # 455 函数定义
│   │   ├── registry.py              # 白名单注册表 (7技能 + risk_level)
│   │   ├── slot_processor.py        # 槽位归一化
│   │   └── dm/                      # DM 处理器 (maps/music/weather)
│   ├── knowledge/                   # 知识 RAG (SU7_RAG src/ 全部移植)
│   │   ├── retriever/ (8个)         # BM25, FAISS, Milvus, Hybrid, TF-IDF, Qwen3
│   │   ├── reranker/ (5个)          # MiniCPM, BGE-M3, JinaV2, Qwen3, Qwen3-vLLM
│   │   ├── parser/                  # PDF 解析
│   │   ├── generator.py             # LLM 答案生成
│   │   ├── synthesizer.py           # 引用拼装
│   │   ├── chunker.py               # 语义分块
│   │   ├── web_search.py            # Web 垂直搜索
│   │   ├── semantic_chunk_server.py # 语义切分服务 (端口6000)
│   │   └── manual_store.py          # MongoDB 手册存储
│   ├── llm/                         # LLM 抽象层
│   │   ├── doubao.py                # 豆包 (CarVoice 原始技术栈)
│   │   ├── vllm.py                  # vLLM (SU7_RAG 原始技术栈)
│   │   ├── openai_client.py         # OpenAI 兼容
│   │   └── mock.py                  # Mock 开发
│   ├── mcp/                         # MCP 协议 (Amap 13工具 + QQ音乐)
│   ├── prompts/                     # 7个 System Prompt (逐字移植)
│   ├── train/                       # BERT 训练框架 (CarVoice train/ 全部移植)
│   │   ├── core/                    # 自定义 BERT (modeling 1206行 + tokenization 400行)
│   │   ├── models/bert_tiny.py      # BERT Tiny
│   │   ├── servers.py               # 推理服务 (intent:8008, reject:8007, NLU:8009)
│   │   └── run.py                   # 训练入口
│   ├── rl/                          # RL 模块 (SU7_RAG src/rl/ 全部移植, 12文件)
│   ├── data_pipeline/               # 数据管道 (QA生成/过滤/缩写/训练集)
│   ├── eval/                        # 评估框架 (语义+关键词 + RAGas)
│   └── shared/                      # 共享层 (schemas/config/logging/redis/WRRF)
├── scripts/                         # 执行脚本 (17个)
│   ├── autodl_start.sh              # AutoDL 一键启动
│   ├── start.sh / start.ps1         # 本地启动
│   ├── start_all_services.sh        # 5服务并行启动
│   ├── build_index.py               # RAG 索引构建
│   ├── generate_data.py             # 训练数据生成
│   ├── eval_rag.py                  # RAG 离线评估
│   ├── run_agent.py                 # Agent 测试 (交互/批量/评测)
│   ├── download_models.py           # 模型下载 (Agent + RAG)
│   ├── run_vllm.py                  # vLLM 自动启动
│   ├── benchmark_vllm.py            # vLLM 性能压测
│   ├── baseline_compare.py          # 基线对比
│   ├── rl_infer.py                  # RL 推理入口
│   ├── train_intent.py              # 意图模型训练
│   ├── train_reject.py              # 拒识模型训练
│   └── train_3class.py              # 三分类模型训练
├── configs/                         # 训练配置 (sft/grpo)
├── data/                            # 手工整理源数据 (21个文件)
│   ├── knowledge/                   # su7_docs.json + Xiaomi_SU7_Manual.pdf
│   ├── nlu/                         # slot_intent(437) + intent_map(439) + 测试语料
│   ├── training/                    # intent/reject 训练集 + 停用词 + 闲聊样本
│   └── abbr/                        # 汽车缩写表 (53条)
├── tests/                           # 63个测试用例
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
