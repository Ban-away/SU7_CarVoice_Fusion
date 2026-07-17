# SU7_CarVoice_Fusion

基于 **CarVoice_Agent**（车载实时会话 + 任务技能）与 **XIAOMI_SU7_RAG**（知识检索 + 可溯源回答）的完整融合后端。

---

## 目录

- [快速开始](#快速开始)
  - [方式一：Mock 模式（30 秒启动）](#方式一mock-模式30-秒启动)
  - [方式二：生产模式（含 GPU）](#方式二生产模式含-gpu)
- [项目验证](#项目验证)
  - [单元测试](#单元测试)
  - [Agent 流水线测试](#agent-流水线测试)
  - [RAG 检索测试](#rag-检索测试)
- [RAG 流水线（对应 XIAOMI_SU7_RAG）](#rag-流水线对应-xiaomi_su7_rag)
  - [1. 构建知识库索引](#1-构建知识库索引)
  - [2. 生成训练数据](#2-生成训练数据)
  - [3. 模型训练](#3-模型训练)
  - [4. RAG 评估](#4-rag-评估)
- [Agent 流水线（对应 CarVoice_Agent）](#agent-流水线对应-carvoice_agent)
  - [1. 下载模型](#1-下载模型)
  - [2. 训练模型](#2-训练模型)
  - [3. 启动服务](#3-启动服务)
  - [4. 测试与评测](#4-测试与评测)
- [API 接口](#api-接口)
- [配置项](#配置项)
- [项目结构](#项目结构)

---

## 快速开始

### 方式一：Mock 模式（30 秒启动）

零依赖，无需 GPU、API Key、外部服务：

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

验证：

```bash
# 健康检查
curl http://127.0.0.1:8080/healthz                    # → {"status":"ok"}

# Task（技能执行）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请导航到公司"}'                       # → task_result

# FAQ（RAG 检索 + 引用）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 续航是多少"}'                     # → faq_answer + citations

# Chitchat（闲聊）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好"}'                               # → chitchat

# Unknown（澄清）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"asdfghjkl"}'                         # → clarification
```

### 方式二：生产模式（含 GPU）

```bash
# 环境初始化
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# RAG 检索依赖
pip install rank-bm25 jieba sentence-transformers faiss-gpu
# Milvus（原始技术栈）
pip install pymilvus transformers torch
# vLLM 推理
pip install vllm
# 训练
pip install datasets accelerate peft bitsandbytes

# 国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
python scripts/download_models.py --preset core

# 配置
cp .env.example .env
# 编辑 .env: LLM_PROVIDER=vllm, RETRIEVER_BACKEND=hybrid
```

启动：

```bash
# 终端 1：vLLM 推理
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --max-model-len 4096

# 终端 2：融合服务
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## 项目验证

### 单元测试

```bash
pytest -q -v
# 63 tests passed
```

覆盖：仲裁分类、NLU/NLG/拒识/改写、技能白名单、知识引用、数据管道、评估框架、完整 API 链路。

### Agent 流水线测试

对应 CarVoice_Agent 的 `dialog.py` + `test.py` + `intent_client.py` + `reject_client.py` + `nlu_client.py`：

```bash
# 单条测试
python scripts/run_agent.py --query "请导航到公司"

# 交互模式
python scripts/run_agent.py -i

# 批量测试（从文件，每行一条 query）
python scripts/run_agent.py --file data/nlu/single_test.txt

# 评测模式（统计分类分布）
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt
```

### RAG 检索测试

```bash
# 知识检索 API
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"SU7 电池容量","top_k":3}'
```

---

## RAG 流水线（对应 XIAOMI_SU7_RAG）

### 1. 构建知识库索引

对应 `build_index.py`：PDF 解析 → 文本清洗 → 语义分块 → BM25/Milvus 索引：

```bash
# 构建 BM25 索引（CPU 即可）
python scripts/build_index.py --backend bm25

# 构建全部索引（含 Milvus，需 GPU）
python scripts/build_index.py --backend all

# 指定 PDF
python scripts/build_index.py --pdf data/knowledge/Xiaomi_SU7_Manual.pdf --backend all
```

### 2. 生成训练数据

对应 `generate_all_data.py` + `generate_sft_data.py`：QA 生成 → 过滤 → 训练集构建：

```bash
# 生成 QA 对（需配置 LLM_PROVIDER=doubao 或 vllm）
python scripts/generate_data.py --step qa

# 过滤
python scripts/generate_data.py --step filter --input data/training/qa_pairs/qa_pair.json

# 构建 Summary/Rerank 训练集
python scripts/generate_data.py --step dataset --input data/training/qa_pairs/qa_pair_filtered.json

# 全流程一键
python scripts/generate_data.py --step all
```

### 3. 模型训练

对应 LLaMA-Factory SFT → 导出 → 量化流程。训练配置见 `configs/sft.yaml` 和 `configs/grpo.yaml`：

```bash
# 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory-main
cd LLaMA-Factory-main && pip install -r requirements.txt && pip install -e . && cd ..

# 复制训练数据
cp data/training/summary/train.json LLaMA-Factory-main/data/summary_train.json
cp data/training/summary/test.json LLaMA-Factory-main/data/summary_test.json

# SFT 训练
cd LLaMA-Factory-main
llamafactory-cli train ../configs/sft.yaml

# 导出合并模型
llamafactory-cli export ../configs/sft.yaml
```

### 4. RAG 评估

对应 `final_score.py`：检索 → 生成 → 语义评分 + 关键词加权 + RAGas：

```bash
# 评估（需先生成评估集）
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json

# 快速验证
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --dry-run

# 跳过 RAGas（省 API 费用）
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --skip-ragas
```

---

## Agent 流水线（对应 CarVoice_Agent）

### 1. 下载模型

对应 `download_models.py`：

```bash
python scripts/download_models.py --preset core    # 核心模型（默认）
python scripts/download_models.py --preset agent   # 仅 Agent 模型
python scripts/download_models.py --preset all     # 全部模型
```

模型列表：

| 模型 | 用途 | 预设 |
|------|------|------|
| chinese-roberta-wwm-ext | 意图分类 | agent/core/all |
| roberta_chinese_3L312_clue_tiny | 拒识模型 | agent/core/all |
| BGE-Large-zh-v1.5 | Dense 检索 | rag/core/all |
| SPLADE v2 | Sparse 检索 | rag/core/all |
| MiniCPM Layerwise | 重排序 | rag/core/all |
| Qwen3-8B | 答案生成 | rag/core/all |
| m3e-small | 语义切分 | rag/core/all |
| text2vec-base-chinese | 评估相似度 | rag/core/all |

### 2. 训练模型

对应 `train/run.py`。意图分类（RoBERTa-wwm-ext）和拒识（3 层 BERT Tiny）的训练需要原始 CarVoice_Agent 的训练框架。训练数据和模型配置已包含在本项目中：

- 训练数据：`data/training/intent/`、`data/training/reject/`
- 意图映射：`data/nlu/class_labels.txt`、`data/nlu/intent_map.json`
- 槽位映射：`data/nlu/slot_intent.json`

```bash
# 训练流程参考 CarVoice_Agent：
cd train
python run.py --model bert --data intent       # 意图模型
python run.py --model bert_tiny --data reject  # 拒识模型
```

### 3. 启动服务

对应 `server.sh`。融合项目将多服务统一为单入口：

```bash
# Mock 模式（无需外部服务）
uvicorn app.main:app --host 0.0.0.0 --port 8080

# 生产模式（需先启动外部 NLU/拒识/意图服务 + vLLM）
# 配置 .env:
#   NLU_URL=http://127.0.0.1:8009/chatnlu-server/v1
#   REJECT_URL=http://127.0.0.1:8007/reject-server/v1
#   LLM_PROVIDER=vllm
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 4. 测试与评测

```bash
# ---- 功能测试 ----

# 交互式测试
python scripts/run_agent.py -i

# 单条测试
python scripts/run_agent.py --query "打开空调"

# 批量测试
python scripts/run_agent.py --file data/nlu/single_test.txt

# ---- 评测 ----

# Agent 分类评测
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt

# 单元测试（覆盖仲裁/NLU/NLG/拒识/改写/技能白名单）
pytest tests/ -q -v
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话（统一入口） |
| `GET` | `/api/v1/skills` | 技能白名单（7 个） |
| `GET` | `/api/v1/functions` | 函数定义（455 个） |
| `POST` | `/api/v1/knowledge/retrieve` | 知识检索调试 |
| `WS` | `/ws/chat` | WebSocket 实时会话 |

### 请求/响应

```json
// 请求
{"message":"请导航到公司"}

// 响应
{
  "type": "task_result",
  "text": "已开始导航到公司。",
  "citations": [],
  "trace": {
    "route": "Task",
    "classifier_confidence": 0.9,
    "latency_ms": 1,
    "fallback_reason": null,
    "risk_level": "medium"
  },
  "session_id": "xxx-xxx"
}
```

`type` 取值：`task_result` | `faq_answer` | `chitchat` | `clarification` | `error`

### WebSocket

```
ws://127.0.0.1:8080/ws/chat
→ {"message":"请播放音乐"}
← {"type":"task_result","text":"已执行媒体控制指令。",...}

→ {"message":"请关闭安全系统"}
← {"type":"clarification","text":"该操作风险较高...","trace":{"fallback_reason":"high_risk_needs_confirmation"}}

→ {"message":"确认执行","confirm":true,"session_id":"上次session_id"}
← {"type":"task_result","text":"高风险车辆控制指令已接收..."}
```

---

## 配置项

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | mock | LLM 后端：mock/doubao/vllm/openai |
| `DOUBAO_API_KEY` | — | 豆包 API Key（CarVoice 原始技术栈） |
| `VLLM_BASE_URL` | http://127.0.0.1:8000/v1 | vLLM 推理地址 |
| `RETRIEVER_BACKEND` | mock | 检索后端：mock/bm25/milvus/hybrid |
| `RERANKER_BACKEND` | mock | 重排后端：mock/minicpm |
| `HYBRID_DENSE_BACKEND` | milvus | hybrid 向量后端：milvus/faiss |
| `NLU_URL` | — | NLU 服务地址（CarVoice 原始技术栈） |
| `REJECT_URL` | — | 拒识服务地址 |
| `REDIS_URL` | — | Redis 连接（无则内存） |
| `AMAP_API_KEY` | — | 高德地图 API Key |
| `TASK_CONFIDENCE_THRESHOLD` | 0.75 | Task 路由阈值 |
| `FAQ_CONFIDENCE_THRESHOLD` | 0.65 | FAQ 路由阈值 |

完整见 `.env.example`。

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/
│   ├── main.py                    # FastAPI 入口
│   ├── api/                       # HTTP + WebSocket 网关
│   │   ├── http_routes.py
│   │   └── ws_routes.py
│   ├── core/                      # 主控编排
│   │   ├── orchestrator.py        # 中央调度（路由决策）
│   │   ├── classifier.py          # 意图分类（关键词 + LLM 仲裁双通道）
│   │   └── session.py             # 会话管理
│   ├── nlp/                       # NLP 管道（对应 CarVoice client/）
│   │   ├── arbitration.py         # 仲裁（A/B/C/D → task/faq/chat）
│   │   ├── rewrite.py             # 查询改写（指代消解）
│   │   ├── nlu.py                 # NLU 意图槽位提取
│   │   ├── nlg.py                 # NLG 自然语言生成
│   │   ├── reject.py              # 拒识模型
│   │   └── correlation.py         # 多轮关联判断
│   ├── skills/                    # 技能执行（对应 CarVoice function_call/）
│   │   ├── definitions.py         # 455 函数定义
│   │   ├── registry.py            # 白名单注册表
│   │   ├── nlu_data.py            # NLU 数据加载
│   │   ├── slot_processor.py      # 槽位归一化
│   │   └── dm/                    # DM 处理器（maps/music/weather）
│   ├── knowledge/                 # 知识 RAG（对应 XIAOMI_SU7_RAG src/）
│   │   ├── retriever/             # BM25/FAISS/Milvus/Hybrid
│   │   ├── reranker/              # MiniCPM 重排
│   │   ├── generator.py           # LLM 答案生成
│   │   ├── synthesizer.py         # 引用拼装
│   │   ├── chunker.py             # 语义分块
│   │   ├── parser/                # PDF 解析
│   │   └── web_search.py          # Web 垂直搜索
│   ├── llm/                       # LLM 抽象层
│   │   ├── base.py                # 基类 + 工厂
│   │   ├── doubao.py              # 豆包（CarVoice 原始技术栈）
│   │   ├── vllm.py                # vLLM（SU7_RAG 原始技术栈）
│   │   └── mock.py                # Mock（开发）
│   ├── mcp/                       # MCP（高德13工具 + QQ音乐）
│   ├── prompts/                   # 7 个 System Prompt
│   ├── data_pipeline/             # 数据管道（QA生成/过滤/训练集构建）
│   ├── eval/                      # 评估框架（语义+关键词+RAGas）
│   └── shared/                    # 共享层（schema/config/logging/redis/WRRF）
├── scripts/
│   ├── build_index.py             # 构建知识库索引
│   ├── generate_data.py           # 生成训练数据
│   ├── eval_rag.py               # RAG 离线评估
│   ├── test_agent.py              # Agent 流水线测试
│   ├── download_models.py         # 模型下载
│   ├── autodl_start.sh            # AutoDL 一键启动
│   ├── start.sh / start.ps1       # 本地启动
├── configs/                       # 训练配置（sft/grpo）
├── data/                          # 手工整理源数据
│   ├── knowledge/                 # PDF + su7_docs.json
│   ├── nlu/                       # 意图/槽位映射 + 测试语料
│   ├── training/                  # 意图/拒识训练集 + 停用词 + 闲聊样本
│   └── abbr/                      # 汽车术语缩写表
├── tests/                         # 63 个测试用例
├── docs/architecture.md
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

### 执行脚本对照

| 原始项目脚本 | 融合项目对应 |
|------------|------------|
| `XIAOMI_SU7_RAG/build_index.py` | `scripts/build_index.py` |
| `XIAOMI_SU7_RAG/generate_all_data.py` | `scripts/generate_data.py --step qa/filter` |
| `XIAOMI_SU7_RAG/generate_sft_data.py` | `scripts/generate_data.py --step dataset` |
| `XIAOMI_SU7_RAG/final_score.py` | `scripts/eval_rag.py` |
| `XIAOMI_SU7_RAG/infer.py` | `app/core/orchestrator.py` (在线) + `scripts/run_agent.py --query` |
| `CarVoice_Agent/test.py` | `scripts/run_agent.py --file` |
| `CarVoice_Agent/dialog.py` | `scripts/run_agent.py -i` |
| `CarVoice_Agent/server.sh` | `scripts/autodl_start.sh` |
| `CarVoice_Agent/download_models.py` + `XIAOMI_SU7_RAG/deploy/download_models.py` | `scripts/download_models.py` |
