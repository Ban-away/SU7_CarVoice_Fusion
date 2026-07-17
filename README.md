# SU7_CarVoice_Fusion

基于 **CarVoice_Agent**（车载实时会话 + 任务技能 + NLU/NLG/仲裁）与 **XIAOMI_SU7_RAG**（知识检索 + 可溯源回答 + 评估 + RL 训练）的完整融合后端。

---

## 目录

- [架构说明](#架构说明)
- [快速开始](#快速开始)
- [RAG 流水线](#rag-流水线)
- [Agent 流水线](#agent-流水线)
- [API 接口](#api-接口)
- [配置项](#配置项)
- [项目结构](#项目结构)
- [已知限制](#已知限制)

---

## 架构说明

```
用户输入
  │
  ├─ rewrite_with_context (多轮指代消解)
  ├─ classify_intent (关键词快速通道 + LLM 仲裁双通道)
  │
  ├─ Task ── resolve_skill ── execute ── NLG ──▶ task_result
  │   └─ 高风险 ──▶ clarification (等 confirm=true)
  │   └─ 未匹配 ──▶ NLU+DM ──▶ task_result
  │   └─ 都失败 ──▶ chat 回退
  │
  ├─ FAQ ── reject+correlation ── knowledge.retrieve ── synthesize ──▶ faq_answer + citations
  ├─ Chitchat ── reject+correlation ── chat ──▶ chitchat
  └─ Unknown ──▶ clarification
```

### 技术栈

| 层 | 组件 | 来源 |
|---|------|------|
| 检索 | BM25 + Milvus (BGE+SPLADE) → WRRF → MiniCPM | XIAOMI_SU7_RAG |
| 生成 | Qwen3-8B (vLLM) / Doubao / OpenAI | 两者 |
| 仲裁 | 182 行仲裁 Prompt, A/B/C/D 四分类 | CarVoice_Agent |
| NLU | 外部 NLU 服务 + 455 函数定义 + 槽位归一化 | CarVoice_Agent |
| NLG | LLM 工具响应 → 自然语言 | CarVoice_Agent |
| 改写 | LLM 指代消解 + 字符重叠安全校验 | CarVoice_Agent |
| 拒识 | 外部拒识服务 + 多轮关联判断 | CarVoice_Agent |
| 会话 | Redis（内存回退） | CarVoice_Agent |
| MCP | Amap 13 工具 + QQ Music | CarVoice_Agent |

---

## 快速开始

### Mock 模式（零依赖，30 秒启动）

无需 GPU、API Key、外部服务：

```bash
# 克隆
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion

# 环境
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

验证：

```bash
curl http://127.0.0.1:8080/healthz
# → {"status":"ok"}

curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请导航到公司"}'
# → {"type":"task_result","text":"已开始导航到公司。","trace":{"route":"Task",...}}

curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 续航是多少"}'
# → {"type":"faq_answer","citations":[{"source":"su7_manual.pdf","page":12}],...}

curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好"}'
# → {"type":"chitchat",...}

curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"asdfghjkl"}'
# → {"type":"clarification","trace":{"fallback_reason":"low_confidence"},...}
```

### 生产模式（GPU + vLLM）

```bash
# 安装完整依赖
pip install -r requirements.txt
pip install rank-bm25 jieba sentence-transformers faiss-gpu
pip install pymilvus transformers torch
pip install vllm datasets accelerate peft bitsandbytes

# 下载模型（~20GB）
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --preset core

# 配置
cp .env.example .env
# 编辑 .env：
#   LLM_PROVIDER=vllm
#   RETRIEVER_BACKEND=hybrid
#   RERANKER_BACKEND=minicpm

# 终端 1：启动 vLLM
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 --gpu-memory-utilization 0.90

# 终端 2：启动融合服务
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## RAG 流水线

> 对应 XIAOMI_SU7_RAG 的全部流程：建索引 → 生成数据 → 训练 → 评估 → 推理。

### 1. 构建知识库索引

对应 `build_index.py`。

```bash
# 构建 BM25 索引（CPU 即可）
python scripts/build_index.py --backend bm25

# 构建全部索引（BM25 + FAISS + Milvus，需 GPU）
python scripts/build_index.py --backend all

# 指定 PDF
python scripts/build_index.py --pdf data/knowledge/Xiaomi_SU7_Manual.pdf --backend all
```

流程：PDF 解析（PyMuPDF + pdfplumber）→ 文本清洗 → 语义分块（m3e-small）→ BM25 + FAISS/Milvus 索引。

索引产物：

| 文件 | 说明 |
|------|------|
| `data/knowledge/saved_index/bm25retriever.pkl` | BM25 关键词索引 |
| `data/knowledge/saved_index/faiss.db/` | FAISS 向量索引 |
| `data/knowledge/saved_index/milvus.db` | Milvus 混合索引（BGE+SPLADE） |
| `data/knowledge/processed_docs/*.pkl` | 解析/清洗/切分后的文档缓存 |

### 2. 生成训练数据

对应 `generate_all_data.py` + `generate_sft_data.py`。

```bash
# 生成 QA 对（需配置 LLM_PROVIDER=doubao 或 vllm）
python scripts/generate_data.py --step qa

# 质量过滤 + 缩写扩展
python scripts/generate_data.py --step filter --input data/training/qa_pairs/qa_pair.json

# 构建 Summary / Rerank 训练集
python scripts/generate_data.py --step dataset --input data/training/qa_pairs/qa_pair_filtered.json

# 全流程一键
python scripts/generate_data.py --step all
```

产出：

| 文件 | 说明 | 记录数 |
|------|------|--------|
| `qa_pair.json` | 原始 QA 对 | ~823 |
| `expand_qa_pair.json` | 扩展 QA 对（每问题 5 个同义问法） | ~3864 |
| `train_qa_pair.json` | 训练集（质量审核后） | ~21595 |
| `test_qa_pair_verify.json` | 评估输入文件 | ~2325 |
| `summary/train.json` | 摘要训练集 | ~19878 |
| `summary/test.json` | 摘要测试集 | ~1717 |
| `rerank/train.json` | 重排训练集 | ~40849 |

### 3. 模型训练

对应 LLaMA-Factory SFT → 导出 → INT4 量化 → GRPO 强化学习。

```bash
# 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory-main
cd LLaMA-Factory-main
pip install -r requirements.txt && pip install -e . && cd ..

# 复制训练数据
cp data/training/summary/train.json LLaMA-Factory-main/data/summary_train.json
cp data/training/summary/test.json LLaMA-Factory-main/data/summary_test.json

# SFT 训练
cd LLaMA-Factory-main
llamafactory-cli train ../configs/sft.yaml

# 导出合并模型
llamafactory-cli export ../configs/sft.yaml

# （可选）INT4 量化
python awq_quant.py

# （可选）GRPO 强化学习
llamafactory-cli train ../configs/grpo.yaml
```

训练配置：`configs/sft.yaml`（QLoRA 4-bit, LoRA rank=16, 5 epoch）、`configs/grpo.yaml`（LoRA rank=8, num_generations=4）。

训练结果参考（来自原始项目）：

| 指标 | 值 |
|------|------|
| 训练轮数 | 3.0 epoch |
| 训练损失 | 0.3644 |
| 评估损失 | 0.1533 |
| 训练耗时 | 49分41秒 |

### 4. RAG 离线评估

对应 `final_score.py`。

```bash
# 完整评估
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json

# 快速验证（5 条）
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --dry-run

# 跳过 RAGas（省 API 费用）
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json --skip-ragas
```

评分机制：

- **语义相似度**（text2vec-base-chinese）：预测答案与标准答案的向量余弦相似度（权重 0.7）
- **关键词加权**：提取标准答案中的关键词，检查预测答案是否命中（权重 0.3）
- **综合评分**：`max(semantic_score, 0.3 × keyword_score + 0.7 × semantic_score)`
- **短答案补偿**：精确匹配和字符重叠率保底机制
- **RAGas 评估**（可选）：`LLMContextRecall` + `LLMContextPrecisionWithReference`

参考结果（来自原始项目）：

| 指标 | 得分 |
|------|------|
| 语义+关键词 | 0.8965 |
| RAGas context_recall | 0.9386 |
| RAGas context_precision | 0.9488 |

### 5. 在线推理

对应 `infer.py`。融合架构中，在线推理通过 HTTP API：

```bash
# 知识检索
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"SU7 电池容量","top_k":3}'

# 端到端 FAQ 问答（检索 → 生成 → 引用）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 充电需要多长时间"}'
```

### 6. vLLM 性能压测

对应 `deploy/benchmark.py`。

```bash
# 启动 vLLM 服务后
python deploy/benchmark.py
```

参考结果（来自原始项目）：

| 配置 | 吞吐量 |
|------|--------|
| 单卡（非量化） | 465 token/s |
| 单卡（INT4） | 669 token/s (+43.8%) |
| 8 卡（INT4） | ~4,550 token/s |

---

## Agent 流水线

> 对应 CarVoice_Agent 的全部流程：下载模型 → 训练 → 启动服务 → 测试 → 评测 → 压测。

### 1. 下载预训练模型

对应 `download_models.py`。

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 核心模型（Agent + RAG，默认）
python scripts/download_models.py --preset core

# 仅 Agent 模型（意图分类 + 拒识）
python scripts/download_models.py --preset agent

# 全部模型（含备选重排器）
python scripts/download_models.py --preset all
```

Agent 模型清单：

| 模型 | 用途 | 存放路径 |
|------|------|---------|
| chinese-roberta-wwm-ext | 意图分类（RoBERTa-wwm-ext） | `models/chinese_roberta_wwm_ext/` |
| roberta_chinese_3L312_clue_tiny | 拒识模型（3层 BERT Tiny） | `models/roberta_tiny_clue/` |

### 2. 训练分类模型

对应 `train/run.py`。

```bash
# 意图分类模型（RoBERTa-wwm-ext，31w 训练语料）
cd train
python run.py --model bert --data intent

# 拒识模型（BERT-Tiny，32w 训练语料）
python run.py --model bert_tiny --data reject
cd ..
```

训练完成后输出：
- `train/saved/intent/bert.ckpt`
- `train/saved/reject/bert_tiny.ckpt`

训练数据已包含在项目中：

| 数据 | 路径 | 量级 |
|------|------|------|
| 意图训练集 | `data/training/intent/train.txt` | 31w 条 |
| 意图验证集 | `data/training/intent/dev.txt` | — |
| 意图测试集 | `data/training/intent/test.txt` | — |
| 意图分类标签 | `data/training/intent/class.txt` | — |
| 拒识训练集 | `data/training/reject/train.txt` | 32w 条 |
| 拒识验证集 | `data/training/reject/dev.txt` | — |
| 拒识测试集 | `data/training/reject/test.txt` | — |

> **说明**：训练数据来源于线上，由专门的工程团队负责数据采集、清洗与标注。

参考训练结果（来自原始项目）：

**意图识别模型（RoBERTa-wwm-ext）**：

| 指标 | 数值 |
|------|------|
| Accuracy | 85.19% |
| Accuracy@3 | 96.59% |
| Accuracy@5 | 97.62% |
| Precision | 88.88% |
| Recall | 84.60% |
| F1 | 84.24% |

**拒识模型（BERT-Tiny）**：

| 指标 | 数值 |
|------|------|
| Accuracy | 89.71% |
| Precision | 89.65% |
| Recall | 89.74% |
| F1 | 89.69% |

### 3. 启动服务

对应 `server.sh`（5 个微服务端口）。

```bash
# Mock 模式：融合架构单进程启动（无需外部服务）
uvicorn app.main:app --host 0.0.0.0 --port 8080

# 生产模式：先启动外部服务，再启动融合服务
# 需在 .env 中配置 NLU_URL、REJECT_URL、LLM_PROVIDER 等
```

原始 CarVoice_Agent 多服务模型（参考）：

| 服务 | 端口 | 路径 |
|------|------|------|
| Redis | 6379 | 会话缓存 |
| 拒识服务 | 8007 | `POST /reject-server/v1` |
| 意图服务 | 8008 | `POST /intent-server/v1` |
| NLU 服务 | 8009 | `POST /chatnlu-server/v1` |
| 入口服务 | 8080 | Socket 事件 `request_nlu` |

### 4. 测试

对应 `dialog.py`（交互式）+ `test.py`（批量）。

```bash
# 交互式测试
python scripts/run_agent.py -i

# 单条测试
python scripts/run_agent.py --query "打开空调"

# 批量测试（从文件，每行一条）
python scripts/run_agent.py --file data/nlu/single_test.txt

# 多轮测试
python scripts/run_agent.py --file data/nlu/multi_test.txt

# 评测模式（统计分类分布和延迟）
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt
```

输出示例：

```
>>> 打开空调
  type:     task_result
  route:    Task (0.9)
  text:     已执行空调控制。
  latency:  1ms
```

### 5. 模型准确率评测

对应 `test/reject_client.py` + `test/intent_client.py` + `test/nlu_client.py`。

```bash
# 单元测试（覆盖仲裁/NLU/NLG/拒识/改写/技能/知识引用/API）
pytest -q -v
# 63 passed

# Agent 分类评测（测试集统计）
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt
```

参考准确率（来自原始项目）：

| 指标 | 数值 |
|------|------|
| 拒识准确率 | 89.7% |
| 意图 TOP1 准确率 | 85.2% |
| 意图 TOP5 准确率 | 97.6% |
| 意图+槽位联合（intent） | 85.21% |
| 意图+槽位联合（slots） | 90.14% |
| 端到端准确率 | 88.6% |

### 6. QPS 压测

对应 `test/*benchmark*.py`。

```bash
cd test
pip install locust

# 拒识服务
locust -f reject_benchmark.py --host http://127.0.0.1:8007 --headless -u 1000 -r 100 -t 60s

# 意图服务
locust -f intent_benchmark.py --host http://127.0.0.1:8008 --headless -u 1000 -r 100 -t 60s

# NLU 服务（含 LLM 调用）
locust -f nlu_benchmark.py --host http://127.0.0.1:8009 --headless -u 10 -r 5 -t 60s
```

参考 QPS（来自原始项目）：

| 服务 | 并发 | QPS | P50 | P95 | 失败率 |
|------|------|-----|-----|-----|--------|
| 拒识 | 1000 | 377 | 430ms | 770ms | 0% |
| 意图 | 1000 | 145 | 660ms | — | 0% |
| NLU | 10 | — | 1.3s | 4.3s | 0% |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话（任务/FAQ/闲聊/澄清统一入口） |
| `GET` | `/api/v1/skills` | 技能白名单元数据（7 个，含 risk_level） |
| `GET` | `/api/v1/functions` | 全部 455 个函数/工具定义 |
| `POST` | `/api/v1/knowledge/retrieve` | 知识检索调试 |
| `WS` | `/ws/chat` | WebSocket 实时会话 |

### POST /api/v1/chat

请求：

```json
{
  "message": "请导航到公司",
  "confirm": false,
  "session_id": null
}
```

响应：

```json
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

`type` 取值：

| type | 触发条件 |
|------|---------|
| `task_result` | 命中技能白名单，执行成功 |
| `faq_answer` | FAQ 分类，RAG 检索命中，含 `citations` |
| `chitchat` | 闲聊分类，LLM 回复 |
| `clarification` | 低置信度 / 拒识 / 高风险需确认 / 召回不足 |
| `error` | 服务内部异常 |

### WebSocket /ws/chat

```
连接: ws://127.0.0.1:8080/ws/chat

→ {"message":"请播放音乐"}
← {"type":"task_result","text":"已执行媒体控制指令。",...}

→ {"message":"请关闭安全系统"}
← {"type":"clarification","text":"该操作风险较高...","trace":{"fallback_reason":"high_risk_needs_confirmation"}}

→ {"message":"确认执行","confirm":true,"session_id":"上次session_id"}
← {"type":"task_result",...}
```

---

## 配置项

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | mock | LLM 后端：mock / doubao / vllm / openai |
| `DOUBAO_API_KEY` | — | 豆包 API Key（CarVoice 原始技术栈） |
| `DOUBAO_ENDPOINT` | https://ark.cn-beijing.volces.com/api/v3 | 豆包 API 地址 |
| `DOUBAO_MODEL` | — | 豆包模型部署 ID |
| `VLLM_BASE_URL` | http://127.0.0.1:8000/v1 | vLLM 推理地址 |
| `RETRIEVER_BACKEND` | mock | mock / bm25 / milvus / hybrid |
| `RERANKER_BACKEND` | mock | mock / minicpm |
| `HYBRID_DENSE_BACKEND` | milvus | hybrid 模式向量后端：milvus / faiss |
| `MILVUS_URI` | data/knowledge/saved_index/milvus.db | Milvus Lite 路径 |
| `NLU_URL` | — | NLU 服务地址（CarVoice 原始技术栈） |
| `REJECT_URL` | — | 拒识服务地址 |
| `REDIS_URL` | — | Redis 连接串（无则内存） |
| `AMAP_API_KEY` | — | 高德地图 API Key（MCP 地图工具） |
| `TASK_CONFIDENCE_THRESHOLD` | 0.75 | Task 路由置信度阈值 |
| `FAQ_CONFIDENCE_THRESHOLD` | 0.65 | FAQ 路由置信度阈值 |
| `CHITCHAT_CONFIDENCE_THRESHOLD` | 0.60 | Chitchat 路由置信度阈值 |
| `KNOWLEDGE_TOP_K` | 3 | 检索返回文档数 |
| `WEB_SEARCH_ENABLED` | false | Web 垂直搜索开关 |
| `ABBR_CSV_PATH` | data/abbr/abbr_ch.csv | 汽车缩写表路径 |

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
│   │   ├── orchestrator.py        # 中央调度（对应 CarVoice dialog.py）
│   │   ├── classifier.py          # 意图分类（关键词 + LLM 双通道）
│   │   └── session.py             # 会话管理
│   ├── nlp/                       # NLP 管道（对应 CarVoice client/）
│   │   ├── arbitration.py         # LLM 仲裁（A/B/C/D → task/faq/chat）
│   │   ├── rewrite.py             # 查询改写（指代消解 + 安全校验）
│   │   ├── nlu.py                 # NLU 意图槽位提取
│   │   ├── nlg.py                 # NLG 工具响应 → 自然语言
│   │   ├── reject.py              # 拒识模型
│   │   └── correlation.py         # 多轮关联判断
│   ├── skills/                    # 技能执行（对应 CarVoice function_call/）
│   │   ├── definitions.py         # 455 函数定义（对应 function.py）
│   │   ├── registry.py            # 白名单注册表（7 技能 + risk_level）
│   │   ├── nlu_data.py            # NLU 数据加载
│   │   ├── slot_processor.py      # 槽位归一化（对应 slot_process.py）
│   │   └── dm/                    # DM 处理器
│   │       ├── factory.py         # DM 工厂
│   │       ├── maps.py            # 导航（Go_POI）
│   │       ├── music.py           # 音乐（Search_Music）
│   │       └── weather.py         # 天气（Query_Weather）
│   ├── knowledge/                 # 知识 RAG（对应 XIAOMI_SU7_RAG src/）
│   │   ├── retriever/             # BM25 / FAISS / Milvus / Hybrid 检索器
│   │   ├── reranker/              # MiniCPM 重排序
│   │   ├── generator.py           # LLM 答案生成（对应 llm_local_client.py）
│   │   ├── synthesizer.py         # 引用拼装（对应 post_processing）
│   │   ├── chunker.py             # 语义分块
│   │   ├── parser/                # PDF 解析
│   │   └── web_search.py          # Web 垂直搜索
│   ├── llm/                       # LLM 抽象层
│   │   ├── base.py                # 基类 + 工厂
│   │   ├── doubao.py              # 豆包（CarVoice 原始技术栈）
│   │   ├── vllm.py                # vLLM（SU7_RAG 原始技术栈）
│   │   └── mock.py                # Mock（本地开发）
│   ├── mcp/                       # MCP 协议
│   │   ├── client.py              # MCP 客户端
│   │   ├── amap_server.py         # 高德地图（13 工具）
│   │   └── music_server.py        # QQ 音乐
│   ├── prompts/                   # 7 个 System Prompt
│   ├── data_pipeline/             # 数据管道（对应 gen_qa/ + generate_*.py）
│   │   ├── qa_generator.py        # QA 对生成
│   │   ├── qa_filter.py           # 质量过滤
│   │   ├── abbr_expander.py       # 缩写扩展
│   │   └── dataset_builder.py     # Summary / Rerank 训练集构建
│   ├── eval/                      # 评估框架（对应 final_score.py）
│   │   ├── scorer.py              # 语义 + 关键词加权评分
│   │   └── ragas_eval.py          # RAGas 评估
│   └── shared/                    # 共享层
│       ├── schemas.py             # Pydantic 模型
│       ├── config.py              # 环境配置
│       ├── logging.py             # 日志
│       ├── errors.py              # 错误码
│       ├── redis_client.py        # Redis（内存回退）
│       └── utils.py               # WRRF 融合算法
│
├── scripts/                       # 执行脚本
│   ├── build_index.py             # 构建知识库索引
│   ├── generate_data.py           # 生成训练数据
│   ├── eval_rag.py               # RAG 离线评估
│   ├── run_agent.py               # Agent 流水线测试
│   ├── download_models.py         # 模型下载
│   ├── autodl_start.sh            # AutoDL 一键启动
│   ├── start.sh                   # Linux/Mac 启动
│   └── start.ps1                  # Windows 启动
│
├── configs/                       # 训练配置
│   ├── sft.yaml                   # SFT 训练
│   └── grpo.yaml                  # GRPO 强化学习
│
├── data/                          # 手工整理源数据（不可由代码生成）
│   ├── knowledge/
│   │   ├── Xiaomi_SU7_Manual.pdf  # 小米 SU7 用户手册（源文档，20MB）
│   │   └── su7_docs.json          # 手工编写知识库文档
│   ├── nlu/                       # NLU 配置（手工整理）
│   │   ├── slot_intent.json       # 槽位字段映射（437 条）
│   │   ├── intent_map.json        # 意图 ID → 函数名（439 条）
│   │   ├── class_labels.txt       # 意图分类标签
│   │   ├── single_test.txt        # 单轮测试语料
│   │   └── multi_test.txt         # 多轮测试语料
│   ├── training/                  # 手工标注训练数据
│   │   ├── intent/                # 意图分类（train/dev/test/class）
│   │   ├── reject/                # 拒识分类（train/dev/test/class）
│   │   ├── stopwords.txt          # 中文停用词表
│   │   ├── chats.txt              # 闲聊样本
│   │   ├── raw_general_chats.txt  # 通用对话负样本
│   │   ├── test_docs.txt          # 检索测试文档
│   │   └── test_doc2.txt          # 检索测试文档
│   └── abbr/
│       └── abbr_ch.csv            # 汽车术语缩写表（53 条）
│
├── docs/
│   └── architecture.md            # 架构文档
│
├── tests/                         # 63 个测试用例
│   ├── test_classifier.py         # 分类器（12 条，参数化）
│   ├── test_router.py             # 路由决策（7 条，含高风险防绕过）
│   ├── test_skills_registry.py    # 技能白名单（3 条）
│   ├── test_knowledge_citations.py# 知识引用（1 条）
│   ├── test_gateway_api.py        # API 集成（9 条，含 WebSocket）
│   ├── test_nlp.py                # NLP 管道（10 条）
│   ├── test_data_pipeline.py      # 数据管道（8 条）
│   ├── test_eval.py               # 评估框架（9 条）
│   ├── test_llm.py                # LLM 抽象层（4 条）
│   └── conftest.py                # 共享 fixtures
│
├── .env.example                   # 环境变量模板
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
└── requirements.txt
```

### 脚本对照：原始项目 → 融合项目

| 原始项目 | 原始脚本 | 融合脚本 |
|---------|---------|---------|
| XIAOMI_SU7_RAG | `build_index.py` | `scripts/build_index.py` |
| XIAOMI_SU7_RAG | `generate_all_data.py` | `scripts/generate_data.py --step qa/filter` |
| XIAOMI_SU7_RAG | `generate_sft_data.py` | `scripts/generate_data.py --step dataset` |
| XIAOMI_SU7_RAG | `final_score.py` | `scripts/eval_rag.py` |
| XIAOMI_SU7_RAG | `infer.py` | `POST /api/v1/chat` + `scripts/run_agent.py` |
| XIAOMI_SU7_RAG | `deploy/download_models.py` | `scripts/download_models.py` |
| CarVoice_Agent | `download_models.py` | `scripts/download_models.py` |
| CarVoice_Agent | `server.sh` | `scripts/autodl_start.sh` |
| CarVoice_Agent | `dialog.py` | `scripts/run_agent.py -i` |
| CarVoice_Agent | `test.py` | `scripts/run_agent.py --file` |
| CarVoice_Agent | `*/_client.py, *_benchmark.py` | `scripts/run_agent.py --eval` |
| CarVoice_Agent | `e2e_score.py` | `scripts/run_agent.py --eval` |
| 两者 | pytest | `pytest -q -v` (63 passed) |

---

## 已知限制

| 限制 | 说明 |
|------|------|
| 分类模型训练 | BERT 训练框架（`train/core/`, `train/models/`）未移植，需用原始 CarVoice_Agent 训练 |
| 外部 NLU 服务 | 生产模式需要独立部署的 NLU/拒识/意图微服务（或配置 Mock 模式） |
| Milvus 检索 | 需 `pymilvus` + GPU（BGE+SPLADE 双模型），Mock 模式用 BM25 替代 |
| Qwen3-8B 训练 | SFT 需 24GB+ 显存，GRPO 需 40GB+ |
| 平台差异 | `server.sh` 偏 Linux，Windows 用 `scripts/start.ps1` |
| 网络搜索 API | RL 网络兜底依赖 SerpAPI（免费 100 次/月），未设置时降级为 LLM 模拟 |

---

<p align="center">
  <b>CarVoice_Agent + XIAOMI_SU7_RAG</b><br/>
  <em>Task First + Knowledge RAG — 车载智能语音助手融合架构</em>
</p>
