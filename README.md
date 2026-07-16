# SU7_CarVoice_Fusion

基于 **CarVoice_Agent**（实时会话与任务路由）与 **XIAOMI_SU7_RAG**（知识检索与可溯源回答）的完整融合后端。

---

## 快速开始

### 方式一：Mock 模式（本地零依赖，30 秒启动）

无需 GPU、无需 API Key、无需外部服务，开箱即验证全部逻辑链路：

```bash
# 1. 克隆
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion

# 2. 虚拟环境
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. 安装依赖（仅基础包，无需 CUDA）
pip install -r requirements.txt

# 4. 配置
cp .env.example .env

# 5. 启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

验证：

```bash
curl http://127.0.0.1:8080/healthz              # → {"status":"ok"}

curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请导航到公司"}'                  # → task_result

curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 续航是多少"}'                # → faq_answer + citations
```

---

### 方式二：AutoDL 云 GPU 完整部署

以下为在 [AutoDL](https://www.autodl.com) 租用 GPU 实例从头到尾跑通全部能力的完整流程。

#### 硬件选择

| 用途 | GPU 建议 | 参考机型 |
|------|---------|---------|
| Mock 验证 / API 调试 | 最低配 | RTX 3060 |
| RAG 检索 + vLLM 推理 | 24GB+ | RTX 3090 / 4090 |
| Qwen3-8B LoRA 训练 | 24GB+ | RTX 3090 / 4090 |
| Qwen3-8B 全量训练 / GRPO | 40GB+ | A100 |

镜像选择：**PyTorch 2.x + CUDA 12.x + Python 3.10/3.11**

#### 第一步：环境初始化

```bash
# 克隆仓库
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 核心依赖
pip install -r requirements.txt

# RAG 检索依赖
pip install rank-bm25 jieba sentence-transformers faiss-gpu

# Milvus 混合检索（原始技术栈）
pip install pymilvus transformers torch

# vLLM 推理
pip install vllm

# 训练依赖
pip install datasets accelerate peft bitsandbytes

# MCP 协议（高德地图 / QQ 音乐）
pip install mcp fastmcp

# HuggingFace 国内镜像（必设）
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型（~20GB，首次运行必需）
#   core — Agent + RAG 核心模型（默认）
#   all  — 含备选重排器
python scripts/download_models.py --preset core

# 配置
cp .env.example .env
vim .env    # 按需修改，Mock 模式可跳过
```

#### 第二步：构建知识库索引

> 对应 XIAOMI_SU7_RAG 的 `build_index.py` 流程。

```bash
# 创建必要目录
mkdir -p data/knowledge/{processed_docs,saved_index/faiss.db,saved_images}
mkdir -p data/training/{qa_pairs,rerank,summary,rl,benchmark}
mkdir -p log models

# 解析 PDF → 清洗 → 语义分块 → 构建索引
python -c "
from app.knowledge.parser.pdf_parser import PDFParser
from app.knowledge.chunker import SemanticChunker
from app.knowledge.retriever.bm25 import BM25Retriever
from app.knowledge.retriever.faiss import FAISSRetriever
import pickle

# 1. 解析 PDF
parser = PDFParser()
pages = parser.parse('data/knowledge/Xiaomi_SU7_Manual.pdf')
print(f'PDF 解析完成: {len(pages)} 页')

# 2. 语义分块
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.split([p['text'] for p in pages if p.get('text')])
print(f'分块完成: {len(chunks)} 个块')

# 3. 构建 BM25 索引
bm25 = BM25Retriever(chunks)
with open('data/knowledge/saved_index/bm25retriever.pkl', 'wb') as f:
    pickle.dump(bm25, f)
print('BM25 索引保存完成')

# 4. 构建 FAISS 索引
faiss = FAISSRetriever(chunks)
faiss.save('data/knowledge/saved_index/faiss.db')
print('FAISS 索引保存完成')
"
```

> **注意**：项目已附带预构建索引（`data/knowledge/saved_index/`），如果已存在可跳过此步骤。Milvus 索引也需要构建（需 GPU）：
>
> ```bash
> # 构建 Milvus 混合索引（BGE-Large-zh-v1.5 + SPLADE v2）
> python -c "
> from app.knowledge.retriever.milvus import MilvusRetriever
> import json
> with open('data/knowledge/su7_docs.json') as f:
>     docs = [item['content'] for item in json.load(f)]
> retriever = MilvusRetriever(docs)
> print(f'Milvus 索引构建完成')
> "
> ```

#### 第三步：生成 QA 训练数据

> 对应 XIAOMI_SU7_RAG 的 `generate_all_data.py` + `generate_sft_data.py` 流程。

```bash
python -c "
from app.data_pipeline.qa_generator import generate_qa_pairs
from app.data_pipeline.qa_filter import filter_qa_pairs
from app.data_pipeline.dataset_builder import build_summary_dataset, build_rerank_dataset
from app.knowledge.service import KnowledgeService
import json

# 1. 生成 QA 对（需 Doubao API 或 Mock）
ks = KnowledgeService()
docs = [d.content for d in ks._documents]
qa_pairs = generate_qa_pairs(docs)
print(f'生成 QA 对: {len(qa_pairs)} 条')

# 2. 质量过滤
clean_pairs = filter_qa_pairs(qa_pairs)
print(f'过滤后保留: {len(clean_pairs)} 条')

# 3. 构建训练数据集
build_summary_dataset(clean_pairs, output_dir='data/training/summary')
print('Summary 数据集构建完成')
"
```

> **注意**：完整 QA 生成需要配置 Doubao API Key（`DOUBAO_API_KEY`）。项目已附带预生成的 QA 数据和训练集（`data/training/qa_pairs/`、`data/training/summary/`、`data/training/rerank/`），如果已存在可跳过此步骤。

#### 第四步：模型训练

> 对应 XIAOMI_SU7_RAG 的 LLaMA-Factory SFT + GRPO 流程。

```bash
# SFT 微调（需 LLaMA-Factory）
cd LLaMA-Factory-main 2>/dev/null || (
    git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory-main
    cd LLaMA-Factory-main
    pip install -r requirements.txt
    pip install -e .
)

# 复制训练数据
cp ../data/training/summary/train.json data/summary_train.json
cp ../data/training/summary/test.json data/summary_test.json

# 启动 SFT 训练
llamafactory-cli train ../configs/sft.yaml

# 导出合并模型
llamafactory-cli export ../configs/sft.yaml

# （可选）GRPO 强化学习训练
# llamafactory-cli train ../configs/grpo.yaml
```

#### 第五步：启动完整服务

```bash
# 终端 1：启动 vLLM 推理服务
vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90

# 终端 2：启动融合服务
source .venv/bin/activate
cp .env.example .env
# 编辑 .env: LLM_PROVIDER=vllm, RETRIEVER_BACKEND=hybrid
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

#### 第六步：验证全链路

```bash
# 健康检查
curl http://127.0.0.1:8080/healthz

# Task 路径（技能执行）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请导航到公司"}'

# FAQ 路径（RAG 检索 + 引用）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 续航是多少"}'

# Chitchat 路径（闲聊）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好"}'

# 技能白名单
curl http://127.0.0.1:8080/api/v1/skills

# 函数定义（455 个）
curl http://127.0.0.1:8080/api/v1/functions

# 知识检索调试
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"SU7 电池容量","top_k":3}'
```

---

## 架构说明

```
api/              统一入口（HTTP + WebSocket）
core/             主控编排（分类、路由、会话）
nlp/              NLP 管道（仲裁、改写、NLU、NLG、拒识、关联）
skills/           技能执行（455 函数定义 + 白名单 + 槽位处理 + DM）
knowledge/        知识 RAG（BM25/Milvus/Hybrid 检索 + MiniCPM 重排 + LLM 生成）
llm/              LLM 抽象层（Doubao / vLLM / OpenAI / Mock）
mcp/              MCP 基础设施（高德地图 13 工具 + QQ 音乐）
prompts/          系统提示词（7 个，逐字移植自 CarVoice_Agent）
data_pipeline/    数据管道（QA 生成 / 过滤 / 缩写扩展 / 训练集构建）
eval/             评估框架（语义 + 关键词加权 + RAGas）
shared/           共享层（schema、配置、日志、Redis、WRRF 融合算法）
```

### 请求处理流程

```
用户输入
  │
  ├─ rewrite_with_context (多轮指代消解)
  │
  ├─ classify_intent (关键词 + LLM 仲裁双通道)
  │
  ├─ Task  → resolve_skill → execute → NLG → task_result
  ├─ FAQ   → reject+correlation → knowledge.retrieve → synthesize → faq_answer + citations
  ├─ Chitchat → reject+correlation → LLM chat → chitchat
  └─ Unknown  → clarification
```

### 技术栈

| 层 | 组件 | 来源 |
|---|------|------|
| 检索 | BM25 + Milvus (BGE+SPLADE) → WRRF 融合 → MiniCPM 重排 | XIAOMI_SU7_RAG |
| 生成 | vLLM / Doubao / OpenAI | 两者 |
| 仲裁 | 182 行仲裁 prompt，Doubao LLM | CarVoice_Agent |
| NLU | 外部 NLU 服务 + 455 函数定义 + 槽位归一化 | CarVoice_Agent |
| 改写 | LLM 指代消解 + 字符重叠安全校验 | CarVoice_Agent |
| 拒识 | 外部拒识服务 + 多轮关联判断 | CarVoice_Agent |
| 会话 | Redis（内存回退）| CarVoice_Agent |
| MCP | 高德地图 13 工具 + QQ 音乐 | CarVoice_Agent |

### 配置驱动

```bash
LLM_PROVIDER=mock      # mock | doubao | vllm | openai
RETRIEVER_BACKEND=mock  # mock | bm25 | milvus | hybrid
RERANKER_BACKEND=mock   # mock | minicpm
HYBRID_DENSE_BACKEND=milvus  # milvus | faiss
```

---

## 配置项

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 后端 | mock |
| `DOUBAO_API_KEY` | 豆包 API Key（CarVoice 原始技术栈） | 空 |
| `VLLM_BASE_URL` | vLLM 推理地址 | http://127.0.0.1:8000/v1 |
| `RETRIEVER_BACKEND` | 检索后端：mock / bm25 / milvus / hybrid | mock |
| `RERANKER_BACKEND` | 重排后端：mock / minicpm | mock |
| `HYBRID_DENSE_BACKEND` | hybrid 模式的向量后端：milvus / faiss | milvus |
| `MILVUS_URI` | Milvus Lite 数据库路径 | data/knowledge/saved_index/milvus.db |
| `REDIS_URL` | Redis 连接（无则内存存储） | 空 |
| `NLU_URL` | NLU 服务地址（CarVoice 原始技术栈） | 空 |
| `REJECT_URL` | 拒识服务地址 | 空 |
| `AMAP_API_KEY` | 高德地图 API Key | 空 |
| `TASK_CONFIDENCE_THRESHOLD` | Task 置信度阈值 | 0.75 |
| `FAQ_CONFIDENCE_THRESHOLD` | FAQ 置信度阈值 | 0.65 |

完整配置见 `.env.example`。

---

## HTTP 接口

### `GET /healthz`
健康检查。

### `POST /api/v1/chat`
单轮请求响应。

```json
{"message":"请导航到公司"}
```

返回：`type` ∈ {task_result, faq_answer, chitchat, clarification, error}，含 `citations[].source` + `citations[].page` 和 `trace`。

### `GET /api/v1/skills`
已注册的技能白名单（7 个），含 risk_level + category + keywords。

### `GET /api/v1/functions`
全部 455 个函数/工具定义。

### `POST /api/v1/knowledge/retrieve`
知识检索调试接口。

```json
{"query":"SU7 续航","top_k":3}
```

## WebSocket

连接地址：`ws://127.0.0.1:8080/ws/chat`

消息样例：

```json
{"message":"请播放音乐"}
```

高风险技能需二次确认：

```json
{"message":"请关闭安全系统"}
{"message":"确认执行","confirm":true,"session_id":"<上次的session_id>"}
```

---

## 测试

```bash
pytest -q -v
# 63 tests passed
```

---

## Docker

```bash
docker compose up --build
```

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/
│   ├── main.py                       # FastAPI 入口
│   ├── api/                          # HTTP + WebSocket 网关
│   ├── core/                         # 主控编排
│   │   ├── orchestrator.py           # 中央调度（对应 CarVoice dialog.py）
│   │   ├── classifier.py             # 意图分类
│   │   └── session.py                # 会话管理
│   ├── nlp/                          # NLP 管道
│   │   ├── arbitration.py            # 仲裁（A/B/C/D → task/faq/chat）
│   │   ├── rewrite.py                # 查询改写（指代消解）
│   │   ├── nlu.py                    # NLU 意图槽位提取
│   │   ├── nlg.py                    # NLG 自然语言生成
│   │   ├── reject.py                 # 拒识模型
│   │   └── correlation.py            # 多轮关联判断
│   ├── skills/                       # 技能执行
│   │   ├── definitions.py            # 455 函数定义（对应 CarVoice function.py）
│   │   ├── registry.py               # 白名单注册表
│   │   ├── nlu_data.py               # NLU 数据加载
│   │   ├── slot_processor.py         # 槽位归一化
│   │   └── dm/                       # DM 处理器（maps/music/weather）
│   ├── knowledge/                    # 知识 RAG（对应 XIAOMI_SU7_RAG src/）
│   │   ├── retriever/                # BM25 / FAISS / Milvus / Hybrid 检索器
│   │   ├── reranker/                 # MiniCPM 重排序
│   │   ├── generator.py              # LLM 答案生成（对应 llm_local_client）
│   │   ├── synthesizer.py            # 引用拼装（对应 post_processing）
│   │   ├── chunker.py                # 语义分块
│   │   └── parser/pdf_parser.py     # PDF 解析
│   ├── llm/                          # LLM 抽象层
│   │   ├── doubao.py                 # Doubao（CarVoice 原始技术栈）
│   │   ├── vllm.py                   # vLLM（SU7_RAG 原始技术栈）
│   │   └── mock.py                   # Mock（本地开发）
│   ├── mcp/                          # MCP 协议
│   │   ├── client.py                 # MCP 客户端
│   │   ├── amap_server.py            # 高德地图（13 工具）
│   │   └── music_server.py           # QQ 音乐
│   ├── prompts/                      # 7 个 System Prompt
│   ├── data_pipeline/                # 数据管道
│   │   ├── qa_generator.py           # QA 对生成
│   │   ├── qa_filter.py              # 质量过滤
│   │   ├── abbr_expander.py          # 缩写扩展
│   │   └── dataset_builder.py        # 训练集构建
│   ├── eval/                         # 评估框架
│   │   ├── scorer.py                 # 语义 + 关键词加权
│   │   └── ragas_eval.py             # RAGas 评估
│   └── shared/                       # 共享层
│       ├── schemas.py / config.py / logging.py
│       ├── redis_client.py           # Redis（内存回退）
│       └── utils.py                  # WRRF 融合算法
├── configs/                          # 训练配置
├── data/
│   ├── knowledge/                    # PDF + 文档 + 索引 + 图片
│   ├── nlu/                          # slot_intent + intent_map + class_labels
│   ├── training/                     # QA / rerank / summary / RL 数据集
│   └── abbr/                         # 汽车术语缩写表
├── tests/                            # 63 个测试用例
├── scripts/                          # 启动脚本（含 autodl_start.sh）
├── docs/architecture.md
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
