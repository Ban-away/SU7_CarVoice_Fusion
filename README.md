# SU7_CarVoice_Fusion

基于 **CarVoice_Agent**（实时会话与任务路由）与 **XIAOMI_SU7_RAG**（知识检索与可溯源回答）的完整融合后端。

## 架构说明

```
api/              统一入口（HTTP + WebSocket）
core/             主控编排（分类、路由、会话）
nlp/              NLP 管道（仲裁、改写、NLU、NLG、拒识、关联）
skills/           技能执行（455 函数定义 + 白名单 + 槽位处理 + DM）
knowledge/        知识 RAG（BM25/FAISS/Hybrid 检索 + MiniCPM 重排 + LLM 生成）
llm/              LLM 抽象层（Mock / Doubao / vLLM / OpenAI）
mcp/              MCP 基础设施（客户端 + 高德地图13工具 + QQ音乐）
prompts/          系统提示词（仲裁/改写/NLG/NLU/闲聊/关联）
data_pipeline/    数据管道（QA生成/过滤/缩写扩展/数据集构建）
eval/             评估框架（自定义评分 + RAGas）
shared/           共享层（schema、配置、日志、Redis、WRRF 融合算法）
```

---

## 本地快速启动（零依赖 Mock 模式）

默认 `LLM_PROVIDER=mock`，无需 GPU，无需外部 API Key，开箱即运行：

```bash
# 1. 克隆仓库
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装依赖（仅基础依赖，Mock 模式不需要 GPU 库）
pip install -r requirements.txt

# 4. 准备配置
cp .env.example .env

# 5. 启动服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## AutoDL 云 GPU 部署指南

以下为在 [AutoDL](https://www.autodl.com) 租用 GPU 实例的完整部署流程。

### 1. 租用实例建议

| 用途 | GPU 建议 | 显存 | 参考机型 |
|------|---------|------|---------|
| Mock 模式 / API 调试 | 最低配 | 任意 | RTX 3060 / 2080Ti |
| vLLM + RAG 检索 | 24GB+ | RTX 3090 / 4090 / A5000 |
| Qwen3-8B 训练 (LoRA) | 24GB+ | RTX 3090 / 4090 / A5000 |
| Qwen3-8B 训练 (全量) | 40GB+ | A100 / 2×RTX 3090 |

镜像建议：**PyTorch 2.x + CUDA 12.x + Python 3.10/3.11**

### 2. 环境初始化

```bash
# ========== 登录 AutoDL 后，打开终端 ==========

# 克隆仓库
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion

# 创建虚拟环境（AutoDL 通常已有 conda）
python -m venv .venv
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip

# ========== 安装核心依赖 ==========
pip install -r requirements.txt

# ========== Mock 模式验证（无需 GPU 库，先确认基础可用）==========
cp .env.example .env
python -c "from app.main import app; print('OK')"
# 输出 OK 即可

# ========== 安装 RAG 检索依赖（生产模式）==========
pip install rank-bm25 jieba
pip install sentence-transformers
pip install faiss-cpu    # 或 faiss-gpu（CUDA 版）

# ========== 安装 LLM 推理依赖（使用 vLLM 时）==========
pip install vllm
pip install transformers

# ========== 安装数据库依赖（可选）==========
pip install redis pymongo

# ========== 安装 MCP 协议依赖（使用高德地图/QQ音乐时）==========
pip install mcp fastmcp
```

### 3. 配置环境变量

```bash
cp .env.example .env
vim .env    # 或 nano .env
```

**AutoDL 上建议配置：**

```bash
# LLM — 使用本地 vLLM（AutoDL 有 GPU）
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8000/v1

# 检索后端 — 使用 BM25（无需额外服务）或 Hybrid（BM25+FAISS）
RETRIEVER_BACKEND=bm25        # bm25 | faiss | hybrid
RERANKER_BACKEND=mock          # 改为 minicpm 以获得更好效果

# 知识库文档路径（指向小米 SU7 手册 PDF）
KNOWLEDGE_DOCS_PATH=data/knowledge/Xiaomi_SU7_Manual.pdf

# 以下按需填写：
# DOUBAO_API_KEY=sk-xxx       # 如果使用豆包 API
# AMAP_API_KEY=xxx            # 如果使用高德地图
# REDIS_URL=redis://127.0.0.1:6379/0
```

### 4. 构建知识库索引

如果使用 BM25/FAISS 检索，先构建索引：

```bash
# 解析 PDF → 清洗 → 分块 → 构建索引
python -c "
from app.knowledge.parser.pdf_parser import PDFParser
from app.knowledge.chunker import SemanticChunker
from app.knowledge.retriever.bm25 import BM25Retriever
from app.knowledge.retriever.faiss import FAISSRetriever
from app.knowledge.retriever.hybrid import HybridRetriever
import pickle

# 1. 解析 PDF
parser = PDFParser()
pages = parser.parse('data/knowledge/Xiaomi_SU7_Manual.pdf')
print(f'Parsed {len(pages)} pages')

# 2. 语义分块
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.split([p['text'] for p in pages if p.get('text')])
print(f'Created {len(chunks)} chunks')

# 3. 构建 BM25 索引
bm25 = BM25Retriever(chunks)
with open('data/knowledge/saved_index/bm25retriever.pkl', 'wb') as f:
    pickle.dump(bm25, f)
print('BM25 index saved')

# 4. 构建 FAISS 索引
faiss = FAISSRetriever(chunks)
faiss.save('data/knowledge/saved_index/faiss.db')
print('FAISS index saved')
"
```

### 5. 启动 vLLM 推理服务（使用 LLM 能力时）

```bash
# 在一个终端启动 vLLM（Qwen3-8B 为例）
# 24GB 显存可跑 4-bit 量化版
vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90

# 等待模型加载完成（看到 "Uvicorn running on http://0.0.0.0:8000" 即可）
```

### 6. 启动融合服务

```bash
# 在另一个终端
cd SU7_CarVoice_Fusion
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 7. 验证服务

```bash
# 健康检查
curl http://127.0.0.1:8080/healthz

# Task 请求
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请导航到公司"}'

# FAQ 请求（RAG 检索 + LLM 生成）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 续航是多少"}'

# 查看已注册的 455 个函数定义
curl http://127.0.0.1:8080/api/v1/functions

# 查看 7 个技能白名单
curl http://127.0.0.1:8080/api/v1/skills

# 直接用知识检索接口（不经过路由）
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"SU7 电池容量","top_k":3}'
```

### 8. 外网访问（AutoDL 自定义服务）

AutoDL 支持通过 SSH 隧道或自定义服务端口暴露：

```bash
# 方法 1：AutoDL 自定义服务（推荐）
# 在 AutoDL 控制台 → 实例 → 自定义服务 → 添加端口 8080
# 然后通过 AutoDL 提供的公网 URL 访问

# 方法 2：SSH 隧道（本地调试时）
# 在本地电脑执行：
ssh -L 8080:127.0.0.1:8080 -p <SSH端口> root@<AutoDL_IP>
# 然后本地访问 http://127.0.0.1:8080
```

---

## AutoDL 一行启动脚本

将以下内容保存为 `scripts/autodl_start.sh`，在 AutoDL 上执行：

```bash
#!/bin/bash
set -e
cd SU7_CarVoice_Fusion
source .venv/bin/activate
cp -n .env.example .env 2>/dev/null || true

# 启动 vLLM（后台）
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 --gpu-memory-utilization 0.90 &
sleep 30  # 等待模型加载

# 启动融合服务
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## HTTP 测试示例

### 健康检查

```bash
curl http://127.0.0.1:8080/healthz
```

### Task 路径

```bash
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请导航到公司"}'
```

返回示例：

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
    "risk_level": "medium"
  }
}
```

### FAQ 路径（含 citations）

```bash
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 续航是多少"}'
```

返回示例（含引用）：

```json
{
  "type": "faq_answer",
  "text": "小米 SU7 标准版 CLTC 续航约 700km。",
  "citations": [
    {"source": "su7_manual.pdf", "page": 12}
  ],
  "trace": {
    "route": "FAQ",
    "classifier_confidence": 0.82,
    "knowledge_hit_count": 1,
    "latency_ms": 1,
    "fallback_reason": null,
    "risk_level": null
  }
}
```

### Unknown 路径（澄清）

```bash
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"asdfghjkl"}'
```

### 技能白名单元数据

```bash
curl http://127.0.0.1:8080/api/v1/skills
```

### 函数定义（455个）

```bash
curl http://127.0.0.1:8080/api/v1/functions
```

### 知识检索调试

```bash
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"SU7 续航","top_k":2}'
```

---

## WebSocket 测试

连接地址：`ws://127.0.0.1:8080/ws/chat`

消息样例：

```json
{"message":"请播放音乐"}
```

高风险确认样例（同一 session）：

```json
{"message":"请关闭安全系统"}
{"message":"确认执行","confirm":true,"session_id":"<上次的session_id>"}
```

---

## 配置项

见 `.env.example`：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 后端：mock / doubao / vllm / openai | mock |
| `RETRIEVER_BACKEND` | 检索后端：mock / bm25 / faiss / hybrid | mock |
| `RERANKER_BACKEND` | 重排后端：mock / minicpm | mock |
| `WEB_SEARCH_ENABLED` | 是否启用 Web 搜索 | false |
| `TASK_CONFIDENCE_THRESHOLD` | Task 路由置信度阈值 | 0.75 |
| `FAQ_CONFIDENCE_THRESHOLD` | FAQ 路由置信度阈值 | 0.65 |
| `CHITCHAT_CONFIDENCE_THRESHOLD` | Chitchat 路由置信度阈值 | 0.60 |
| `REDIS_URL` | Redis 连接串（可选，无则内存存储） | 空 |
| `AMAP_API_KEY` | 高德地图 API Key（MCP 地图工具需要） | 空 |
| `DOUBAO_API_KEY` | 豆包 API Key（使用 Doubao LLM 时） | 空 |

---

## 测试

```bash
pytest -q -v
# 61 tests passed
```

---

## Docker Compose

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
│   │   ├── orchestrator.py           # 中央调度
│   │   ├── classifier.py             # 意图分类
│   │   └── session.py                # 会话管理
│   ├── nlp/                          # NLP 管道
│   │   ├── arbitration.py            # LLM 仲裁（A/B/C/D四分类）
│   │   ├── rewrite.py                # 查询改写（指代消解）
│   │   ├── nlu.py                    # NLU 意图槽位提取
│   │   ├── nlg.py                    # NLG 工具响应转自然语言
│   │   ├── reject.py                 # 拒识模型
│   │   └── correlation.py            # 多轮关联判断
│   ├── skills/                       # 技能执行
│   │   ├── definitions.py            # 455 函数定义（来自 CarVoice_Agent）
│   │   ├── registry.py               # 白名单注册表
│   │   ├── nlu_data.py               # NLU 数据加载器
│   │   ├── slot_processor.py         # 槽位归一化
│   │   └── dm/                       # DM 处理器（maps/music/weather）
│   ├── knowledge/                    # 知识 RAG
│   │   ├── retriever/                # BM25 / FAISS / Hybrid 检索器
│   │   ├── reranker/                 # MiniCPM 重排序
│   │   ├── generator.py              # LLM 答案生成
│   │   ├── synthesizer.py            # 引用拼装
│   │   ├── chunker.py                # 语义分块
│   │   └── parser/                   # PDF 解析
│   ├── llm/                          # LLM 客户端
│   │   ├── base.py                   # 抽象基类 + 工厂
│   │   ├── mock.py                   # Mock 客户端
│   │   ├── doubao.py                 # 豆包（字节）客户端
│   │   └── vllm.py                   # vLLM 客户端
│   ├── mcp/                          # MCP 协议
│   │   ├── client.py                 # MCP 客户端
│   │   ├── amap_server.py            # 高德地图（13 工具）
│   │   └── music_server.py           # QQ 音乐
│   ├── prompts/                      # 系统提示词库
│   ├── data_pipeline/                # 数据管道
│   │   ├── qa_generator.py           # QA 对生成
│   │   ├── qa_filter.py              # 质量过滤
│   │   ├── abbr_expander.py          # 缩写扩展
│   │   └── dataset_builder.py        # 训练集构建
│   ├── eval/                         # 评估框架
│   │   ├── scorer.py                 # 自定义评分
│   │   └── ragas_eval.py             # RAGas 评估
│   └── shared/                       # 共享层
│       ├── schemas.py                # Pydantic 模型
│       ├── config.py                 # 配置
│       ├── redis_client.py           # Redis（内存回退）
│       └── utils.py                  # WRRF 融合算法
├── configs/                          # 训练配置
│   ├── sft.yaml                      # SFT 训练
│   ├── grpo.yaml                     # GRPO RL 训练
│   ├── original_grpo.yaml            # 原始 GRPO 配置（参考）
│   └── original_sft.yaml             # 原始 SFT 配置（参考）
├── data/
│   ├── knowledge/
│   │   ├── Xiaomi_SU7_Manual.pdf     # 小米 SU7 用户手册（20MB）
│   │   ├── su7_docs.json             # 知识库文档
│   │   ├── processed_docs/           # 预处理文档缓存（pkl）
│   │   └── saved_index/              # 预构建检索索引（BM25/FAISS）
│   ├── nlu/                          # NLU 数据（slot_intent + intent_map + class_labels）
│   ├── training/                     # 训练数据（intent/reject/闲聊/QA对）
│   └── abbr/abbr_ch.csv              # 汽车术语缩写表（48条）
├── tests/                            # 61 个测试用例
├── docs/architecture.md              # 架构文档
├── scripts/                          # 启动脚本
│   ├── start.sh                      # Linux/Mac
│   └── start.ps1                     # Windows
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
