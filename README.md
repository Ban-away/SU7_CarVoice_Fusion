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

## 快速启动

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## HTTP 测试示例

### 健康检查

```bash
curl http://127.0.0.1:8000/healthz
```

### Task 路径

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
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
curl -X POST http://127.0.0.1:8000/api/v1/chat \
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
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"asdfghjkl"}'
```

### 技能白名单元数据

```bash
curl http://127.0.0.1:8000/api/v1/skills
```

### 函数定义（455个）

```bash
curl http://127.0.0.1:8000/api/v1/functions
```

### 知识检索调试

```bash
curl -X POST http://127.0.0.1:8000/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"SU7 续航","top_k":2}'
```

## WebSocket 测试

连接地址：`ws://127.0.0.1:8000/ws/chat`

消息样例：

```json
{"message":"请播放音乐"}
```

高风险确认样例（同一 session）：

```json
{"message":"请关闭安全系统"}
{"message":"确认执行","confirm":true,"session_id":"<上次的session_id>"}
```

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

## 测试

```bash
pytest -q -v
# 61 tests passed
```

## Docker Compose

```bash
docker compose up --build
```

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
│   └── grpo.yaml                     # GRPO RL 训练
├── data/
│   ├── knowledge/su7_docs.json       # 知识库文档
│   └── abbr/abbr_ch.csv              # 汽车术语缩写表
├── tests/                            # 61 个测试用例
├── docs/architecture.md              # 架构文档
├── scripts/                          # 启动脚本
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
