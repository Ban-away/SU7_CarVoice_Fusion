# SU7_CarVoice_Fusion 架构文档

## 总体架构

```
┌──────────────────────────────────────────────────┐
│                    Gateway                        │
│    FastAPI: POST /api/v1/chat + WS /ws/chat       │
│    Health: GET /healthz                           │
│    Debug: POST /api/v1/knowledge/retrieve         │
│    Info: GET /api/v1/skills /api/v1/functions      │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│                Orchestrator (app/core/)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Classifier│  │  Router  │  │  Session │       │
│  │ 三级分类  │  │ 三路分发  │  │ Redis/内存│       │
│  └──────────┘  └────┬─────┘  └──────────┘       │
└──────────────────────┼───────────────────────────┘
          ┌────────────┼────────────┐
          │            │            │
┌─────────▼──┐ ┌──────▼──────┐ ┌──▼──────────┐
│   Skills   │ │  Knowledge  │ │   Chitchat   │
│ (app/skills)│ │(app/knowledge)│ │ (app/nlp)   │
│ 白名单+DM   │ │ RAG+引用     │ │ 拒识+闲聊    │
└────────────┘ └─────────────┘ └──────────────┘
```

## 核心模块与文件路径

| 模块 | 路径 | 职责 |
|------|------|------|
| API 网关 | `app/api/http_routes.py` `app/api/ws_routes.py` | REST + WebSocket 入口 |
| 编排器 | `app/core/orchestrator.py` | 请求主控，调度分类→路由→处理 |
| 分类器 | `app/core/classifier.py` | BERT→规则→LLM 三级意图分类 |
| 会话 | `app/core/session.py` | Redis/内存会话管理 |
| NLP | `app/nlp/intent.py` `app/nlp/reject.py` `app/nlp/arbitration.py` `app/nlp/rewrite.py` `app/nlp/nlu.py` `app/nlp/nlg.py` | BERT意图、拒识、仲裁、改写、NLU、NLG |
| 技能 | `app/skills/registry.py` `app/skills/dm/` | 白名单注册 + 对话管理 (maps/music/weather) |
| 知识库 | `app/knowledge/service.py` `app/knowledge/retriever/` `app/knowledge/reranker/` | RAG 检索 + 8种retriever + 5种reranker |
| LLM | `app/llm/base.py` `app/llm/doubao.py` `app/llm/vllm.py` `app/llm/mock.py` | 统一 LLM 客户端 (doubao/vllm/openai/mock) |
| MCP | `app/mcp/` | Amap 13工具 + QQ音乐 |
| BERT 训练 | `app/train/run.py` `app/train/models/` `app/train/servers.py` | 模型训练 + 在线推理微服务 |
| RL | `app/rl/` | Search-R1 + GRPO 全流程 |
| 数据管线 | `app/data_pipeline/` | QA生成/过滤/训练集 |
| 评估 | `app/eval/` | scorer + RAGas |

## 请求处理流程

```
用户输入 → Gateway → Orchestrator.handle()
                        │
                 classify_intent()
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        Task          FAQ          Chitchat
     (≥0.75)       (≥0.65)       (≥0.60)
          │             │             │
   resolve_skill   retrieve()    reject_check()
          │             │             │
   白名单检查    synthesize()   web_search()
          │             │             │
    技能执行     返回citations  LLM闲聊
          │             │             │
          ▼             ▼             ▼
       统一 ChatResponse
```

## 路由策略

| 意图 | 阈值 | 处理方式 |
|------|------|----------|
| Task | 0.75 | 白名单技能执行 → DM → NLG |
| FAQ | 0.65 | Knowledge RAG 检索 → 引用拼装 |
| Chitchat | 0.60 | 拒识过滤 → 联网搜索 → 闲聊 |
| Unknown | — | 澄清问题 |

## 三级意图分类

| 级别 | 方案 | 准确率 | 条件 |
|------|------|--------|------|
| 1 | BERT (RoBERTa-wwm-ext) | Top-1 86.07% | 模型已训练 |
| 2 | 启发式规则 | ~90% | BERT 未加载时 |
| 3 | LLM 仲裁 (Doubao) | ~98% | 前两级低置信度时 |

## 知识检索管线

### 离线建库
```
PDF → [可选: 豆包API清洗] → 语义切分(m3e-small)
    → MongoDB存储 → BM25索引 + Milvus向量索引
```

### 在线检索
```
query → [可选: HyDE扩写] → BM25召回(top-15) + Milvus召回(top-40)
      → WRRF融合(weights=[0.7,0.7], k=60)
      → MiniCPM精排(top-12)
      → LLM答案生成
      → citations[{source,page}]
```

## RL 强化学习管线 (Search-R1)

```
问题 → <search_local>检索 → <information>注入
     → model判断: 本地够→<answer> / 不够→<search_web>
     → <read_page> 深度阅读
     → <answer> 最终回答
```

## 服务端口

| 服务 | 端口 | 路径 |
|------|------|------|
| 融合主服务 | 8080 | `/api/v1/chat` |
| 拒识服务 | 8007 | `/reject-server/v1` |
| 意图服务 | 8008 | `/intent-server/v1` |
| NLU 服务 | 8009 | `/chatnlu-server/v1` |
| 语义切分 | 6000 | `/v1/semantic-chunks` |
| vLLM 推理 | 8000 | `/v1/models` |
| Redis | 6379 | — |
| MongoDB | 27017 | — |

## 安全边界

- Task:risk_level=high → 会话内二次确认后执行
- 技能仅走白名单注册表，不允许任意函数调用
- RAG 检索结果不直接触发车辆控制

## 扩展指南

1. **新增技能**：在 `app/skills/` 新建 → `registry.py` 注册 SkillSpec
2. **替换分类器**：修改 `app/core/classifier.py`，保持接口不变
3. **接入真实向量库**：设置 `.env` 中 `RETRIEVER_BACKEND=milvus`
4. **切换 LLM 提供商**：设置 `.env` 中 `LLM_PROVIDER=doubao|vllm|openai`
