# SU7_CarVoice_Fusion 架构文档

## 总体架构

```
┌─────────────────────────────────────────────┐
│                  Gateway                     │
│         HTTP (POST /api/v1/chat)            │
│         WebSocket (ws /ws/chat)              │
│         Health (GET /healthz)                │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              Orchestrator                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │Classifier│  │  Router  │  │  Session  │ │
│  │ 意图分类  │  │ 路由决策  │  │  会话管理  │ │
│  └──────────┘  └────┬─────┘  └───────────┘ │
└──────────────────────┼──────────────────────┘
          ┌────────────┼────────────┐
          │            │            │
┌─────────▼──┐ ┌──────▼──────┐ ┌──▼──────────┐
│   Skills   │ │  Knowledge  │ │  (Future)    │
│  白名单执行  │ │  RAG + 引用  │ │  LLM Direct  │
└────────────┘ └─────────────┘ └──────────────┘
```

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
     resolve_skill   retrieve()    直接回复
            │             │
     白名单检查      synthesize()
            │             │
      技能执行      返回 citations
            │             │
            ▼             ▼
         统一 ChatResponse
```

## 路由策略

| 意图 | 阈值 | 处理方式 |
|------|------|----------|
| Task | 0.75 | 白名单技能执行 |
| FAQ  | 0.65 | Knowledge RAG 检索 |
| Chitchat | 0.60 | LLM 直答（当前为模板） |
| Unknown | — | 澄清问题 |

## 高风险安全边界

- Task:risk_level=high → 会话内二次确认后执行
- 技能仅走白名单注册表，不允许任意函数调用
- RAG 检索结果不直接触发车辆控制

## 模块分层

```
app/
├── gateway/         # 接口接入层
│   ├── http_api.py  # RESTful 路由
│   └── ws_api.py    # WebSocket 路由
├── orchestrator/    # 编排决策层
│   ├── classifier.py # 意图分类
│   ├── router.py     # 主控路由
│   └── session.py    # 会话状态
├── services/        # 能力服务层
│   ├── skills/      # 任务技能
│   │   ├── registry.py    # 白名单注册表
│   │   └── handlers/      # 技能处理函数（每个技能独立文件）
│   └── knowledge/   # 知识检索
│       ├── models.py       # 数据模型
│       ├── local_store.py  # 本地文档存储
│       ├── retriever.py    # 混合检索
│       ├── synthesizer.py  # 引用拼装
│       ├── web_search.py   # Web垂直搜索
│       └── service.py      # 对外Facade
└── shared/          # 共享基础
    ├── schemas.py   # Pydantic 模型
    ├── config.py    # 配置管理
    ├── logging.py   # 日志
    └── errors.py    # 错误码
```

## 扩展指南

1. **新增技能**：在 `handlers/` 新建文件 → 在 `registry.py` 注册 SkillSpec
2. **替换分类器**：修改 `classifier.py`，保持接口不变
3. **接入真实向量库**：替换 `local_store.py` 的检索实现
4. **接入真实 LLM**：在 `orchestrator/` 增加 LLM 调用层
