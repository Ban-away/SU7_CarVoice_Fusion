# SU7_CarVoice_Fusion

基于 **CarVoice_Agent + XIAOMI_SU7_RAG** 的融合后端全量骨架，提供可运行闭环（HTTP + WebSocket）与可扩展的会话、技能、知识检索能力。

## 架构说明

- `app/gateway`：统一入口（HTTP + WebSocket）
- `app/orchestrator`：意图分类、路由决策、置信度门控、会话状态与安全边界
- `app/services/skills`：任务型技能执行层（白名单注册）
- `app/services/knowledge`：知识服务（本地文档库加载 + 混合检索 + 可选 web 检索 + 引用拼装）
- `app/shared`：schema、配置、日志、错误处理

## 已实现能力

1. **四分类路由**：Task / FAQ / Chitchat / Unknown
2. **置信度门控与降级**：
   - 高置信度 Task 直达 skills
   - FAQ 优先 knowledge
   - 低置信度或召回不足返回澄清
3. **统一响应结构**：
   - `type`: `task_result` / `faq_answer` / `chitchat` / `clarification` / `error`
   - `text`
   - `citations`
   - `trace`: `route`, `classifier_confidence`, `knowledge_hit_count`, `latency_ms`, `fallback_reason`, `risk_level`
4. **Skills 白名单与风险边界**：
   - 技能集：`media_control`, `navigate_to`, `vehicle_status`, `ac_control`, `window_control`, `charge_management`
   - 高风险技能：`sensitive_vehicle_control`（会话内二次确认后执行）
5. **Knowledge 全量检索链路**：
   - `search_local_docs(query, top_k)`（本地文档库 + 混合打分）
   - `search_web_vertical(query)`（默认关闭，配置开启）
   - `retrieve(query, top_k)`（本地优先 + web 回补）
   - `synthesize_with_citations(...)`（输出 `source` + `page`）
6. **多轮会话融合**：
   - `session_id` 贯穿 HTTP/WS
   - 上下文改写（短追问拼接上一轮语义）
   - 高风险待确认状态在会话内持久化

## 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
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
  "text": "小米 SU7 标准版 CLTC 续航 700km。",
  "citations": [
    {
      "source": "su7_manual.pdf",
      "page": 12
    }
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

### 知识检索调试接口

```bash
curl -X POST http://127.0.0.1:8000/api/v1/knowledge/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"SU7 续航", "top_k": 2}'
```

## WebSocket 测试示例

连接地址：`ws://127.0.0.1:8000/ws/chat`

消息样例（JSON）：

```json
{"message":"请播放音乐"}
```

高风险确认样例（同一 session）：

```json
{"message":"请关闭安全系统"}
{"message":"确认执行","confirm":true,"session_id":"<上次响应中的session_id>"}
```

第一条会返回 `clarification`，并在 `trace.fallback_reason=high_risk_needs_confirmation` 中标记风险确认占位。

## 配置项

见 `.env.example`：

- `TASK_CONFIDENCE_THRESHOLD`
- `FAQ_CONFIDENCE_THRESHOLD`
- `CHITCHAT_CONFIDENCE_THRESHOLD`
- `KNOWLEDGE_TOP_K`
- `WEB_SEARCH_ENABLED`
- `KNOWLEDGE_DOCS_PATH`

## 测试

```bash
pytest -q
```

## Docker Compose

```bash
docker compose up --build
```
