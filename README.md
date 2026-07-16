# SU7_CarVoice_Fusion

基于 **CarVoice_Agent + XIAOMI_SU7_RAG** 的融合后端 MVP，提供可运行的最小闭环（HTTP + WebSocket），并保留后续扩展骨架。

## 架构说明

- `app/gateway`：统一入口（HTTP + WebSocket）
- `app/orchestrator`：意图分类、路由决策、置信度门控与安全边界
- `app/services/skills`：任务型技能执行层（白名单注册）
- `app/services/knowledge`：知识服务（本地检索 + 可选 web 检索 + 引用拼装）
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
   - 示例技能：`media_control`, `navigate_to`, `vehicle_status`
   - 高风险技能占位：`sensitive_vehicle_control`（需二次确认）
5. **Knowledge MVP**：
   - `search_local_docs(query, top_k)`（内存 mock）
   - `search_web_vertical(query)`（默认关闭，配置开启）
   - `synthesize_with_citations(...)`（输出 `source` + `page`）

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

### FAQ 路径（含 citations）

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 续航是多少"}'
```

### Unknown 路径（澄清）

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"asdfghjkl"}'
```

## WebSocket 测试示例

连接地址：`ws://127.0.0.1:8000/ws/chat`

消息样例（JSON）：

```json
{"message":"请播放音乐"}
```

高风险确认样例：

```json
{"message":"请关闭安全系统"}
{"message":"请关闭安全系统","confirm":true}
```

## 配置项

见 `.env.example`：

- `TASK_CONFIDENCE_THRESHOLD`
- `FAQ_CONFIDENCE_THRESHOLD`
- `CHITCHAT_CONFIDENCE_THRESHOLD`
- `KNOWLEDGE_TOP_K`
- `WEB_SEARCH_ENABLED`

## 测试

```bash
pytest -q
```

## Docker Compose

```bash
docker compose up --build
```
