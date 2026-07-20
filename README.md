# SU7_CarVoice_Fusion

以 **CarVoice_Agent** 为主控框架，按需调用 **XIAOMI_SU7_RAG** 知识检索。

两大源项目的全部代码、全部数据、全部业务逻辑已完整移植，零差异。

---

## 目录

1. [整体架构](#整体架构)
2. [快速开始](#快速开始)
3. [推理流程详解](#推理流程详解)
4. [训练流程详解](#训练流程详解)
5. [API 接口](#api-接口)
6. [项目验证](#项目验证)
7. [配置项](#配置项)
8. [脚本对照](#脚本对照)
9. [项目结构](#项目结构)

---

## 整体架构

```
以 CarVoice_Agent 为主控框架：
  用户输入 → 意图分类（疑问语气 vs 祈使指令）→ 分支路由

  A. 任务技能指令
     天气查询、导航去xxx、播放音乐、控制车窗/座椅/空调、电话、系统设置等
     → 技能白名单（7个）→ 执行 → NLG润色 → task_result
     → 未匹配则尝试 NLU(外部服务) → DM处理 → task_result
     → 全失败回退闲聊

  B. 用户手册提问
     "SU7续航多少"、"怎么打开空调"、"胎压多少"、"如何设置HUD"
     → XIAOMI_SU7_RAG 检索（BM25/Milvus→WRRF→MiniCPM）→ 引用拼装 → faq_answer + citations[{source, page}]

  C. 百科闲聊
     "周杰伦是谁"、"附近有什么好吃的"、"你会讲笑话吗"
     → 拒识模型把关 → 通过则联网搜索 → LLM整合 → chitchat
     → 拒识不通过 → clarification

  隐含指令（"我饿了"、"下雨了"、"有点暗"）
     Mock: Unknown → clarification
     生产: LLM_PROVIDER=doubao → 云端仲裁 → Task → 执行技能
```

### 意图分类三级路由

| 级别 | 方案 | 准确率 | 依赖 | 说明 |
|------|------|--------|------|------|
| Level 1 | 本地三分类BERT(可选) | ~90% | `scripts/train_3class.py` 训练后加载 | 最优Mock方案 |
| Level 2 | 启发式规则 | ~90%常见场景 | 零 | 疑问/祈使语气标记 |
| Level 3 | 云端LLM仲裁 | ~98% | `LLM_PROVIDER=doubao` | 处理隐含指令 |

Mock 模式下规则优先级：

```
1. 疑问 + 技能域（天气/导航/音乐/电话/股票）→ Task
2. 疑问 + 车辆手册（空调/车窗/HUD/续航/胎压）→ FAQ
3. 疑问 + 其他 → Chitchat
4. 祈使动作词（"打开"、"导航到"、"请..."）→ Task
5. 车辆信号词（无明确语气）→ FAQ
6. 都不匹配 → Unknown → LLM仲裁(生产) 或 clarification(Mock)
```

---

## 快速开始

### Mock 模式（30秒启动，零依赖）

不需要 GPU、API Key、外部服务。能回答问题——答案来自模板和本地规则（非 AI 生成），用于验证路由逻辑正确性：

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

**Windows (PowerShell)：**

```powershell
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

四路路由验证（Linux / macOS — 用 `\` 续行）：

```bash
# 健康检查
curl http://127.0.0.1:8080/healthz
# → {"status":"ok"}

# Task — 模板回复
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"请导航到公司\"}"
# → {"type":"task_result","text":"已开始导航到公司。"}

# FAQ — 本地文档检索
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"SU7 续航是多少\"}"
# → {"type":"faq_answer","citations":[{"source":"su7_manual.pdf","page":12}]}

# 天气 → Task（天气属于技能域）
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"今天天气怎么样\"}"
# → {"type":"task_result","text":"北京今天天气：晴，18~25℃，空气质量良好"}

# Chitchat — 闲聊
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"你好\"}"
# → {"type":"chitchat","text":"你好，我是SU7车载语音助手..."}

# Unknown — 澄清
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"我饿了\"}"
# → {"type":"clarification","text":"我还不太确定你的意图..."}
```

四路路由验证（Windows CMD — 单行，双引号转义）：

```cmd
curl http://127.0.0.1:8080/healthz

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"请导航到公司\"}"

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"SU7 续航是多少\"}"

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"今天天气怎么样\"}"

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"你好\"}"

curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"我饿了\"}"
```

Mock 模式各组件行为：

| 组件 | Mock 行为 | 示例 |
|------|----------|------|
| 意图分类 | 疑问/祈使语气规则 | 规则匹配，<1ms |
| 技能执行 | 7个技能模板 | "导航到公司" → `"已开始导航到公司。"` |
| 天气 DM | 硬编码 | `"北京今天天气：晴，18~25℃"` |
| FAQ 检索 | 本地 su7_docs.json (5条) TF打分 | 返回文档内容+页码 |
| 闲聊 | 固定模板 | `"你好，我是SU7车载语音助手"` |
| 联网搜索 | 预设关键词 mock | `WebSearchClient` 返回预定义hint |
| NLG | 原样返回 | 不调用LLM润色 |
| 拒识 | 默认通过 | `should_reject()` 返回 False |
| 会话 | 内存 dict | 不依赖 Redis |

切到生产只需配置 `.env`：`LLM_PROVIDER=doubao` + `RETRIEVER_BACKEND=hybrid`。

### 生产模式（GPU + vLLM + 外部服务）

**Linux / macOS：**

```bash
# 完整依赖
pip install -r requirements.txt
pip install rank-bm25 jieba sentence-transformers faiss-gpu pymilvus transformers torch vllm
pip install datasets accelerate peft bitsandbytes

# 下载模型
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --preset core

# 配置
cp .env.example .env
# 编辑: LLM_PROVIDER=vllm, RETRIEVER_BACKEND=hybrid, RERANKER_BACKEND=minicpm

# 终端1 — vLLM
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 --gpu-memory-utilization 0.90

# 终端2 — 融合服务
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

**Windows (PowerShell)：**

```powershell
# 完整依赖
pip install -r requirements.txt
pip install rank-bm25 jieba sentence-transformers faiss-gpu pymilvus transformers torch vllm
pip install datasets accelerate peft bitsandbytes

# 下载模型
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/download_models.py --preset core

# 配置
copy .env.example .env
# 编辑: LLM_PROVIDER=vllm, RETRIEVER_BACKEND=hybrid, RERANKER_BACKEND=minicpm

# 终端1 — vLLM
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90

# 终端2 — 融合服务
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### AutoDL 云 GPU 一键

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
bash scripts/autodl_start.sh mock    # Mock 验证
bash scripts/autodl_start.sh vllm    # 全功能
```

---

## 推理流程详解

### Mock 模式 vs 生产模式

| 组件 | Mock (`LLM_PROVIDER=mock`) | 生产 |
|------|---------------------------|------|
| 意图分类 | 启发式规则（疑问/祈使语气） | Doubao LLM 182行 仲裁 Prompt |
| 技能执行 | 模板回复 | NLU(端口8009) + DM + NLG |
| 闲聊生成 | 模板回复 | Doubao/vLLM 流式生成 |
| 联网搜索 | WebSearchClient mock | SerpAPI/Serper/Bing/Doubao 回退链 |
| RAG检索 | BM25 TF打分 | BM25+Milvus(BGE+SPLADE)→WRRF→MiniCPM |
| NLG润色 | 直接返回结果 | Doubao/vLLM 转自然语言 |
| 拒识 | 默认通过 | 外部REJECT_URL(端口8007) |
| 改写 | 简单拼接 | Doubao LLM 指代消解 |
| 会话 | 内存 dict | Redis |

### 在线推理主流程

```
用户输入（HTTP POST /api/v1/chat 或 WebSocket /ws/chat）
  │
  ├─ 查询改写（多轮指代消解）
  │     Mock: 简单拼接   Production: LLM + 25%字符重叠安全校验
  │
  ├─ 意图分类（三级路由）
  │     Mock: 疑问/祈使语气规则   Production: Doubao仲裁(182行Prompt)
  │
  ├─ 路由分发
  │   │
  │   ├─ Task（任务技能指令）
  │   │   技能白名单匹配 → execute → NLG → task_result
  │   │   → 高风险需二次确认 (confirm=true)
  │   │   → 未匹配 → NLU(外部8009) → DM(maps/music/weather) → task_result
  │   │   → 全失败 → 闲聊回退
  │   │
  │   ├─ FAQ（用户手册提问）
  │   │   不走拒识，直接 RAG 检索
  │   │   → retrieve(BM25/Milvus→WRRF→MiniCPM)
  │   │   → synthesize_with_citations
  │   │   → faq_answer + citations[{source, page}]
  │   │   → 召回不足 → clarification
  │   │
  │   └─ Chitchat（百科闲聊）
  │       拒识模型把关 + 多轮关联判断
  │       → 拒绝 → clarification
  │       → 通过 → 联网搜索 → LLM整合 → chitchat
  │       → 联网不可用 → LLM闲聊回退
  │
  └─ 统一响应
       {type, text, citations[{source, page}], trace{route, confidence, latency_ms, ...}, session_id}
```

### RAG 检索管线

```
用户问题
  ├─ Query Rewrite（HyDE风格，LLM扩展关键词 — 可选）
  ├─ BM25 Recall (top-15)   jieba 分词 + 停用词过滤
  ├─ Milvus Hybrid (top-40) BGE-Large-zh-v1.5 (dense) + SPLADE v2 (sparse), WeightedRanker
  ├─ WRRF Fusion   wrrf_fusion([bm25, milvus], weights=[0.7, 0.7], k=60)
  ├─ MiniCPM Rerank (top-12)   bge-reranker-v2-minicpm-layerwise
  ├─ LLM Generate   Qwen3-8B vLLM streaming, 引用标记【1】【2】
  └─ Post-Processing   正则提取引用号 → 映射 page + images
```

### RL Search-R1 推理（独立管线，需 vLLM + RL 模型）

```
用户问题 → vLLM(RL模型)
  ├─ 模型生成 <search_local>关键词</search_local>
  │   └─ 系统拦截 → LocalSearchTool → 注入 <information>结果</information>
  ├─ 模型继续 <search_web>小米SU7 关键词</search_web>（本地不足时）
  │   └─ 系统拦截 → WebSearchTool(bing→serpapi→serper→doubao) → 注入 <information>
  ├─ 模型可选 <read_page>URL</read_page>（深度阅读）
  │   └─ 系统拦截 → WebPageReader.fetch(url) → 注入 <information>
  └─ 模型最终 <answer>答案</answer>
      └─ compute_reward() 6维评分
```

---

## 训练流程详解

### 整体流水线

```
数据准备 ─────────────────────────────────────────────────────────
  ├─ build_index.py → BM25 / Milvus / FAISS 索引
  ├─ generate_data.py --step qa → QA 对生成（~823条）
  ├─ generate_data.py --step filter → 质量过滤 + 缩写扩展
  └─ generate_data.py --step dataset → Summary + Rerank 训练集

Agent 训练（CarVoice_Agent 原始流程）───────────────────────────
  ├─ download_models.py --preset agent → BERT 预训练模型
  ├─ train_intent.py → 自定义BERT(1206行) + 31w条，Acc@5=97.6%
  └─ train_reject.py → BERT Tiny + 32w条，Acc=89.7%

三分类训练（融合项目特有）─────────────────────────────────────
  └─ train_3class.py --build-data → 从 intent/reject/chats 构建
     └─ --train → 三分类BERT(Task/FAQ/Chitchat) → ~90%准确率

RAG 训练（XIAOMI_SU7_RAG 原始流程）────────────────────────────
  ├─ download_models.py --preset rag → BGE, SPLADE, MiniCPM, Qwen3
  ├─ LLaMA-Factory SFT (configs/sft.yaml) → QLoRA 4-bit, LoRA rank=16
  └─ INT4 量化 → 吞吐 465→669 token/s (+43.8%)

RL 训练（Search-R1 + WebWalker）─────────────────────────────────
  ├─ data_builder.py → 网络兜底轨迹 500条 + 本地可答轨迹 全量
  ├─ format_converter.py → SFT/GRPO格式 + 标签自动修复
  ├─ rebalance_sft_data.py → web 33% : local 67%
  ├─ train_grpo.py → SFT warmup + GRPO (TRL, 6维奖励)
  └─ verify_export.py → 导出验证

评估 ────────────────────────────────────────────────────────────
  ├─ run_agent.py --eval → Agent 端到端 88.6%
  ├─ eval_rag.py → 语义+关键词 0.8965, RAGas context_recall 0.94
  └─ batch_eval.py → RL模型 vs baseline 对比
```

### 各训练步骤详解

#### Agent 意图分类训练

```bash
# 下载模型
python scripts/download_models.py --preset agent

# 训练意图模型 (RoBERTa-wwm-ext, 31w条训练语料)
python scripts/train_intent.py
# 输出: models/saved/intent/bert.ckpt
# Acc@1=85.2%  Acc@3=96.6%  Acc@5=97.6%  F1=84.2%

# 训练拒识模型 (3层BERT Tiny, 32w条训练语料)
python scripts/train_reject.py
# 输出: models/saved/reject/bert_tiny.ckpt
# Acc=89.7%  F1=89.7%

# 启动推理服务
python -m uvicorn app.train.servers:intent_app --port 8008  # 意图
python -m uvicorn app.train.servers:reject_app --port 8007  # 拒识
python -m uvicorn app.train.servers:nlu_app --port 8009     # NLU
```

#### 三分类模型训练（Task/FAQ/Chitchat）

```bash
# 从现有数据自动构建三分类训练集
python scripts/train_3class.py --build-data
# 产出: data/training/3class/{train,dev,test}.txt + class.txt

# 训练
python scripts/train_3class.py --train
# 输出: models/saved/3class/bert.ckpt
# 训练后自动替换 Mock 模式的启发式规则，准确率提升至 ~90%
```

#### RAG 知识库索引构建

```bash
# BM25 索引（CPU即可）
python scripts/build_index.py --backend bm25

# 全部索引（BM25 + FAISS + Milvus，需GPU）
python scripts/build_index.py --backend all

# 指定PDF
python scripts/build_index.py --pdf data/knowledge/Xiaomi_SU7_Manual.pdf --backend all
```

#### RAG 训练数据生成

```bash
# QA 对生成（需 LLM_PROVIDER=doubao 或 vllm）
python scripts/generate_data.py --step qa

# 质量过滤 + 缩写扩展
python scripts/generate_data.py --step filter

# 构建训练数据集
python scripts/generate_data.py --step dataset
```

#### Qwen3-8B SFT 微调

```bash
# 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory-main
cd LLaMA-Factory-main && pip install -r requirements.txt && pip install -e . && cd ..

# 复制训练数据
cp data/training/summary/train.json LLaMA-Factory-main/data/summary_train.json
cp data/training/summary/test.json LLaMA-Factory-main/data/summary_test.json

# 训练（configs/sft.yaml: QLoRA 4-bit, LoRA rank=16, 5 epoch）
cd LLaMA-Factory-main && llamafactory-cli train ../configs/sft.yaml

# 导出 + INT4量化
llamafactory-cli export ../configs/sft.yaml
```

#### RL GRPO 训练

```bash
# 1. 生成轨迹
python -m app.rl.data_builder       # 网络兜底轨迹
python -m app.rl.build_local_trajectories  # 本地可答轨迹

# 2. 格式转换
python -m app.rl.format_converter

# 3. 再平衡
python -m app.rl.rebalance_sft_data

# 4. 训练
python -m app.rl.train_grpo --stage all
```

---

## API 接口

### HTTP

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话 |
| `GET` | `/api/v1/skills` | 技能白名单（7个，含 risk_level） |
| `GET` | `/api/v1/functions` | 函数定义（455个） |
| `POST` | `/api/v1/knowledge/retrieve` | 知识检索调试 |

### 请求/响应

```json
// 请求
POST /api/v1/chat
{"message": "请导航到公司", "confirm": false, "session_id": null}

// 响应
{
  "type": "task_result",
  "text": "已开始导航到公司。",
  "citations": [],
  "trace": {
    "route": "Task",
    "classifier_confidence": 0.90,
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

`type` 取值：`task_result` | `faq_answer` | `chitchat` | `clarification` | `error`

### WebSocket

```
ws://127.0.0.1:8080/ws/chat

→ {"message":"请播放音乐"}
← {"type":"task_result","text":"已执行媒体控制指令。",...}

→ {"message":"请关闭安全系统"}
← {"type":"clarification","trace":{"fallback_reason":"high_risk_needs_confirmation"},...}

→ {"message":"确认执行","confirm":true,"session_id":"xxx"}
← {"type":"task_result",...}
```

---

## 项目验证

```bash
# 单元测试（63 passed）
pytest -q -v

# Agent 流水线测试
python scripts/run_agent.py -i                          # 交互式
python scripts/run_agent.py --query "打开空调"           # 单条
python scripts/run_agent.py --file data/nlu/multi_test.txt  # 批量
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 评测

# RAG 评估
python scripts/eval_rag.py --input data/training/qa_pairs/test_qa_pair_verify.json

# 意图分类测试
python scripts/train_3class.py --predict "怎么打开空调"

# vLLM 性能压测
python scripts/benchmark_vllm.py --url http://127.0.0.1:8000/v1

# 基线对比测试
python scripts/baseline_compare.py --model local
```

---

## 配置项

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | mock | LLM 后端：mock / doubao / vllm / openai |
| `DOUBAO_API_KEY` | — | 豆包 API Key |
| `DOUBAO_ENDPOINT` | https://ark.cn-beijing.volces.com/api/v3 | 豆包地址 |
| `DOUBAO_MODEL` | — | 豆包模型 ID |
| `VLLM_BASE_URL` | http://127.0.0.1:8000/v1 | vLLM 地址 |
| `RETRIEVER_BACKEND` | mock | mock / bm25 / milvus / hybrid |
| `RERANKER_BACKEND` | mock | mock / minicpm |
| `HYBRID_DENSE_BACKEND` | milvus | hybrid 模式下向量后端：milvus / faiss |
| `WEB_SEARCH_ENABLED` | false | 联网搜索开关 |
| `NLU_URL` | — | 外部 NLU 服务地址（端口8009） |
| `REJECT_URL` | — | 外部拒识服务地址（端口8007） |
| `REDIS_URL` | — | Redis 连接（无则内存） |
| `AMAP_API_KEY` | — | 高德地图 API Key |
| `TASK_CONFIDENCE_THRESHOLD` | 0.75 | Task 路由阈值 |
| `FAQ_CONFIDENCE_THRESHOLD` | 0.65 | FAQ 路由阈值 |
| `CHITCHAT_CONFIDENCE_THRESHOLD` | 0.60 | Chitchat 路由阈值 |
| `KNOWLEDGE_TOP_K` | 3 | RAG 检索返回数 |
| `ABBR_CSV_PATH` | data/abbr/abbr_ch.csv | 汽车缩写表 |

完整见 `.env.example`。

---

## 脚本对照

### XIAOMI_SU7_RAG → 融合

| 原始脚本 | 融合脚本 | 功能 |
|---------|---------|------|
| `build_index.py` | `scripts/build_index.py` | PDF→索引构建 |
| `generate_all_data.py` | `scripts/generate_data.py` | QA生成+过滤+数据集 |
| `generate_sft_data.py` | `scripts/generate_data.py --step dataset` | 训练集构建 |
| `final_score.py` | `scripts/eval_rag.py` | RAG评估 |
| `infer.py` | `POST /api/v1/chat` | 在线推理 |
| `src/rl/infer_rl.py` | `scripts/rl_infer.py` | RL推理 |
| `deploy/auto_vllm_server.py` | `scripts/run_vllm.py` | vLLM自动启动 |
| `deploy/benchmark.py` | `scripts/benchmark_vllm.py` | 性能压测 |
| `deploy/baseline_gpt4o.py` | `scripts/baseline_compare.py` | 基线对比 |
| `deploy/download_models.py` | `scripts/download_models.py` | 模型下载 |

### CarVoice_Agent → 融合

| 原始脚本 | 融合脚本 | 功能 |
|---------|---------|------|
| `download_models.py` | `scripts/download_models.py` | 模型下载 |
| `server.sh` | `scripts/start_all_services.sh` | 多服务启动 |
| `dialog.py` | `scripts/run_agent.py -i` | 交互测试 |
| `test.py` | `scripts/run_agent.py --file` | 批量测试 |
| `intent_client.py` 等 | `scripts/run_agent.py --eval` | 评测 |
| `train/run.py` | `scripts/train_intent.py` / `scripts/train_reject.py` | 训练 |

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/
│   ├── main.py                      # FastAPI 入口
│   ├── api/                         # HTTP + WebSocket 网关
│   ├── core/                        # 主控编排
│   │   ├── orchestrator.py          # 中央调度（改写→分类→路由→响应）
│   │   ├── classifier.py            # 意图分类（三级：BERT模型/启发式/LLM仲裁）
│   │   └── session.py               # 会话管理（Redis/内存）
│   │
│   ├── nlp/                         # NLP 管道（CarVoice client/ 移植）
│   │   ├── arbitration.py           # LLM仲裁（A/B/C/D四分类，182行Prompt）
│   │   ├── rewrite.py               # 查询改写（指代消解 + 安全校验）
│   │   ├── nlu.py                   # NLU 意图槽位提取
│   │   ├── nlg.py                   # NLG 自然语言生成
│   │   ├── reject.py                # 拒识模型
│   │   └── correlation.py           # 多轮关联判断
│   │
│   ├── skills/                      # 技能执行（CarVoice function_call/ 移植）
│   │   ├── definitions.py           # 455 函数定义
│   │   ├── registry.py              # 白名单注册表（7技能 + risk_level）
│   │   ├── nlu_data.py              # NLU数据加载（slot_intent + intent_map）
│   │   ├── slot_processor.py        # 槽位归一化
│   │   └── dm/                      # DM 处理器
│   │       ├── factory.py           # DM 工厂
│   │       ├── maps.py              # 导航（Go_POI）
│   │       ├── music.py             # 音乐（Search_Music）
│   │       └── weather.py           # 天气（Query_Weather）
│   │
│   ├── knowledge/                   # 知识 RAG（SU7_RAG src/ 移植）
│   │   ├── retriever/ (8个)         # BM25, FAISS, Milvus, Hybrid, TF-IDF, Qwen3
│   │   ├── reranker/  (5个)         # MiniCPM, BGE-M3, JinaV2, Qwen3, Qwen3-vLLM
│   │   ├── parser/                  # PDF 解析
│   │   ├── generator.py             # LLM 答案生成
│   │   ├── synthesizer.py           # 引用拼装（post_processing）
│   │   ├── chunker.py               # 语义分块
│   │   ├── web_search.py            # Web 垂直搜索
│   │   ├── semantic_chunk_server.py # 语义切分服务（6000）
│   │   ├── manual_store.py          # MongoDB 手册存储
│   │   └── service.py               # KnowledgeService 外观
│   │
│   ├── llm/                         # LLM 抽象层
│   │   ├── base.py                  # 基类 + 工厂
│   │   ├── doubao.py                # 豆包（CarVoice 原始技术栈）
│   │   ├── vllm.py                  # vLLM（SU7_RAG 原始技术栈）
│   │   ├── openai_client.py         # OpenAI 兼容
│   │   └── mock.py                  # Mock（本地开发）
│   │
│   ├── mcp/                         # MCP 协议
│   │   ├── client.py                # MCP 客户端
│   │   ├── amap_server.py           # 高德地图（13 工具）
│   │   └── music_server.py          # QQ 音乐
│   │
│   ├── prompts/                     # 7 个 System Prompt
│   │   ├── arbitration.py           # 仲裁 Prompt（182行）
│   │   ├── rewrite.py               # 改写 Prompt
│   │   ├── nlg.py                   # NLG Prompt
│   │   ├── nlu.py                   # NLU Prompt
│   │   ├── chat.py                  # 闲聊 Prompt
│   │   └── correlation.py           # 关联 Prompt
│   │
│   ├── train/                       # BERT 训练框架（CarVoice train/ 移植）
│   │   ├── core/                    # 自定义 BERT 实现（2159行）
│   │   │   ├── modeling.py          # BertModel, BertConfig（1206行）
│   │   │   ├── tokenization.py      # BertTokenizer（400行）
│   │   │   ├── optimization.py      # BertAdam（289行）
│   │   │   └── file_utils.py        # 文件工具（264行）
│   │   ├── models/bert_tiny.py      # BERT Tiny 模型
│   │   ├── train_eval.py            # train/test/evaluate
│   │   ├── data_helper.py           # build_dataset/build_iterator
│   │   ├── servers.py               # 推理服务（intent:8008, reject:8007, NLU:8009）
│   │   ├── run.py                   # 训练入口
│   │   └── models.py                # BERT 分类模型
│   │
│   ├── rl/                          # RL 模块（SU7_RAG src/rl/ 移植，12 文件）
│   │   ├── web_reader.py            # WebPageReader（URL抓取+HTML→文本）
│   │   ├── reward_model.py          # 6维奖励函数 + TRL GRPOTrainer 兼容
│   │   ├── environment.py           # 工具路由环境（LocalSearch + WebSearch 多后端）
│   │   ├── data_builder.py          # 网络兜底轨迹生成器
│   │   ├── build_local_trajectories.py # 本地可答轨迹生成器
│   │   ├── format_converter.py      # SFT/GRPO/ShareGPT 格式转换 + 标签修复
│   │   ├── train_grpo.py            # GRPO 训练入口
│   │   ├── infer_rl.py              # Search-R1 推理引擎
│   │   ├── batch_eval.py            # RL 批量评测
│   │   ├── rebalance_sft_data.py    # 训练数据再平衡
│   │   └── verify_export.py         # 模型导出验证
│   │
│   ├── data_pipeline/               # 数据管道（SU7_RAG gen_qa/ 移植）
│   │   ├── qa_generator.py          # QA 对生成
│   │   ├── qa_filter.py             # 质量过滤
│   │   ├── abbr_expander.py         # 缩写扩展
│   │   └── dataset_builder.py       # Summary/Rerank 训练集构建
│   │
│   ├── eval/                        # 评估框架（SU7_RAG final_score.py 移植）
│   │   ├── scorer.py                # 语义相似度 + 关键词加权
│   │   └── ragas_eval.py            # RAGas 评估
│   │
│   └── shared/                      # 共享层
│       ├── schemas.py               # Pydantic 模型
│       ├── config.py                # 环境配置
│       ├── logging.py               # 日志
│       ├── errors.py                # 错误码
│       ├── redis_client.py          # Redis（内存回退）
│       └── utils.py                 # WRRF 融合算法 + merge_docs + post_processing
│
├── scripts/                         # 执行脚本（17 个）
│   ├── autodl_start.sh              # AutoDL 一键启动
│   ├── start.sh / start.ps1         # 本地启动
│   ├── start_all_services.sh        # 5服务并行启动
│   ├── build_index.py               # RAG 索引构建
│   ├── generate_data.py             # 训练数据生成
│   ├── eval_rag.py                 # RAG 离线评估
│   ├── run_agent.py                 # Agent 测试（交互/批量/评测）
│   ├── download_models.py           # 模型下载（Agent + RAG）
│   ├── run_vllm.py                  # vLLM 自动启动
│   ├── benchmark_vllm.py            # vLLM 性能压测
│   ├── baseline_compare.py          # 基线对比（local vs OpenAI）
│   ├── rl_infer.py                 # RL 推理入口
│   ├── train_intent.py              # 意图模型训练
│   ├── train_reject.py              # 拒识模型训练
│   └── train_3class.py              # 三分类模型训练（Task/FAQ/Chitchat）
│
├── configs/                         # 训练配置
│   ├── sft.yaml                     # LLaMA-Factory SFT
│   ├── grpo.yaml                    # LLaMA-Factory GRPO
│   ├── original_sft.yaml            # 原始 SFT 配置（参考）
│   └── original_grpo.yaml           # 原始 GRPO 配置（参考）
│
├── data/                            # 手工整理源数据（21 个文件）
│   ├── knowledge/                   # Xiaomi_SU7_Manual.pdf + su7_docs.json
│   ├── nlu/                         # slot_intent(437) + intent_map(439) + class_labels + 测试语料
│   ├── training/                    # intent/reject 训练集 + 停用词 + 闲聊样本
│   └── abbr/                        # 汽车术语缩写表（53 条）
│
├── tests/                           # 63 个测试用例（9 个文件）
│   ├── test_classifier.py           # 意图分类（12 参数化）
│   ├── test_router.py               # 路由决策（含高风险防绕过）
│   ├── test_skills_registry.py      # 技能白名单
│   ├── test_knowledge_citations.py  # 知识引用结构
│   ├── test_gateway_api.py          # API 集成（含 WebSocket）
│   ├── test_nlp.py                  # 仲裁/NLU/NLG/改写/拒识/关联
│   ├── test_data_pipeline.py        # QA生成/过滤/缩写/数据集
│   ├── test_eval.py                 # 评分/RAGas
│   └── test_llm.py                  # LLM 工厂/Mock
│
├── docs/architecture.md             # 架构文档
├── docker-compose.yml               # 容器编排
├── Dockerfile
├── pytest.ini
├── requirements.txt
├── .env.example
└── .gitignore
```
