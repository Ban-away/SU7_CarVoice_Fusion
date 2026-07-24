# SU7_CarVoice_Fusion

> 小米 SU7 车载智能语音 Agent —— 任务技能执行 + 用户手册问答 + 百科闲聊，三路路由统一调度。

**核心能力**：意图识别 → 路由分发 → 工具调用 / RAG 检索 / 联网搜索 → 回复生成。支持 BERT 本地推理、LLM Function Calling、MCP 工具协议、Search-R1 自主检索决策。

**验证环境**：RTX 4090 (48GB) ×2, Python 3.12, PyTorch 2.11, CUDA 13.2 | **日期**：2026-07-10 ~ 2026-07-24

---

## 1. 整体架构

```
用户输入
  │
  ▼
┌─────────────────────────────────┐
│       三级意图分类               │
│  BERT(本地) → 规则 → LLM 仲裁    │
└─────────────┬───────────────────┘
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
    Task    FAQ   Chitchat
  (技能执行)(手册问答)(百科闲聊)
      │       │       │
      ▼       ▼       ▼
  ┌──────┐ ┌──────┐ ┌──────┐
  │BERT  │ │RAG   │ │拒识  │
  │意图  │ │检索  │ │检查  │
  ├──────┤ ├──────┤ ├──────┤
  │LLM   │ │BM25+ │ │多轮  │
  │Func  │ │Milvus│ │关联  │
  │Call  │ │→Mini │ │判断  │
  ├──────┤ │CPM→  │ ├──────┤
  │DM    │ │LLM   │ │联网  │
  │对话  │ │生成  │ │搜索  │
  ├──────┤ ├──────┤ ├──────┤
  │MCP   │ │引用  │ │LLM   │
  │工具  │ │拼装  │ │闲聊  │
  ├──────┤ └──────┘ └──────┘
  │NLG   │
  │回复  │
  └──────┘
      │       │       │
      └───────┼───────┘
              ▼
         统一响应
```

### 路由决策

| 分类 | 典型问题 | 处理管线 | 输出 |
|------|---------|---------|------|
| **Task** | "导航到天安门"、"播放周杰伦的歌" | BERT 意图 → LLM Function Calling → DM → MCP 工具 → NLG | 任务执行结果 |
| **FAQ** | "SU7 续航多少"、"怎么开空调" | RAG 检索 (BM25+Milvus→MiniCPM→LLM) | 手册答案 + 页码引用 |
| **Chitchat** | "你好"、"今天天气怎么样" | 拒识检查 → 联网搜索 → LLM 闲聊 | 闲聊回复 |

### 意图分类

| 级别 | 方案 | 准确率 | 说明 |
|------|------|:---:|------|
| 1 | BERT (RoBERTa-wwm-ext, 31w 语料) | 86.07% Top-1 | 本地推理，<5ms |
| 2 | 启发式规则 | ~90% | 祈使词 + 疑问句式 + 车辆信号词 |
| 3 | LLM 仲裁 (Doubao) | ~98% | 处理隐含意图（"我饿了"→导航） |

---

## 2. 环境要求

| 组件 | 用途 | 必需? |
|------|------|:---:|
| Python 3.12 + PyTorch 2.11 + CUDA 13.2 | 核心运行环境 | ✅ |
| Redis | 会话存储、多轮改写缓存 | Mock 模式可跳过 |
| MongoDB | RAG 文本块存储 | RAG 模式必需 |
| Milvus Lite | 向量检索 | RAG 模式必需 |
| vLLM | Qwen3-8B 推理服务 | 生产模式必需 |
| SerpAPI / Serper / Bing API Key | 联网搜索 | Chitchat 模式可选 |

---

## 3. 快速开始（Mock 模式，无需 GPU）

Mock 模式下所有 LLM 调用返回预设回复，适合验证流程是否跑通。

### 3.1 安装依赖

```bash
git clone git@github.com:Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion

# 基础依赖
pip install -r requirements.txt

# requirements.txt 未覆盖的依赖（需手动安装）
pip install huggingface_hub transformers PyMuPDF python-dotenv
```

### 3.2 配置环境变量

```bash
cp .env.example .env
```

`.env` 中确认以下配置（Mock 模式保持默认即可）：

```ini
LLM_PROVIDER=mock          # mock 模式，不调用真实 LLM
LLM_MODEL=mock
API_KEY=sk-mock
```

### 3.3 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

看到 `Uvicorn running on http://0.0.0.0:8080` 表示启动成功。

### 3.4 验证

```bash
# 健康检查
curl http://127.0.0.1:8080/healthz
# → {"status": "ok"}

# 任务型：导航
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请导航到公司"}'
# → {"type":"task_result","text":"...",...}

# 问答型：手册查询
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"SU7 续航是多少"}'
# → {"type":"faq_answer","text":"...","citations":[...],...}

# 闲聊型：问候
curl -X POST http://127.0.0.1:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好"}'
# → {"type":"chitchat","text":"...",...}
```

---

## 4. 完整部署（生产模式）

生产模式启动全部 7 个服务，三路路由完整可用。

### 4.1 配置 .env

```ini
LLM_PROVIDER=doubao         # 或 vllm / openai
LLM_MODEL=ep-2024xxxx       # 豆包模型 ID
DOUBAO_API_KEY=your_key     # 豆包 API Key
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# RAG
RETRIEVER_BACKEND=milvus    # 或 faiss / bm25
RERANKER_BACKEND=minicpm    # 或 none

# 联网搜索（至少配一个）
SERPAPI_KEY=your_key        # https://serpapi.com
SERPER_API_KEY=your_key     # https://serper.dev
BING_SEARCH_KEY=your_key    # https://portal.azure.com

# 会话
REDIS_URL=redis://localhost:6379/0
MONGODB_URL=mongodb://localhost:27017
```

### 4.2 启动依赖服务

```bash
# Redis（会话存储）
redis-server --port 6379 --daemonize yes

# MongoDB（RAG 文本块存储）
mongod --dbpath ./mongodb-data --fork --logpath ./mongodb.log

# Milvus Lite（向量检索，Python 进程内启动，无需单独部署）
```

### 4.3 训练 / 下载必需模型

```bash
# 下载预训练模型（首次使用）
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --preset agent    # BERT 意图 + 拒识
python scripts/download_models.py --preset rag      # BGE + SPLADE + MiniCPM
```

如果已经训练过 BERT 模型，确认 checkpoint 存在：
```
models/saved/intent/bert.ckpt       (392MB)
models/saved/reject/bert_tiny.ckpt  (24MB)
```

### 4.4 构建 RAG 索引

```bash
# 解析 SU7 用户手册 PDF，构建 BM25 + FAISS + Milvus 索引
python scripts/build_index.py --backend all --pdf data/knowledge/Xiaomi_SU7_Manual.pdf

# 验证索引质量
python scripts/check_training_data.py     # 7/7 数据检查通过
python scripts/evaluate_parse_quality.py  # PDF 解析准确率 98.31%
```

### 4.5 启动推理服务

```bash
# 启动 vLLM 推理服务（Qwen3-8B INT4，5.7GB 显存）
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000

# 验证
curl http://127.0.0.1:8000/v1/models
# → {"data":[{"id":"qwen3_lora_sft_int4",...}]}
```

### 4.6 启动全部业务服务

```bash
bash scripts/start_all_services.sh
```

这会后台启动以下服务：

| 服务 | 端口 | 用途 |
|------|:---:|------|
| 融合主服务 | 8080 | 对外 API 入口 |
| 拒识服务 | 8007 | 问题拒识 |
| 意图服务 | 8008 | BERT 意图分类 |
| NLU 服务 | 8009 | 槽位提取 |

### 4.7 端到端验证

```bash
# 运行测试套件
pytest -q -v                          # 63 个用例

# 单条推理验证
python scripts/run_agent.py --query "导航到天安门"

# 批量评测
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt
```

---

## 5. 模型训练

### 5.1 BERT 意图分类模型

**目的**：训练 BERT 识别 439 类用户意图（导航、音乐、天气等），本地推理延迟 <5ms。

```bash
# 训练（约 12 分钟，单卡）
python scripts/train_intent.py

# 输出
# models/saved/intent/bert.ckpt (392MB)
# 验证指标：Top-1 86.07%, Top-5 97.64%
```

- 基座模型：`chinese_roberta_wwm_ext`（哈工大）
- 训练数据：31 万条中文车载场景语料
- 架构：RoBERTa + Linear 分类头，全参数微调

### 5.2 BERT-Tiny 拒识模型

**目的**：判断用户问题是否应该拒绝回答（非车载领域、敏感问题等）。

```bash
# 训练（约 2 分钟，单卡）
python scripts/train_reject.py

# 输出
# models/saved/reject/bert_tiny.ckpt (24MB)
# 验证指标：Accuracy 89.56%
```

- 基座模型：`roberta_tiny_clue`（CLUE 社区，3 层）
- 训练数据：32 万条
- 架构：3 层 BERT + Linear 二分类头

### 5.3 Qwen3-8B SFT 微调

**目的**：在 SU7 手册问答数据上微调 Qwen3-8B，提升手册问答质量。

| 方案 | 显存需求 | 命令 | 适用场景 |
|------|:---:|------|------|
| peft + trl | 16GB | `python scripts/run_sft_minimal.py` | 最小验证 |
| Unsloth | 12GB | `python scripts/train_sft_unsloth.py` | 单卡优化 |
| LLaMA-Factory | 24GB | `llamafactory-cli train configs/sft.yaml` | 生产训练 |
| DeepSpeed ZeRO-3 | 2×48GB | `deepspeed --num_gpus=2` | 双卡并行 |

```bash
# 以 LLaMA-Factory 为例
llamafactory-cli train configs/sft.yaml

# 训练后 INT4 量化（部署用）
# → LLaMA-Factory-main/output/qwen3_lora_sft_int4/ (5.7GB)

# 启动 vLLM 推理
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000

# 压测
python scripts/benchmark_vllm.py --url http://127.0.0.1:8000/v1
# → 1832 tok/s, TTFT 133ms
```

> **注意**：`configs/sft.yaml` 需确认 `do_train: true`，注释掉 `quantization_bit`/`quantization_method`（CUDA 13 + bitsandbytes 不兼容，训练时使用 FP16 而非 QLoRA 4-bit）。

### 5.4 GRPO 强化学习（Search-R1）

**目的**：训练模型自主决策检索策略——何时搜本地、何时联网、何时读页面、何时终止。

**核心思路**：不同于传统 RAG 的硬编码 pipeline（先搜 A 再搜 B），Search-R1 让模型**自己选择**每一步做什么。

```
推理示例：
  问题: "SU7 冬季续航下降多少？"
  → <search_local> SU7 冬季续航          ← 模型自主决定先搜本地
  → <information> 手册相关段落...          ← 获得本地结果
  → （内部判断：手册没写冬季数据）
  → <search_web> SU7 冬季续航测试        ← 模型自主决定联网
  → <information> 搜索结果 + URL           ← 获得网络结果
  → <answer> 冬季续航约下降 30-40%...     ← 模型自主决定终止、回答
```

**奖励函数设计**（6 维度 / 最高 1.0）：

| 维度 | 权重 | 评估内容 |
|------|:---:|------|
| 答案质量 | 0.40 | 答案是否基于检索到的信息、包含具体数据 |
| 工具合理性 | 0.15 | 是否优先本地、本地不够才联网 |
| 探索深度 | 0.15 | 是否必要时深度阅读网页、是否多轮检索 |
| 领域合规 | 0.15 | 是否拒绝回答非车载领域问题 |
| 来源标注 | 0.10 | 联网答案是否标注来源域名 |
| 格式完整性 | 0.05 | 标签是否正确闭合 |

**数据管线**：

```bash
# 1. 生成训练轨迹（共 22,820 条）
python app/rl/data_builder.py              # 500 条网络兜底轨迹（需 SerpAPI）
python app/rl/build_local_trajectories.py  # 22,320 条本地可答轨迹

# 2. 格式转换（合并为 GRPO / SFT 训练格式）
python app/rl/format_converter.py          # → 1,500 条合并后样本

# 3. 再平衡（防止本地/网络样本比例失衡）
python app/rl/rebalance_sft_data.py
```

**训练**：

```bash
# 方案 A：TRL 单卡快速验证
python app/rl/train_grpo.py --stage grpo

# 方案 B：VeRL 多卡训练（21K+ 样本，多卡并行）
python app/rl/train_grpo_verl.py --n-gpus 2
```

**推理与评测**：

```bash
# 启动 RL 模型推理
vllm serve LLaMA-Factory-main/output/qwen3_lora_rl --port 8000 --served-model-name su7_rl

# 单条推理（展示完整轨迹）
python scripts/rl_infer.py --model su7_rl --show-trajectory

# 批量评测
python app/rl/batch_eval.py --model su7_rl --vllm-url http://localhost:8000/v1 --dry-run
```

---

## 6. 评测结果

### BERT 精度

| 指标 | 实测值 |
|------|:---:|
| 意图 Top-1 | 86.07% |
| 意图 Top-5 | 97.64% |
| 拒识 Accuracy | 89.56% |

### RAG 检索

| 指标 | 实测值 |
|------|:---:|
| RAGAs context_recall | 88.47% |
| RAGAs context_precision | 97.64% |
| PDF 解析准确率 | 98.31% |

### 推理性能

| 指标 | 值 |
|------|-----|
| vLLM 吞吐率 | 1832 tok/s |
| TTFT 首 token 延迟 | 133ms |
| 拒识服务 QPS | 89ms |
| 意图服务 QPS | 180ms |

### E2E 批量测试（242 条）

| 类型 | 占比 |
|------|:---:|
| 任务执行 (task_result) | 79% |
| 闲聊 (chitchat) | 17% |
| 手册问答 (faq_answer) | 3% |

### 全量验证清单

| 模块 | 状态 | 说明 |
|------|:---:|------|
| Mock 推理 | ✅ | 5/5 curl 通过 |
| pytest | ✅ | 63/63 |
| BERT 意图训练 | ✅ | 86.07% Top-1 |
| BERT 拒识训练 | ✅ | 89.56% |
| BM25 索引 | ✅ | 144 chunks |
| FAISS 索引 | ✅ | |
| Milvus 混合索引 | ✅ | Dense + Sparse |
| PDF 解析 + 语义切分 | ✅ | 98.31% |
| 联网搜索四级回退 | ✅ | SerpAPI → Serper → Bing → Doubao |
| SFT 微调 (3 框架) | ✅ | peft + Unsloth + LLaMA-Factory |
| DeepSpeed ZeRO-3 | ✅ | 双卡 Qwen3-8B |
| vLLM INT4 部署 | ✅ | 5.7GB, 1832 tok/s |
| RL 数据管线 | ✅ | 22,820 条轨迹生成 |
| GRPO 验证训练 (TRL) | ✅ | 5 steps 流程验证 |
| VeRL 训练脚本 | ✅ | Ray+NCCL+FSDP 初始化通过，21K+ 样本训练就绪 |
| RL 推理 | ✅ | 五标签 + read_page |
| NLU 评测 | ✅ | 服务在线 |
| QPS 压测 ×3 | ✅ | 89 / 180 / 1483ms |
| E2E 批量 | ✅ | 242 条 |
| Redis / MongoDB / Milvus | ✅ | |
| 全部模型下载 | ✅ | 8/8 |

---

## 7. 项目结构

```
SU7_CarVoice_Fusion/
├── app/
│   ├── api/             HTTP + WebSocket 入口
│   ├── core/            Agent 编排器 (orchestrator, classifier, session)
│   ├── nlp/             NLP (intent, reject, NLU, NLG, arbitration, rewrite)
│   ├── skills/          技能定义 + DM 对话管理 (maps/music/weather)
│   ├── knowledge/       RAG 检索 (BM25/FAISS/Milvus + MiniCPM reranker + web_search)
│   ├── llm/             LLM 客户端 (doubao/vllm/openai/mock)
│   ├── mcp/             MCP 工具 (高德地图 13 工具 + QQ 音乐)
│   ├── train/           BERT 训练 (models/servers)
│   └── rl/              Search-R1 RL (data_builder, train_grpo, infer_rl, reward_model)
├── scripts/             30+ 执行脚本（训练、评测、压测）
├── configs/             sft.yaml / grpo.yaml / export.yaml
├── data/                训练数据 + 知识库 + RL 轨迹
├── models/              预训练模型 + BERT checkpoints
├── LLaMA-Factory-main/  训练框架及产物 (SFT/INT4/RL)
├── tests/               63 单元测试
├── docs/                架构文档
└── CLAUDE.md            AI 协作指南
```

---

## 8. API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话 |
| `GET` | `/api/v1/skills` | 技能白名单 |
| `POST` | `/api/v1/knowledge/retrieve` | 知识检索（调试用） |
| `WS` | `/ws/chat` | WebSocket 流式对话 |

### 请求体

```json
{
  "message": "导航到天安门",
  "session_id": "optional-session-id",
  "confirm": false
}
```

### 响应体

```json
{
  "type": "task_result | faq_answer | chitchat | clarification",
  "text": "已为您规划前往天安门的路线...",
  "citations": [{"source": "用户手册第 45 页", "page": 45}],
  "trace": {
    "route": "Task",
    "classifier_confidence": 0.95,
    "latency_ms": 234,
    "session_id": "uuid"
  },
  "session_id": "uuid"
}
```

---

## 9. 常见问题

### CUDA 13 + bitsandbytes 不兼容

训练 SFT 时使用 FP16 而非 QLoRA 4-bit。`configs/sft.yaml` 中注释掉 `quantization_bit` 和 `quantization_method`。

### HuggingFace Xet 存储下载 401

使用 ModelScope 国内镜像或 `HF_ENDPOINT=https://hf-mirror.com`。

### vLLM + ragas openai 版本冲突

锁定 `openai==2.47.0`。

### MongoDB OpenSSL 不兼容

捆绑安装 `libssl1.1`：`dpkg -x libssl1.1.deb /usr/lib/x86_64-linux-gnu/`

### Milvus 连接失败

需先 `milvus-lite` 启动本地服务（`localhost:19530`），或 PyMilvus 版本降级到 2.3.x。

### MiniCPM transformers 5.x 报错

需要 `auto_map` 本地化 + `is_torch_sdpa_available` 函数补丁。

---

## 许可证

MIT License
