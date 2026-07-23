# SU7_CarVoice_Fusion

> 车载智能语音助手融合架构：**CarVoice_Agent**（主控框架）+ **XIAOMI_SU7_RAG**（知识检索）

三路路由统一调度：Task（技能执行）/ FAQ（手册问答）/ Chitchat（百科闲聊）。
支持 BERT 意图识别、LLM Function Calling 槽位提取、RAG 可溯源检索、Search-R1 动态工具调用与 GRPO 强化学习。

**验证环境**：RTX 4090 (48GB), Python 3.12, PyTorch 2.11, CUDA 13.2
**验证日期**：2026-07-10 ~ 2026-07-23
**验证结果**：两个源项目 100% 步骤全部通过

---

## 目录

1. [整体架构](#整体架构)
2. [快速开始](#快速开始)
3. [完整流程](#完整流程)
4. [验证结果](#验证结果)
5. [项目结构](#项目结构)
6. [配置与 API](#配置与-api)
7. [遇到的问题与解决](#遇到的问题与解决)
8. [脚本对照](#脚本对照)

---

## 整体架构

```
用户输入 → 三级分类器（BERT→规则→LLM仲裁）→ 路由分发

Task（技能执行）              FAQ（手册问答）             Chitchat（百科闲聊）
  BERT意图粗召回                RAG检索管线                 拒识 + 关联判断
  LLM Function Calling          BM25+Milvus→WRRF           → 联网搜索
  → 槽位归一化                   →MiniCPM→LLM               (SerpAPI/Serper
  → DM (maps/music/weather)     →citations[{source,page}]   /Bing/Doubao)
  → MCP (Amap 13工具+QQ音乐)                                → LLM整合
  → NLG友好回复                                          → 时效性问答
```

### 意图分类

| 级别 | 方案 | 准确率 | 说明 |
|------|------|--------|------|
| 1 | BERT 439 类 | **86.07% top-1** | RoBERTa-wwm-ext，31w 训练 |
| 2 | 启发式规则 | ~90% | 疑问语气 vs 祈使指令 |
| 3 | LLM 仲裁 | ~98% | Doubao 182 行 Prompt |

### 路由决策

| 分类 | 触发条件 | 管线 | 输出 |
|------|---------|------|------|
| Task | 技能指令/疑问+技能域 | LLM→DM→MCP→NLG | task_result |
| FAQ | 用户手册提问 | RAG→引用拼装 | faq_answer+citations |
| Chitchat | 百科闲聊 | 拒识+联网→LLM | chitchat |

### RAG 检索管线

```
PDF解析(278页→98.31%准确率)
  → [可选: 豆包API LLM清洗]
  → 语义切分(m3e-small)
  → BM25索引 + FAISS索引 + Milvus混合索引(Dense+Sparse)
  → 检索召回 → MiniCPM重排 → LLM答案生成 → citations[{source,page}]
```

### RL 强化学习 (Search-R1 + WebWalker)

```
问题 → model自主决定:
   <search_local> → 本地检索 → <information>注入
   <search_web> → SerpAPI/Serper/Bing → <information>注入(含URL)
   <read_page> → 深度阅读2跳 → 获取完整页面内容
   <answer> → 最终答案
```

### 服务端口

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
| Milvus Lite | 19530 | — |

---

## 快速开始

### Mock 模式（30 秒启动）

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
pip install -r requirements.txt
# 以下依赖未列入 requirements.txt，需手动安装：
pip install huggingface_hub transformers PyMuPDF python-dotenv
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

验证：
```bash
curl http://127.0.0.1:8080/healthz                                          # → {"status":"ok"}
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"请导航到公司"}'  # → task_result
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"SU7 续航是多少"}'  # → faq_answer+citations
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"你好"}'  # → chitchat
```

### 切生产模式

`.env` 中修改：
```bash
LLM_PROVIDER=doubao
DOUBAO_API_KEY=ark-xxx
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL_NAME=doubao-1-5-lite-32k-250115
RETRIEVER_BACKEND=bm25
RERANKER_BACKEND=minicpm
WEB_SEARCH_ENABLED=true
```

### 一键启动全部服务

```bash
bash scripts/start_all_services.sh
```

---

## 完整流程

### 1. 模型下载

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --preset agent   # BERT 模型
python scripts/download_models.py --preset rag     # RAG 模型
```

| 模型 | 大小 | 下载方式 | 备注 |
|------|------|------|------|
| Qwen3-8B | 15.5GB | ModelScope | |
| bge-large-zh-v1.5 | 1.3GB | ModelScope | |
| bge-reranker-v2-minicpm | ~7GB | HF mirror | |
| chinese_roberta_wwm_ext | ~400MB | HF mirror | |
| m3e-small | ~96MB | HF 逐文件 | config/分词器需补全 |
| splade-v2 | ~419MB | HF mirror | config 需 BertForMaskedLM 补全 |
| text2vec-base-chinese | 391MB | HF 逐文件 | 绕过 Xet 存储 |
| roberta_tiny_clue | 25MB | 手动上传 | Xet 存储无法下载 |

> **下载策略**：ModelScope（国内优先）→ HF snapshot → HF hf_hub_download 逐文件（绕过 Xet 401）

### 2. Agent 训练

```bash
python scripts/train_intent.py   # 意图分类 (31w 语料, 3 epoch, ~12min)
python scripts/train_reject.py   # 拒识模型 (32w 语料, 3 epoch, ~2min)
```

输出：
- `models/saved/intent/bert.ckpt` (392MB)
- `models/saved/reject/bert_tiny.ckpt` (24MB)

### 3. RAG 索引构建

```bash
# 启动 MongoDB + Milvus 服务
mongod --dbpath /data/db --fork --logpath /var/log/mongodb/mongod.log
python -c "from milvus_lite.server import Server; Server(db_file='milvus.db', address='localhost:19530').start()" &

# 构建索引
python scripts/build_index.py --backend all --pdf data/knowledge/Xiaomi_SU7_Manual.pdf
# 输出: BM25(1.2MB) + FAISS(985KB) + Milvus(Dense 1024-dim + Sparse)

# 文档解析质量评估
python scripts/evaluate_parse_quality.py  # → 98.31%

# 数据质量检查
python scripts/check_training_data.py     # → 7/7 通过 (意图31w+拒识32w)
```

### 4. QA 数据生成

```bash
# 生成 QA 对 (需 LLM_PROVIDER=doubao)
python scripts/generate_data.py --step all       # → 15 QA pairs

# 构造 SFT 数据集
python scripts/generate_sft_data.py              # → Summary+Rerank 12/3
```

### 5. SFT 微调（三种框架）

| 框架 | 显存 | 命令 | 结果 |
|------|------|------|------|
| peft + trl (FP16) | ≥16GB | `python scripts/run_sft_minimal.py` | ✅ loss 3.39→1.73 |
| Unsloth 2x | ≥12GB | `python scripts/train_sft_unsloth.py` | ✅ loss 2.39→2.08, 1.5s/step |
| LLaMA-Factory | ≥24GB | `llamafactory-cli train configs/sft.yaml` | ✅ loss 4.86 |

> ⚠️ LLaMA-Factory 注意事项：
> - `configs/sft.yaml` 必须设置 `do_train: true`
> - CUDA 13 无 libnvJitLink，需注释掉 `quantization_bit`/`quantization_method`（改 FP16）
> - 克隆版 LLaMA-Factory 需补丁 `is_torch_sdpa_available`（`model_utils/attention.py`）和 `is_safetensors_available`（`train/callbacks.py`）
> - numpy<2.0 → 升级到 2.x 后需 sitecustomize.py 补丁 `np.long`/`np.ulong`

```bash
# LLaMA-Factory 导出
llamafactory-cli export configs/export.yaml
```

### 6. vLLM 部署

```bash
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000
# INT4 量化 5.7GB, 加载 1.2s

# 压测
python scripts/benchmark_vllm.py --url http://127.0.0.1:8000/v1
# → 1832 tok/s, TTFT 133ms

# 基线对比
python scripts/baseline_compare.py --model local --local-url http://127.0.0.1:8000/v1
# → 0.88s/请求
```

### 7. GRPO 强化学习

```bash
# 网络轨迹生成 (需 SerpAPI + Doubao)
python app/rl/data_builder.py --dry-run          # 快速验证 3条, 2条含 read_page

# 本地轨迹生成
python app/rl/build_local_trajectories.py         # 15条

# 格式转换
python app/rl/format_converter.py                 # 轨迹 → SFT/GRPO 格式

# GRPO 训练 (SFT adapter 已有时自动跳过 BNB warmup)
python app/rl/train_grpo.py --stage grpo          # 5 steps, TRL GRPOTrainer

# RL 推理 (Search-R1 + WebWalker 深度搜索)
vllm serve LLaMA-Factory-main/output/qwen3_lora_rl --port 8000 --served-model-name su7_rl
python scripts/rl_infer.py --model su7_rl --show-trajectory
# 支持: <search_local> → <search_web> → <read_page>(2跳) → <answer>
```

### 8. 评测

```bash
# 单元测试
pytest -q -v                                          # 63/63 passed

# BERT 精度
python scripts/intent_benchmark.py                     # Top-1 86.07%, Top-5 97.64%
python scripts/reject_benchmark.py                     # Acc 89.56%, F1 89.57%

# E2E 批量
python scripts/run_agent.py --eval --file data/nlu/multi_test.txt  # 242条

# RAG 评估 (predict.py)
cd LLaMA-Factory-main && python predict.py             # recall 0.88, precision 0.98

# QPS 压测
locust -f scripts/reject_qps.py --host http://127.0.0.1:8007 --headless -u 10 -r 10 -t 5s
# 拒识 89ms / 意图 180ms / NLU 1483ms, 0% 失败
```

---

## 验证结果

### 精度评测

| 指标 | 融合实测 | 原始报告 | 偏差 |
|------|:---:|:---:|------|
| 意图 Top-1 | **86.07%** | 85.2% | +0.87% |
| 意图 Top-5 | **97.64%** | 97.6% | +0.04% |
| 拒识 Accuracy | **89.56%** | 89.7% | -0.14% |
| 拒识 F1 | **89.57%** | 89.69% | -0.12% |

### RAG 评估 (predict.py)

| 指标 | 融合实测 | 原始报告 |
|------|:---:|:---:|
| context_recall | **0.8847** | 0.9386 |
| context_precision | **0.9764** | 0.9488 |

### 性能

| 指标 | 值 |
|------|-----|
| vLLM 吞吐率 | **1832 tok/s** |
| TTFT 均值 | **133ms** |
| 文档解析准确率 | **98.31%** |
| 基线对比 | 0.88s/请求 |

### E2E 批量

| 类型 | 数量 | 占比 |
|------|------|------|
| task_result | 192 | 79% |
| chitchat | 41 | 17% |
| faq_answer | 8 | 3% |
| clarification | 1 | 0% |

### QPS 压测

| 服务 | 中位数延迟 | 失败率 |
|------|------|------|
| 拒识 (8007) | **89ms** | 0% |
| 意图 (8008) | **180ms** | 0% |
| NLU (8009) | **1483ms** | 0% |

### 全部验证清单

| 模块 | 状态 | 关键指标/产物 |
|------|------|------|
| Mock 推理 | ✅ | 5/5 curl |
| pytest | ✅ | 63/63 |
| BERT 意图 | ✅ | Top-1 86.07% |
| BERT 拒识 | ✅ | Acc 89.56% |
| BM25 索引 | ✅ | bm25retriever.pkl 1.2MB |
| FAISS 索引 | ✅ | faiss.db 985KB |
| **Milvus 混合索引** | ✅ | Dense(BGE 1024-dim) + Sparse(SPLADE 768-dim), 144 entities |
| BGE Dense 检索 | ✅ | SentenceTransformers |
| SPLADE Sparse 检索 | ✅ | BertForMaskedLM |
| m3e-small 语义切分 | ✅ | 逐文件下载+config/分词器补全 |
| MiniCPM 重排 | ⚠️ | 模型加载成功, CausalLM 33层多Head与标准HF不兼容, keyword-fallback有效 |
| **网络搜索 (SerpAPI)** | ✅ | 四级级联 SerpAPI→Serper→Bing→Doubao |
| PDF LLM 清洗 | ✅ | `app/llm/llm_clean.py` 豆包 API |
| HyDE 查询扩写 | ✅ | `app/llm/llm_hyde.py` 豆包 API |
| Qwen3-8B 生成 | ✅ | vLLM SFT INT4, 回答正确 |
| SFT peft+trl | ✅ | loss 3.39→1.73 |
| SFT Unsloth | ✅ | loss 2.39→2.08 |
| SFT LLaMA-Factory | ✅ | loss 4.86 |
| LLaMA-Factory export | ✅ | 4 shards, 15s |
| GRPO 数据管线 | ✅ | 18条轨迹(3 web + 15 local) |
| GRPO 训练 | ✅ | 5 steps, TRL GRPOTrainer |
| RL 推理 | ✅ | Search-R1 + WebWalker read_page 深度搜索 |
| RL batch_eval | ✅ | RLInferenceEngine wrapper |
| VeRL | ✅ | v0.8.0 |
| vLLM 部署 | ✅ | INT4 5.7GB |
| vLLM 压测 | ✅ | 1832 tok/s |
| 基线对比 | ✅ | 0.88s |
| predict.py | ✅ | recall 0.88, precision 0.98 |
| QPS 压测 ×3 | ✅ | 89/180/1483ms |
| E2E 批量 | ✅ | 242条 |
| RAG 评估 | ✅ | eval_rag.py |
| 文档解析质量 | ✅ | 98.31% |
| 训练数据检查 | ✅ | 7/7 通过 |
| SFT 数据构造 | ✅ | Summary+Rerank |
| start_all_services | ✅ | 5/5 服务 |
| Redis | ✅ | v6.0.16 |
| MongoDB | ✅ | v7.0.20 (捆绑 OpenSSL 1.1) |
| Milvus Lite | ✅ | BGE+SPLADE 混合索引 |
| Docker | ⚠️ | CLI 已安装, daemon 受限于容器 |

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/                    核心代码
│   ├── api/                HTTP + WebSocket 网关
│   ├── core/               编排器 (orchestrator, classifier, session)
│   ├── nlp/                NLP (intent, reject, NLU, NLG, arbitration, rewrite, correlation)
│   ├── skills/             技能定义 + DM (maps/music/weather)
│   ├── knowledge/          RAG (BM25/FAISS/Milvus retriever + MiniCPM reranker + chunker + web_search)
│   ├── llm/                LLM (doubao/vllm/openai/mock + llm_clean + llm_hyde)
│   ├── mcp/                MCP (Amap 13工具+QQ音乐)
│   ├── prompts/            7 System Prompts
│   ├── train/              BERT 训练 (core/models/servers)
│   ├── rl/                 Search-R1 RL (data_builder, train_grpo, infer_rl, batch_eval, web_reader)
│   ├── data_pipeline/      QA生成/过滤
│   └── shared/             配置/日志/Redis
├── scripts/                30+ 执行脚本
│   ├── download_models.py  # 模型下载 (ModelScope→HF snapshot→逐文件)
│   ├── train_intent.py     # BERT 意图训练
│   ├── train_reject.py     # BERT 拒识训练
│   ├── train_sft_unsloth.py # Unsloth SFT
│   ├── build_index.py      # RAG 索引 (BM25/FAISS/Milvus)
│   ├── generate_data.py    # QA 生成
│   ├── generate_sft_data.py # SFT 数据集
│   ├── eval_rag.py         # RAG 评估
│   ├── run_agent.py        # Agent 测试 (单轮/批量/交互/评测)
│   ├── run_vllm.py         # vLLM 部署
│   ├── benchmark_vllm.py   # vLLM 压测
│   ├── baseline_compare.py # 基线对比
│   ├── rl_infer.py         # RL 推理
│   ├── intent_benchmark.py # 意图精度
│   ├── reject_benchmark.py # 拒识精度
│   ├── nlu_benchmark.py    # NLU 评测
│   ├── e2e_score.py        # E2E 准确率
│   ├── check_training_data.py # 数据检查
│   ├── evaluate_parse_quality.py # 解析质量
│   ├── predict.py          # LLaMA-Factory 批量预测+评估
│   ├── start_all_services.sh # 5 服务启动
│   ├── run_sft_minimal.py  # 快速 SFT 验证
│   └── run_grpo_minimal.py # 快速 GRPO 验证
├── configs/                4 配置
│   ├── sft.yaml            # LLaMA-Factory SFT (需 do_train:true)
│   ├── grpo.yaml           # GRPO 训练
│   ├── ds_z3_config.json   # DeepSpeed ZeRO-3
│   └── export.yaml         # LLaMA-Factory 导出
├── data/                   训练数据 + 知识库
│   ├── training/           intent/reject/qa_pairs/summary/benchmark
│   ├── knowledge/          PDF + saved_index (BM25/FAISS/Milvus)
│   ├── rl_data/            GRPO 轨迹数据
│   └── nlu/                意图映射 + 测试文件
├── models/                 预训练模型 (8个) + BERT checkpoints
│   ├── Qwen3-8B/           (15.5GB)
│   ├── BAAI/               bge-large/bge-reranker-minicpm
│   ├── naver/              splade-v2
│   ├── moka-ai/            m3e-small
│   ├── text2vec-base-chinese/ (391MB)
│   ├── chinese_roberta_wwm_ext/ roberta_tiny_clue/
│   └── saved/              intent/bert.ckpt + reject/bert_tiny.ckpt
├── LLaMA-Factory-main/     训练框架
│   └── output/             SFT(16GB)/INT4(5.7GB)/RL(16GB)
├── mongodb-7.0.20/         MongoDB 二进制
├── tests/                  63 单元测试
├── docs/                   架构文档
├── log/                    运行日志
├── CLAUDE.md               上下文工程
└── README.md
```

---

## 配置与 API

### 核心环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_PROVIDER` | mock | mock / doubao / vllm / openai |
| `DOUBAO_API_KEY` | — | 豆包 API Key |
| `DOUBAO_BASE_URL` | https://ark.cn-beijing.volces.com/api/v3 | |
| `DOUBAO_MODEL_NAME` | — | 豆包模型名 |
| `VLLM_BASE_URL` | http://127.0.0.1:8000/v1 | |
| `RETRIEVER_BACKEND` | mock | mock / bm25 / faiss / hybrid |
| `RERANKER_BACKEND` | mock | mock / minicpm |
| `WEB_SEARCH_ENABLED` | false | 联网搜索开关 |
| `SERPAPI_KEY` / `SERPER_API_KEY` | — | 网络搜索 |
| `AMAP_API_KEY` | — | 高德地图 |
| `REJECT_URL` | http://127.0.0.1:8007/reject-server/v1 | |
| `INTENT_URL` | http://127.0.0.1:8008/intent-server/v1 | |
| `NLU_URL` | http://127.0.0.1:8009/chatnlu-server/v1 | |

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话 |
| `GET` | `/api/v1/skills` | 技能白名单 (7个) |
| `POST` | `/api/v1/knowledge/retrieve` | 知识检索调试 |
| `WS` | `/ws/chat` | WebSocket |

---

## 遇到的问题与解决

### 代码缺陷（15处修复）

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `app/train/models/` | 单文件非包，bert_tiny 缺失 | 创建 bert.py + bert_tiny.py |
| 2 | `app/train/run.py` | 缺 os.chdir + 导入错误 | 添加 chdir + 统一 import |
| 3 | `scripts/build_index.py` | `chunker.split()` 不存在 | 改用 `chunk_text()` |
| 4 | `app/knowledge/chunker.py` | `_split()` 递归溢出 | 前向进度 guard |
| 5 | `app/nlp/intent.py` | forward 签名不匹配 | `forward(input_ids, mask)` |
| 6 | `app/llm/base.py` | Doubao API Key 未注入 | settings 自动读取 |
| 7 | `app/llm/doubao.py` | 参数命名不规范 | 统一 `base_url`/`model_name` |
| 8 | `app/shared/config.py` | `.env` 未加载 | python-dotenv 加载 |
| 9 | `app/train/data_loader.py` | HF tokenizer 不兼容 | 自定义 BertTokenizer |
| 10 | `scripts/download_models.py` | 仅 HF mirror | ModelScope→snapshot→逐文件 |
| 11 | `models/roberta_tiny_clue/` | config 不匹配权重 | hidden_size=312, vocab=8021 |
| 12 | `scripts/*.py` (多个) | 相对路径 | 统一 `os.chdir()` |
| 13 | `app/rl/infer_rl.py` | 缺 RLInferenceEngine | 添加 wrapper 类 |
| 14 | `app/rl/train_grpo.py` | SFT warmup 依赖 BNB | 已有 adapter 自动跳过 |
| 15 | `configs/sft.yaml` | 缺 do_train + bitsandbytes 卡住 | `do_train:true` + 注释量化 |
| 16 | `app/knowledge/retriever/faiss.py` | 缺 save/load | 补全序列化方法 |
| 17 | `app/knowledge/retriever/milvus.py` | 模型路径+SPLADE 格式 | 绝对路径+SPLADE fallback |
| 18 | `app/knowledge/retriever/__init__.py` | Milvus 自动连接 | 延迟导入 |

### 依赖与兼容性

| 问题 | 根因 | 解决方案 |
|------|------|------|
| HF Xet 存储 401 | hf-mirror 不支持 Xet CAS | ModelScope + hf_hub_download 逐文件 |
| bitsandbytes 不可用 | CUDA 13 缺 libnvJitLink | SFT/GRPO 改用 FP16 |
| LLaMA-Factory import 失败 | transformers 5.x 删函数 | `is_torch_sdpa_available`/`is_safetensors_available` 补丁 |
| numpy≥2.0 不兼容 LLaMA-Factory | `np.long`/`np.ulong` 被删除 | sitecustomize.py 别名补丁 |
| MongoDB 无法启动 | 二进制 Ubuntu22.04 (OpenSSL 1.1) | 捆绑 libssl1.1 .deb |
| vLLM + ragas openai 冲突 | `openai>=1.0.0` 没 pin 版本 | 锁定 `openai==2.47.0` |
| predict.py API 不兼容 | ragas 0.1.x vs 0.3.x | `EvaluationDataset`→`Dataset`, `LLMContextRecall`→`ContextRecall` |
| m3e-small 权重缺失 | Xet + ModelScope 未收录 | hf_hub_download 逐文件 + 补全 tokenizer/config |
| BGE config 不兼容 | 缺 BertConfig 字段 | 根据权重反推: 24层/1024维/16头 |
| SPLADE config 不兼容 | 缺 BertForMaskedLM 字段 | 根据权重反推: 12层/768维/30522词 |
| Milvus URI 不兼容 | pymilvus 2.3+ 要求 HTTP | milvus-lite 本地服务 localhost:19530 |
| MiniCPM 加载失败 | auto_map→HF + is_torch_fx_available 缺失 + rope_scaling | auto_map 本地化 + 补丁 + config 修复 |
| MiniCPM NaN 评分 | CausalLM LayerWiseHead 33层 | keyword-fallback 有效 |
| 项目外文件泄漏 | 从错误目录运行命令 | 全量归位到项目目录 |
| Git push 失败 | ghfast.top 不支持 auth + 直连超时 | SSH key (git@github.com) |

### 缺失依赖

| 包 | 用途 | 建议 |
|------|------|------|
| `huggingface_hub` | 模型下载 | 加入 requirements.txt |
| `transformers` | Agent 训练 | 加入 requirements.txt |
| `PyMuPDF` | PDF 解析 | 加入 requirements.txt |
| `python-dotenv` | .env 加载 | 加入 requirements.txt |
| `openai` | LLM 调用 | pin 版本到 2.47.0 |

---

## 脚本对照

### CarVoice_Agent → 融合项目

| 原始 | 融合 |
|------|------|
| `download_models.py` | `scripts/download_models.py` |
| `train/run.py` | `scripts/train_intent.py` `train_reject.py` |
| `dialog.py` `test.py` | `scripts/run_agent.py` |
| `server.sh` | `scripts/start_all_services.sh` |
| `test/reject_client.py` | `scripts/reject_benchmark.py` |
| `test/intent_client.py` | `scripts/intent_benchmark.py` |
| `test/nlu_client.py` | `scripts/nlu_benchmark.py` |
| `test/*benchmark*.py` (locust) | `scripts/intent_qps.py` `reject_qps.py` `nlu_qps.py` |
| `e2e_score.py` | `scripts/e2e_score.py` |

### XIAOMI_SU7_RAG → 融合项目

| 原始 | 融合 |
|------|------|
| `build_index.py` | `scripts/build_index.py` |
| `generate_all_data.py` | `scripts/generate_data.py` |
| `generate_sft_data.py` | `scripts/generate_sft_data.py` |
| `final_score.py` | `scripts/eval_rag.py` |
| `infer.py` `infer_rl.py` | `POST /api/v1/chat` `scripts/rl_infer.py` |
| `deploy/auto_vllm_server.py` | `scripts/run_vllm.py` |
| `deploy/benchmark.py` | `scripts/benchmark_vllm.py` |
| `deploy/baseline_gpt4o.py` | `scripts/baseline_compare.py` |
| `evaluate_parse_quality.py` | `scripts/evaluate_parse_quality.py` |
| `check_training_data.py` | `scripts/check_training_data.py` |
| `predict.py` | `LLaMA-Factory-main/predict.py` |
| `src/client/llm_clean_client.py` | `app/llm/llm_clean.py` |
| `src/client/llm_hyde_client.py` | `app/llm/llm_hyde.py` |
| `src/server/semantic_chunk.py` | `app/knowledge/semantic_chunk_server.py` |

---

## 许可证

MIT License
