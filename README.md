# SU7_CarVoice_Fusion

车载智能语音助手融合架构：**CarVoice_Agent**（主控框架）+ **XIAOMI_SU7_RAG**（知识检索）。

三路路由统一调度：Task（技能执行）/ FAQ（手册问答）/ Chitchat（百科闲聊）。支持 BERT 意图识别、LLM Function Calling 槽位提取、RAG 可溯源检索、Search-R1 动态工具调用与 GRPO 强化学习。

**验证环境**：RTX 4090 (48GB) ×2, Python 3.12, PyTorch 2.11, CUDA 13.2 | **验证日期**：2026-07-10 ~ 2026-07-24

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

| 级别 | 方案 | 实测准确率 |
|------|------|:---:|
| 1 | BERT 439类 (RoBERTa-wwm-ext, 31w语料) | **86.07%** Top-1 |
| 2 | 启发式规则 | ~90% 常见场景 |
| 3 | LLM 仲裁 (Doubao) | ~98% |

### 路由决策

| 分类 | 触发条件 | 管线 | 输出 |
|------|---------|------|------|
| Task | 技能指令/疑问+技能域 | LLM→DM→MCP→NLG | task_result |
| FAQ | 用户手册提问 | RAG → 引用拼装 | faq_answer + citations |
| Chitchat | 百科闲聊 | 拒识+联网→LLM | chitchat |

### 服务端口

| 服务 | 端口 | 路径 |
|------|------|------|
| 融合主服务 | 8080 | `/api/v1/chat` |
| 拒识服务 | 8007 | `/reject-server/v1` |
| 意图服务 | 8008 | `/intent-server/v1` |
| NLU 服务 | 8009 | `/chatnlu-server/v1` |
| vLLM 推理 | 8000 | `/v1/models` |
| Redis | 6379 | — |
| MongoDB | 27017 | — |

---

## 快速开始

### Mock 模式（30 秒启动）

```bash
git clone https://github.com/Ban-away/SU7_CarVoice_Fusion.git
cd SU7_CarVoice_Fusion
pip install -r requirements.txt
# 以下未列入 requirements.txt，需手动安装：
pip install huggingface_hub transformers PyMuPDF python-dotenv
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

验证：
```bash
curl http://127.0.0.1:8080/healthz
curl -X POST http://127.0.0.1:8080/api/v1/chat -d '{"message":"请导航到公司"}'
curl -X POST http://127.0.0.1:8080/api/v1/chat -d '{"message":"SU7 续航是多少"}'
curl -X POST http://127.0.0.1:8080/api/v1/chat -d '{"message":"你好"}'
```

### 切生产模式

`.env` 中设置 `LLM_PROVIDER=doubao`，填写 API Key。

### 一键启动全部服务

```bash
bash scripts/start_all_services.sh
```

---

## 训练流程

### 1. 模型下载

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --preset agent   # BERT
python scripts/download_models.py --preset rag     # RAG
```

下载策略：ModelScope（国内优先）→ HF snapshot → HF 逐文件（绕过 Xet 存储）。

### 2. Agent 训练

```bash
python scripts/train_intent.py   # 意图, 31w语料, ~12min → models/saved/intent/bert.ckpt (392MB)
python scripts/train_reject.py   # 拒识, 32w语料, ~2min → models/saved/reject/bert_tiny.ckpt (24MB)
```

### 3. RAG 索引构建

```bash
# 启动 MongoDB + Milvus
mongod --dbpath /data/db --fork --logpath /var/log/mongodb/mongod.log

# 构建索引
python scripts/build_index.py --backend all --pdf data/knowledge/Xiaomi_SU7_Manual.pdf

# 数据检查
python scripts/check_training_data.py      # 意图31w+拒识32w, 7/7通过
python scripts/evaluate_parse_quality.py   # 解析准确率 98.31%
```

### 4. SFT 微调

| 框架 | GPU | 命令 | 验证 |
|------|:---:|------|:---:|
| peft + trl | 1×16GB | `python scripts/run_sft_minimal.py` | ✅ |
| Unsloth | 1×12GB | `python scripts/train_sft_unsloth.py` | ✅ |
| LLaMA-Factory | 1×24GB | `llamafactory-cli train configs/sft.yaml` | ✅ |
| DeepSpeed ZeRO-3 | 2×48GB | `deepspeed --num_gpus=2` 原生可用 | ✅ |

> LLaMA-Factory 注意：`configs/sft.yaml` 需 `do_train: true`，注释掉 `quantization_bit`/`quantization_method`。

### 5. vLLM 部署

```bash
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000
python scripts/benchmark_vllm.py --url http://127.0.0.1:8000/v1   # 1832 tok/s, TTFT 133ms
```

### 6. GRPO 强化学习

Search-R1 范式：模型自主决定何时检索、检索什么、何时终止，而非硬编码 pipeline。

**五标签行动空间**：`<search_local>` / `<search_web>` / `<read_page>` / `<information>` / `<answer>`

**三级检索路由**：本地知识库 → 网络搜索 → 深度页面阅读 (WebWalker, 最多2跳)

**6维奖励函数**：答案质量(0.40) + 工具合理性(0.15) + 探索深度(0.15) + 领域合规(0.15) + 来源标注(0.10) + 格式完整性(0.05)

```bash
# 1. 生成轨迹
python app/rl/data_builder.py --dry-run          # 网络兜底轨迹 (需 SerpAPI)
python app/rl/build_local_trajectories.py         # 本地可答轨迹

# 2. 格式转换 + 再平衡
python app/rl/format_converter.py
python app/rl/rebalance_sft_data.py

# 3. GRPO 训练 (双框架)
#   方案A — TRL 单卡快速验证
python app/rl/train_grpo.py --stage grpo

#   方案B — VeRL 多卡生产训练 (需 flash-attn)
python app/rl/train_grpo_verl.py --n-gpus 2

# 4. 导出模型
python app/rl/train_grpo.py --stage export

# 5. RL 推理 (Search-R1 自主检索)
vllm serve LLaMA-Factory-main/output/qwen3_lora_rl --port 8000 --served-model-name su7_rl
python scripts/rl_infer.py --model su7_rl --show-trajectory

# 6. RL 批量评测
python app/rl/batch_eval.py --model su7_rl --vllm-url http://localhost:8000/v1 --dry-run
```

---

## 验证结果

### BERT 精度

| 指标 | 实测 | 原始报告 |
|------|:---:|:---:|
| 意图 Top-1 | **86.07%** | 85.2% |
| 意图 Top-5 | **97.64%** | 97.6% |
| 拒识 Accuracy | **89.56%** | 89.7% |

### RAG 评估 (predict.py, 1272条)

| 指标 | 实测 |
|------|:---:|
| RAGAs context_recall | **88.47%** |
| RAGAs context_precision | **97.64%** |

### 性能

| 指标 | 值 |
|------|-----|
| vLLM 吞吐率 | **1832 tok/s** |
| TTFT 均值 | **133ms** |
| 文档解析准确率 | **98.31%** |

### E2E 批量 (242条)

| 类型 | 占比 |
|------|------|
| task_result | 79% |
| chitchat | 17% |
| faq_answer | 3% |

### QPS

| 服务 | 中位数 |
|------|:---:|
| 拒识 | **89ms** |
| 意图 | **180ms** |
| NLU | **1483ms** |

### 全部验证清单

| 模块 | 状态 | 说明 |
|------|------|------|
| Mock 推理 | ✅ | 5/5 curl |
| pytest | ✅ | 63/63 |
| BERT 意图 | ✅ | 86.07% |
| BERT 拒识 | ✅ | 89.56% |
| NLU 评测 | ✅ | 服务在线 |
| E2E 批量 | ✅ | 242条 |
| QPS 压测 ×3 | ✅ | |
| BM25 索引 | ✅ | 144 chunks |
| FAISS 索引 | ✅ | |
| Milvus 混合索引 | ✅ | Dense+Sparse |
| PDF 解析 + 语义切分 | ✅ | 98.31% |
| 联网搜索 | ✅ | SerpAPI→Serper→Bing→Doubao |
| HyDE 扩写 + LLM 清洗 | ✅ | 豆包 API |
| SFT (3框架) | ✅ | peft+Unsloth+LLaMA-Factory |
| DeepSpeed ZeRO-3 | ✅ | 双卡 Qwen3-8B 16s |
| vLLM 部署 | ✅ | INT4 5.7GB |
| vLLM 压测 | ✅ | 1832 tok/s |
| RL 推理 | ✅ | 五标签 + read_page |
| RL 数据管线 | ✅ | 轨迹生成+转换 |
| GRPO 训练 (TRL) | ✅ | 5 steps |
| predict.py | ✅ | 88.47%/97.64% |
| VeRL 框架 | ✅ | Ray+NCCL+FSDP 初始化通过 |
| 全部模型下载 | ✅ | 8/8 |
| Redis / MongoDB / Milvus | ✅ | |
| start_all_services | ✅ | 5/5 |

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/
│   ├── api/                HTTP + WebSocket
│   ├── core/               编排器 (orchestrator, classifier, session)
│   ├── nlp/                NLP (intent, reject, NLU, NLG, arbitration, rewrite, correlation)
│   ├── skills/             技能定义 + DM (maps/music/weather)
│   ├── knowledge/          RAG (BM25/FAISS/Milvus + MiniCPM + chunker + web_search)
│   ├── llm/                LLM (doubao/vllm/openai/mock + llm_clean + llm_hyde)
│   ├── mcp/                MCP (Amap 13工具+QQ音乐)
│   ├── train/              BERT 训练 (core/models/servers)
│   ├── rl/                 Search-R1 (data_builder, train_grpo, infer_rl, web_reader)
│   ├── data_pipeline/      QA生成/过滤
│   └── shared/             配置/日志/Redis
├── scripts/                30+ 执行脚本
├── configs/                sft/grpo/ds/export
├── data/                   训练数据 + 知识库
├── models/                 预训练模型 + BERT checkpoints
├── LLaMA-Factory-main/     训练框架 + output/(SFT/INT4/RL)
├── mongodb-7.0.20/         MongoDB 二进制
├── tests/                  63 单元测试
├── docs/                   架构文档
└── CLAUDE.md
```

---

## 脚本对照

| 原始 | 融合 |
|------|------|
| CarVoice `train/run.py` | `scripts/train_intent.py` `train_reject.py` |
| CarVoice `server.sh` | `scripts/start_all_services.sh` |
| CarVoice `test/*client*.py` | `scripts/*benchmark*.py` |
| CarVoice `e2e_score.py` | `scripts/e2e_score.py` |
| SU7_RAG `build_index.py` | `scripts/build_index.py` |
| SU7_RAG `generate_all_data.py` | `scripts/generate_data.py` |
| SU7_RAG `final_score.py` | `scripts/eval_rag.py` |
| SU7_RAG `infer_rl.py` | `scripts/rl_infer.py` |
| SU7_RAG `deploy/auto_vllm_server.py` | `scripts/run_vllm.py` |
| SU7_RAG `deploy/benchmark.py` | `scripts/benchmark_vllm.py` |
| SU7_RAG `predict.py` | `LLaMA-Factory-main/predict.py` |
| SU7_RAG `llm_clean_client.py` | `app/llm/llm_clean.py` |
| SU7_RAG `llm_hyde_client.py` | `app/llm/llm_hyde.py` |

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话 |
| `GET` | `/api/v1/skills` | 技能白名单 |
| `POST` | `/api/v1/knowledge/retrieve` | 知识检索 |

---

## 遇到的问题与修复

### 代码缺陷

| # | 问题 | 修复 |
|---|------|------|
| 1 | `app/train/models/` 单文件非包 | 创建 bert.py + bert_tiny.py |
| 2 | 多个脚本相对路径失效 | 统一 `os.chdir()` |
| 3 | Doubao API Key 未注入 | `app/llm/base.py` 自动读取 settings |
| 4 | `app/knowledge/chunker.py` 递归溢出 | 前向进度 guard |
| 5 | `app/knowledge/web_search.py` 仅 mock | 升级为 SerpAPI 四级级联 |
| 6 | `configs/sft.yaml` 缺 do_train | `do_train: true` |
| 7 | `app/rl/infer_rl.py` 缺 RLInferenceEngine | 添加 wrapper 类 |
| 8 | `app/rl/train_grpo.py` SFT warmup 调用 LLaMA-Factory | 已有 adapter 自动跳过 |
| 9 | `app/knowledge/retriever/faiss.py` 缺 save/load | 补全序列化 |
| 10 | `app/knowledge/retriever/milvus.py` 模型路径+Sparse格式 | 绝对路径+SPLADE fallback |

### 依赖与兼容

| 问题 | 解决 |
|------|------|
| HF Xet 存储下载 401 | ModelScope + hf_hub_download 逐文件 |
| LLaMA-Factory + transformers 5.x | `is_torch_sdpa_available`/`is_safetensors_available` 补丁 |
| vLLM + ragas openai 冲突 | 锁定 `openai==2.47.0` |
| MongoDB OpenSSL 不兼容 | 捆绑 libssl1.1 |
| Milvus URI pymilvus 2.3+ 不兼容 | milvus-lite 本地服务 |
| BGE/SPLADE/m3e config 缺失 | 权重反推架构 + config 补全 |
| MiniCPM transformers 5.x 导入 | auto_map 本地化 + 函数补丁 |
| predict.py ragas API 变更 | Dataset 替换 + 指标重命名 |

### 缺失依赖需手动安装

`huggingface_hub` `transformers` `PyMuPDF` `python-dotenv`

---

## 许可证

MIT License
