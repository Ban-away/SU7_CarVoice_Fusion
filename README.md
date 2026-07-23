# SU7_CarVoice_Fusion

> 车载智能语音助手融合架构：**CarVoice_Agent**（主控框架）+ **XIAOMI_SU7_RAG**（知识检索）

三路路由统一调度：Task（技能执行）/ FAQ（手册问答）/ Chitchat（百科闲聊）。支持 BERT 意图识别、LLM Function Calling 槽位提取、RAG 可溯源检索、Search-R1 动态工具调用与 GRPO 强化学习。

**验证环境**：RTX 4090 (48GB), Python 3.12, PyTorch 2.11, CUDA 13.2 | **验证日期**：2026-07-10 ~ 2026-07-23

---

## 目录

1. [整体架构](#整体架构)
2. [快速开始](#快速开始)
3. [训练流程](#训练流程)
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
| 1 | BERT 439 类 | 86.07% top-1 | RoBERTa-wwm-ext，31w 训练 |
| 2 | 启发式规则 | ~90% 常见场景 | 疑问语气 vs 祈使指令 |
| 3 | LLM 仲裁 | ~98% | Doubao 182 行 Prompt |

### 路由决策

| 分类 | 触发条件 | 管线 | 输出 |
|------|---------|------|------|
| Task | 技能指令/疑问+技能域 | LLM→DM→MCP→NLG | task_result |
| FAQ | 用户手册提问 | RAG→引用拼装 | faq_answer+citations |
| Chitchat | 百科闲聊 | 拒识+联网→LLM | chitchat |

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
curl http://127.0.0.1:8080/healthz                          # → {"status":"ok"}
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"请导航到公司"}'  # → task_result
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"SU7 续航是多少"}'  # → faq_answer
curl -X POST http://127.0.0.1:8080/api/v1/chat -H "Content-Type: application/json" -d '{"message":"你好"}'  # → chitchat
```

### 切生产模式

`.env` 中修改：
```bash
LLM_PROVIDER=doubao
DOUBAO_API_KEY=ark-xxx
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL_NAME=doubao-1-5-lite-32k-250115
```

### 一键启动全部服务

```bash
bash scripts/start_all_services.sh
# 启动: 语义切分(6000) + 拒识(8007) + 意图(8008) + NLU(8009) + 主服务(8080)
```

---

## 训练流程

### 模型下载

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --preset agent   # BERT 模型
python scripts/download_models.py --preset rag     # RAG 模型
```

> 下载策略：ModelScope（国内优先）→ HF snapshot → HF 逐文件（绕过 Xet 存储 401）

### Agent 训练

```bash
python scripts/train_intent.py   # 意图分类 (31w 语料, 3 epoch)
python scripts/train_reject.py   # 拒识模型 (32w 语料, 3 epoch)
```

### SFT 微调（三种框架）

| 框架 | 显存 | 验证结果 |
|------|------|------|
| peft + trl (FP16) | ≥16GB | ✅ loss 3.39→1.73 |
| Unsloth 2x | ≥12GB | ✅ loss 2.39→2.08 |
| LLaMA-Factory | ≥24GB | ✅ loss 4.86 |

```bash
# peft + trl (最简)
python scripts/run_sft_minimal.py

# Unsloth (最快)
python scripts/train_sft_unsloth.py

# LLaMA-Factory (最稳定; 需 do_train:true + 注释量化)
llamafactory-cli train configs/sft.yaml
llamafactory-cli export configs/export.yaml
```

> ⚠️ LLaMA-Factory 注意：`do_train: true`，注释掉 `quantization_bit`/`quantization_method`（CUDA 13 无 libnvJitLink）

### RAG 索引构建

```bash
# BM25 索引（零依赖）
python scripts/build_index.py --backend bm25

# 全量索引（需 MongoDB + Milvus + m3e-small）
python scripts/build_index.py --backend all
```

### GRPO 强化学习

```bash
# 生成轨迹 (需 SerpAPI + Doubao)
python app/rl/data_builder.py --dry-run          # 快速验证
python app/rl/build_local_trajectories.py         # 本地轨迹

# 格式转换 + GRPO 训练
python app/rl/format_converter.py
python app/rl/train_grpo.py --stage grpo          # SFT adapter 已有时自动跳过 warmup
```

### vLLM 部署

```bash
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000
python scripts/benchmark_vllm.py --url http://127.0.0.1:8000/v1
python scripts/rl_infer.py --model su7_rl --show-trajectory
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

### 性能压测

| 指标 | 值 |
|------|-----|
| vLLM 吞吐率 | **1832 tok/s** |
| TTFT 均值 | **133ms** |
| QPS 拒识/意图/NLU | 89ms / 180ms / 1483ms |
| 基线对比 | 0.88s/请求 |

### E2E 批量评测

| 类型 | 数量 | 占比 |
|------|------|------|
| task_result | 192 | 79% |
| chitchat | 41 | 17% |
| faq_answer | 8 | 3% |
| clarification | 1 | 0% |
| 平均延迟 | 4138ms | — |

### 其他验证

| 模块 | 状态 | 关键结果 |
|------|------|------|
| pytest | ✅ | 63/63 |
| 文档解析质量 | ✅ | **98.31%** |
| 训练数据检查 | ✅ | 意图31w+拒识32w, 7/7通过 |
| LLaMA-Factory export | ✅ | 4 shards, 15s |
| VeRL | ✅ | v0.8.0 |
| Redis | ✅ | v6.0.16 |
| PDF LLM 清洗 | ✅ | 豆包 API，已移植 `app/llm/llm_clean.py` |
| 语义切分 (m3e-small) | ✅ | 逐文件下载+config补全, 语义聚类正常 |
| Milvus 混合索引 | ✅ | Dense(BGE-Large 1024-dim) + Sparse(SPLADE), 144 entities |
| SPLADE Sparse 检索 | ✅ | config反推: BERT-base 12层/768维 |
| HyDE 查询扩写 | ✅ | 豆包 API，已移植 `app/llm/llm_hyde.py` |
| MongoDB | ✅ | v7.0.20 (捆绑 OpenSSL 1.1) |

### 全部模型下载

| 模型 | 大小 | 方式 |
|------|------|------|
| Qwen3-8B | 15.5GB | ModelScope |
| bge-large-zh-v1.5 | 1.3GB | ModelScope |
| bge-reranker-v2-minicpm | ~7GB | HF mirror |
| chinese_roberta_wwm_ext | ~400MB | HF mirror |
| m3e-small | ~96MB | HF 逐文件 (含语义切分配置补全) |
| splade-v2 | ~400MB | HF mirror |
| text2vec-base-chinese | 391MB | HF 逐文件 |
| roberta_tiny_clue | 25MB | 手动上传 |

---

## 项目结构

```
SU7_CarVoice_Fusion/
├── app/                    核心代码
│   ├── api/                HTTP + WebSocket 网关
│   ├── core/               编排器 (orchestrator, classifier, session)
│   ├── nlp/                NLP (intent, reject, NLU, NLG, arbitration, rewrite)
│   ├── skills/             技能定义 + DM (maps/music/weather)
│   ├── knowledge/          RAG (8 retriever + 5 reranker + chunker)
│   ├── llm/                LLM 客户端 (doubao/vllm/openai/mock)
│   ├── mcp/                MCP 扩展 (Amap 13工具+QQ音乐)
│   ├── prompts/            7 System Prompts
│   ├── train/              BERT 训练 (core/models/servers)
│   ├── rl/                 Search-R1 强化学习
│   ├── data_pipeline/      QA生成/过滤/训练集
│   └── shared/             配置/日志/Redis/工具
├── scripts/                30+ 执行脚本
├── configs/                4 配置 (sft/grpo/ds/export)
├── data/                   训练数据 + 知识库
├── models/                 预训练模型 (8个) + BERT checkpoints
├── LLaMA-Factory-main/     训练框架 + output/ (SFT 16GB/INT4 5.7GB/RL 16GB)
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
| `RETRIEVER_BACKEND` | mock | mock / bm25 / milvus / hybrid |
| `SERPAPI_KEY` | — | 网络搜索 |
| `AMAP_API_KEY` | — | 高德地图 |
| `REJECT_URL` | http://127.0.0.1:8007/reject-server/v1 | |
| `INTENT_URL` | http://127.0.0.1:8008/intent-server/v1 | |
| `NLU_URL` | http://127.0.0.1:8009/chatnlu-server/v1 | |

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `POST` | `/api/v1/chat` | 单轮对话 |
| `GET` | `/api/v1/skills` | 技能白名单 |
| `POST` | `/api/v1/knowledge/retrieve` | 知识检索 |
| `WS` | `/ws/chat` | WebSocket |

---

## 遇到的问题与解决

### 代码缺陷（15处修复）

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `app/train/models/` | 单文件非包 | 创建 bert.py + bert_tiny.py |
| 2 | `app/train/run.py` | 缺 os.chdir + 导入错误 | 添加 chdir + 统一 import |
| 3 | `scripts/build_index.py` | chunker.split() 不存在 | 改用 chunk_text() |
| 4 | `app/knowledge/chunker.py` | _split() 递归溢出 | 前向进度 guard |
| 5 | `app/nlp/intent.py` | forward 签名不匹配 | forward(input_ids, mask) |
| 6 | `app/llm/base.py` | Doubao API Key 未注入 | settings 自动读取 |
| 7 | `app/llm/doubao.py` | 参数命名不规范 | 统一 base_url/model_name |
| 8 | `app/shared/config.py` | .env 未加载 | python-dotenv 加载 |
| 9 | `app/train/data_loader.py` | HF tokenizer 不兼容 | 自定义 BertTokenizer |
| 10 | `scripts/download_models.py` | 仅 HF mirror | ModelScope→snapshot→逐文件 |
| 11 | `models/roberta_tiny_clue/` | config 不匹配权重 | hidden_size=312, vocab=8021 |
| 12 | `scripts/*.py` (多个) | 相对路径 | 统一 os.chdir() |
| 13 | `app/rl/infer_rl.py` | 缺 RLInferenceEngine | 添加 wrapper 类 |
| 14 | `app/rl/train_grpo.py` | SFT warmup 依赖 BNB | 已有 adapter 自动跳过 |
| 15 | `configs/sft.yaml` | 缺 do_train, 量化卡住 | do_train:true, 注释量化 |

### 依赖问题与解决

| 问题 | 根因 | 解决方案 |
|------|------|------|
| HF 模型下载 401 | hf-mirror 不支持 Xet 存储 | ModelScope + hf_hub_download 逐文件 |
| bitsandbytes 不可用 | CUDA 13 缺 libnvJitLink | SFT/GRPO 改用 FP16 |
| LLaMA-Factory import 失败 | transformers 5.x 删除了旧函数 | 补丁 is_torch_sdpa_available/is_safetensors_available |
| numpy≥2.0 不兼容 LLaMA-Factory | np.long/np.ulong 被删除 | sitecustomize.py 别名补丁 |
| MongoDB 无法启动 | 二进制是 Ubuntu22.04 (OpenSSL 1.1) | 提取捆绑的 libssl1.1 .deb |
| m3e-small 权重缺失 | Xet 存储 + ModelScope 未收录 | hf_hub_download 逐文件 + 补全 tokenizer/config |
| BGE config 不兼容 | 缺 BertConfig 字段 | 根据权重反推架构: 24层/1024维/16头 |
| Milvus URI 不兼容 | pymilvus 2.3+ 要求 HTTP | 启动 milvus-lite 本地服务 localhost:19530 |
| vLLM + ragas openai 冲突 | `requirements.txt` 里 openai>=1.0.0 没 pin 版本 | 锁定 openai==2.47.0 (vLLM + ragas 都兼容) |
| predict.py API 不兼容 | ragas 0.1.x vs 0.3.x API 变更 | EvaluationDataset→Dataset, LLMContextRecall→ContextRecall |

### 缺失依赖（需手动安装）

| 包 | 用途 |
|------|------|
| `huggingface_hub` | 模型下载 |
| `transformers` | Agent 训练 |
| `PyMuPDF` | PDF 解析 |
| `python-dotenv` | .env 加载 |

---

## 脚本对照

| 原始项目 | 原始文件 | 融合后 |
|---------|---------|--------|
| **CarVoice** | `download_models.py` | `scripts/download_models.py` |
| **CarVoice** | `train/run.py` | `scripts/train_intent.py` `train_reject.py` |
| **CarVoice** | `dialog.py` `test.py` | `scripts/run_agent.py` |
| **CarVoice** | `server.sh` | `scripts/start_all_services.sh` |
| **CarVoice** | `test/reject_client.py` | `scripts/reject_benchmark.py` |
| **CarVoice** | `test/intent_client.py` | `scripts/intent_benchmark.py` |
| **CarVoice** | `test/nlu_client.py` | `scripts/nlu_benchmark.py` |
| **CarVoice** | `test/*benchmark*.py` (locust) | `scripts/intent_qps.py` `reject_qps.py` `nlu_qps.py` |
| **CarVoice** | `e2e_score.py` | `scripts/e2e_score.py` |
| **SU7_RAG** | `build_index.py` | `scripts/build_index.py` |
| **SU7_RAG** | `generate_all_data.py` | `scripts/generate_data.py` |
| **SU7_RAG** | `generate_sft_data.py` | `scripts/generate_sft_data.py` |
| **SU7_RAG** | `final_score.py` | `scripts/eval_rag.py` |
| **SU7_RAG** | `infer.py` `infer_rl.py` | `POST /api/v1/chat` `scripts/rl_infer.py` |
| **SU7_RAG** | `deploy/auto_vllm_server.py` | `scripts/run_vllm.py` |
| **SU7_RAG** | `deploy/benchmark.py` | `scripts/benchmark_vllm.py` |
| **SU7_RAG** | `deploy/baseline_gpt4o.py` | `scripts/baseline_compare.py` |
| **SU7_RAG** | `evaluate_parse_quality.py` | `scripts/evaluate_parse_quality.py` |
| **SU7_RAG** | `check_training_data.py` | `scripts/check_training_data.py` |
| **SU7_RAG** | `predict.py` | `LLaMA-Factory-main/predict.py` |

---

## 许可证

MIT License
