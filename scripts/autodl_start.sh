#!/bin/bash
# =============================================================================
# AutoDL 一键启动脚本 — SU7_CarVoice_Fusion
# =============================================================================
# 用法:
#   1. 在 AutoDL 实例上克隆仓库后进入项目根目录
#   2. bash scripts/autodl_start.sh [模式]
#
# 模式:
#   mock    — 零依赖 Mock 模式，无需 GPU/API（默认）
#   vllm    — 启动 vLLM + 融合服务，需要 GPU
#   train   — 仅准备训练环境，不启动服务
# =============================================================================
set -e

MODE="${1:-mock}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  SU7_CarVoice_Fusion — AutoDL 启动"
echo "  模式: $MODE"
echo "  目录: $PROJECT_DIR"
echo "=========================================="

# ---- 1. 环境准备 ----
if [ ! -d ".venv" ]; then
    echo ">>> 创建虚拟环境..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo ">>> 安装核心依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
    echo ">>> 生成 .env 配置..."
    cp .env.example .env
fi

# ---- 2. 按模式执行 ----
case "$MODE" in
    mock)
        echo ">>> Mock 模式：零依赖启动，无需 GPU 或外部 API"
        python -c "from app.main import app; print('App 加载成功')"
        echo ""
        echo ">>> 启动服务 http://0.0.0.0:8080"
        uvicorn app.main:app --host 0.0.0.0 --port 8080
        ;;

    vllm)
        echo ">>> 安装 GPU 依赖..."
        pip install -q vllm transformers rank-bm25 jieba sentence-transformers faiss-gpu

        echo ">>> 后台启动 vLLM（模型: Qwen3-8B）..."
        vllm serve Qwen/Qwen3-8B \
            --host 0.0.0.0 --port 8000 \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.90 \
            --dtype auto &
        VLLM_PID=$!
        echo "    vLLM PID: $VLLM_PID"

        echo ">>> 等待 vLLM 就绪（约 60 秒）..."
        for i in $(seq 1 60); do
            if curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
                echo "    vLLM 就绪！"
                break
            fi
            sleep 2
            echo "    等待中... ($((i*2))s)"
        done

        # 更新配置
        sed -i 's/^LLM_PROVIDER=.*/LLM_PROVIDER=vllm/' .env
        sed -i 's|^VLLM_BASE_URL=.*|VLLM_BASE_URL=http://127.0.0.1:8000/v1|' .env
        sed -i 's/^RETRIEVER_BACKEND=.*/RETRIEVER_BACKEND=bm25/' .env

        echo ">>> 启动融合服务 http://0.0.0.0:8080"
        uvicorn app.main:app --host 0.0.0.0 --port 8080
        ;;

    train)
        echo ">>> 安装训练依赖..."
        pip install -q vllm transformers torch rank-bm25 jieba \
            sentence-transformers faiss-gpu datasets accelerate \
            peft bitsandbytes

        echo ">>> 准备训练数据..."
        python -c "
from app.data_pipeline.qa_generator import generate_qa_pairs
from app.data_pipeline.qa_filter import filter_qa_pairs
from app.data_pipeline.dataset_builder import build_summary_dataset
print('数据管道就绪')
print('训练配置见 configs/sft.yaml 和 configs/grpo.yaml')
print('用 LLaMA-Factory 启动训练：')
print('  llamafactory-cli train configs/sft.yaml')
"
        ;;

    *)
        echo "未知模式: $MODE"
        echo "可用模式: mock | vllm | train"
        exit 1
        ;;
esac
