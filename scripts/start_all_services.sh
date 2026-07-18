#!/bin/bash
# =============================================================================
# SU7_CarVoice_Fusion 一键启动全部服务
# 对应 CarVoice_Agent/server.sh + XIAOMI_SU7_RAG 的 MongoDB + vLLM
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "  SU7_CarVoice_Fusion — 启动全部服务"
echo "=========================================="

# 可选：MongoDB（端口 27017）
# mongod --dbpath /data/db --fork --logpath /var/log/mongodb.log 2>/dev/null &

# 可选：语义切分服务（端口 6000）
echo "[1/5] 启动语义切分服务 (6000)..."
python app/knowledge/semantic_chunk_server.py &
sleep 2

# 可选：拒识服务（端口 8007）
echo "[2/5] 启动拒识服务 (8007)..."
python -m uvicorn app.train.servers:reject_app --host 0.0.0.0 --port 8007 &
sleep 2

# 可选：意图服务（端口 8008）
echo "[3/5] 启动意图服务 (8008)..."
python -m uvicorn app.train.servers:intent_app --host 0.0.0.0 --port 8008 &
sleep 2

# 可选：NLU服务（端口 8009）
echo "[4/5] 启动 NLU 服务 (8009)..."
python -m uvicorn app.train.servers:nlu_app --host 0.0.0.0 --port 8009 &
sleep 2

# 融合主服务（端口 8080）
echo "[5/5] 启动融合主服务 (8080)..."
uvicorn app.main:app --host 0.0.0.0 --port 8080 &

echo ""
echo "服务启动完成："
echo "  语义切分:  http://0.0.0.0:6000"
echo "  拒识服务:  http://0.0.0.0:8007"
echo "  意图服务:  http://0.0.0.0:8008"
echo "  NLU 服务:  http://0.0.0.0:8009"
echo "  融合主服务: http://0.0.0.0:8080"
echo ""
echo "停止全部: pkill -f 'uvicorn|semantic_chunk_server'"
