#!/usr/bin/env bash
# -------------------------------------------------------------------
# SU7_CarVoice_Fusion — 本地开发启动脚本 (Linux / macOS / Git Bash)
# -------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 1. 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo ">>> 创建虚拟环境 .venv ..."
    python3 -m venv .venv
fi

# 2. 激活虚拟环境
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

# 3. 安装依赖
echo ">>> 安装依赖 ..."
pip install -q -r requirements.txt

# 4. 准备配置文件
if [ ! -f ".env" ]; then
    echo ">>> 生成 .env（从 .env.example 复制）..."
    cp .env.example .env
fi

# 5. 启动服务
echo ">>> 启动 SU7_CarVoice_Fusion 服务 ..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
