# -------------------------------------------------------------------
# SU7_CarVoice_Fusion — 本地开发启动脚本 (Windows PowerShell)
# -------------------------------------------------------------------
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Set-Location $ProjectDir

# 1. 检查虚拟环境
if (-not (Test-Path ".venv")) {
    Write-Host ">>> 创建虚拟环境 .venv ..."
    python -m venv .venv
}

# 2. 激活虚拟环境
& .\.venv\Scripts\Activate.ps1

# 3. 安装依赖
Write-Host ">>> 安装依赖 ..."
pip install -q -r requirements.txt

# 4. 准备配置文件
if (-not (Test-Path ".env")) {
    Write-Host ">>> 生成 .env（从 .env.example 复制）..."
    Copy-Item .env.example .env
}

# 5. 启动服务
Write-Host ">>> 启动 SU7_CarVoice_Fusion 服务 ..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
