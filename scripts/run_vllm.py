#!/usr/bin/env python
"""Auto-detect GPU count and start vLLM server.

Ported from XIAOMI_SU7_RAG/deploy/auto_vllm_server.py。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def detect_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Start vLLM server with auto GPU detection")
    parser.add_argument("--model", default="models/Qwen3-8B", help="Model path or HF repo ID")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    args, extra = parser.parse_known_args()

    gpu_count = detect_gpu_count()
    print(f"Detected {gpu_count} GPU(s)")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
    ]
    if gpu_count >= 2:
        cmd += ["--tensor-parallel-size", str(gpu_count)]

    cmd += extra
    print(f"Starting: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
