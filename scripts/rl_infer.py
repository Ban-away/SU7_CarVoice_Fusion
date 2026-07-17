#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL-enhanced inference entry point — Search-R1 generate-while-retrieving.

Ported from XIAOMI_SU7_RAG/src/rl/infer_rl.py.

Usage:
  # Interactive mode (needs vLLM running with RL model)
  python scripts/rl_infer.py --model su7_rl --show-trajectory

  # Single query test
  python scripts/rl_infer.py --model su7_rl --query "小米SU7续航多少"

  # Show 6-dimension reward
  python scripts/rl_infer.py --model su7_rl --query "..." --show-reward
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import logging
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.shared.config import get_settings
from app.shared.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-enhanced inference — Search-R1 paradigm")
    parser.add_argument(
        "--model", default=None,
        help="vLLM model name (default from env RL_MODEL_NAME or 'qwen3-rl-su7')",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="vLLM service address",
    )
    parser.add_argument(
        "--query", help="Single query test (non-interactive mode)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive mode (default when no --query)",
    )
    parser.add_argument(
        "--show-trajectory", action="store_true",
        help="Show full trajectory (including retrieval process)",
    )
    parser.add_argument(
        "--show-reward", action="store_true",
        help="Show 6-dimension reward breakdown",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show intermediate generation steps",
    )

    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    # Determine model name
    model_name = args.model or os.getenv("RL_MODEL_NAME", "qwen3-rl-su7")

    # ── Initialize clients ──────────────────────────────────
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] OpenAI SDK not installed. Install with: pip install openai")
        sys.exit(1)

    llm_client = OpenAI(
        api_key="EMPTY",
        base_url=args.vllm_url,
    )

    from app.rl.environment import RetrievalEnvironment
    from app.rl.reward_model import compute_reward

    logger.info("Loading retrieval environment...")
    env = RetrievalEnvironment()
    logger.info("Retrieval environment ready")

    # Import the inference function
    from app.rl.infer_rl import run_rl_inference

    # ── Single query mode ───────────────────────────────────
    if args.query:
        print(f"Model: {model_name}")
        print(f"Query: {args.query}")
        print(f"{'─' * 60}")

        start_time = time.time()
        result = run_rl_inference(
            question    = args.query,
            llm_client  = llm_client,
            env         = env,
            model_name  = model_name,
            verbose     = args.verbose,
        )
        elapsed = time.time() - start_time

        print(f"\n{'─' * 60}")
        print(f"Answer: {result['answer']}")
        print(f"Time: {elapsed:.1f}s | Rounds: {result['rounds']} | "
              f"Retrievals: local×{result['search_calls']['local']} "
              f"web×{result['search_calls']['web']} "
              f"read_page×{result['search_calls']['read_page']}")

        if args.show_reward:
            detail = result["reward_detail"]
            print(f"Reward: {result['reward']:.3f} "
                  f"(format:{detail['format_score']:.2f} answer:{detail['answer_score']:.2f} "
                  f"tool:{detail['tool_score']:.2f} source:{detail['source_score']:.2f} "
                  f"domain:{detail['domain_score']:.2f} "
                  f"exploration:{detail.get('exploration_score', 0):.2f})")

        if args.show_trajectory:
            print(f"\nFull trajectory:")
            print(f"{'─' * 70}")
            for line in result["trajectory"].split("\n"):
                print(f"  {line}")
            print(f"{'─' * 70}")

        return

    # ── Interactive mode ────────────────────────────────────
    print("=" * 80)
    print("Xiaomi SU7 RL-Enhanced Inference (Search-R1 Paradigm)")
    print("=" * 80)
    print(f"  Model:     {model_name}")
    print(f"  vLLM URL:  {args.vllm_url}")
    print(f"  Paradigm:  Generate-while-retrieving")
    print(f"  Tools:     <search_local> / <search_web> / <read_page>")
    print("=" * 80)
    print("  Enter a question or type ':q' to exit")
    print("=" * 80)

    while True:
        try:
            query = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query in (":q", ":quit", ":exit"):
            print("Goodbye!")
            break
        if not query:
            continue

        start_time = time.time()
        result = run_rl_inference(
            question    = query,
            llm_client  = llm_client,
            env         = env,
            model_name  = model_name,
            verbose     = args.verbose,
        )
        elapsed = time.time() - start_time

        print(f"\n{result['answer']}")
        print(f"({elapsed:.1f}s | rounds:{result['rounds']} | "
              f"local×{result['search_calls']['local']} "
              f"web×{result['search_calls']['web']} "
              f"read×{result['search_calls']['read_page']})")

        if args.show_reward:
            detail = result["reward_detail"]
            print(f"Reward: {result['reward']:.3f} "
                  f"(format:{detail['format_score']:.2f} answer:{detail['answer_score']:.2f} "
                  f"tool:{detail['tool_score']:.2f} source:{detail['source_score']:.2f} "
                  f"domain:{detail['domain_score']:.2f} "
                  f"exploration:{detail.get('exploration_score', 0):.2f})")

        if args.show_trajectory:
            print(f"\nFull trajectory:")
            print(f"{'─' * 70}")
            for line in result["trajectory"].split("\n"):
                print(f"  {line}")
            print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
