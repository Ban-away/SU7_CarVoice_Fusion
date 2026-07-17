# -*- coding: utf-8 -*-
"""
RL-enhanced inference — Search-R1 paradigm: generate while retrieving.

Core mechanism (Search-R1 paradigm):
  The model generates text freely; the system intercepts tool-call tags via
  stop tokens:
    1. Model generates "<search_local>keyword" -> vLLM stops at "</search_local>"
    2. System calls local retrieval backend, obtains real results
    3. Injects "<information>real results</information>" as an assistant message
    4. Model continues generating, may trigger "<search_web>" -> same intercept + web search
    5. Finally model generates "<answer>...</answer>", this round ends

Difference from standard RAG inference:
  - Standard: retrieve -> rerank -> generate (fixed pipeline, model passively receives context)
  - RL-enhanced: model actively decides when and what to retrieve (autonomous tool-call loop)

Usage:
  # Start vLLM server first (RL model)
  python scripts/run_vllm.py --model MODEL_NAME --port 8000

  # Launch interactive Q&A
  python app/rl/infer_rl.py

  # Specify vLLM address
  python app/rl/infer_rl.py --vllm-url http://localhost:8000/v1
"""

import os
import re
import time
import argparse
import logging

from app.shared.config import get_settings
from app.rl.environment import RetrievalEnvironment
from app.rl.reward_model import compute_reward

logger = logging.getLogger(__name__)

# ── System prompt (consistent with training) ─────────────────
SYSTEM_PROMPT = """你是小米SU7车型的专业问答助手，服务范围严格限定在小米SU7相关问题。

回答问题时可以调用以下工具：
- 本地知识库检索（优先）：<search_local>检索关键词</search_local>
- 网络搜索（本地信息不足时）：<search_web>检索关键词</search_web>
- 页面深度阅读（搜索结果不够详细时）：<read_page>URL地址</read_page>

工具返回格式：<information>检索结果内容</information>

最终答案格式：<answer>答案内容</answer>

注意：
1. 优先调用本地知识库，本地无结果或信息严重不足时再调用网络搜索
2. 网络搜索结果中包含"网址："字段，可选择最有价值的页面用 <read_page> 深入阅读，最多读取2个页面
3. 与小米SU7无关的问题（闲聊、百科、娱乐等），直接输出 <answer>很抱歉，我只能回答小米SU7相关问题。</answer>
4. 网络搜索结果来源于互联网，答案中需注明"根据网络信息"
5. 涉及页码引用时格式为【页码】"""

# ── Inference parameters ──────────────────────────────────────
MAX_GENERATE_ROUNDS  = 12   # max generation rounds (prevents infinite loop)
MAX_TOKENS_PER_ROUND = 512  # max tokens per generation round
MAX_SEARCH_STEPS     = 4    # max tool calls per single inference
MAX_READ_PAGE_HOPS   = 2    # max deep page reads (vertical search)

# Key: stop at search tag closing, pausing model for real result injection
SEARCH_STOP_TOKENS = ["</search_local>", "</search_web>", "</read_page>"]

# Time-sensitive keywords: when question contains these, local manual likely stale,
# hint model to use web search
TIME_SENSITIVE_KEYWORDS = (
    "最新", "当前", "现在", "新版", "目前", "刚发布", "刚上市",
    "版本号", "什么时候", "多少钱", "价格", "上市", "交付量", "销量",
)

# ── Tag regex patterns (consistent with environment.py) ──────
_RE_ANSWER       = re.compile(r"<answer>(.*?)</answer>",            re.DOTALL)
_RE_SEARCH_LOCAL = re.compile(r"<search_local>(.*?)</search_local>", re.DOTALL)
_RE_SEARCH_WEB   = re.compile(r"<search_web>(.*?)</search_web>",   re.DOTALL)
_RE_READ_PAGE    = re.compile(r"<read_page>(.*?)</read_page>",     re.DOTALL)


# ────────────────────────────────────────────────────────────
# Core: Search-R1 generate-while-retrieving loop
# ────────────────────────────────────────────────────────────

def run_rl_inference(
    question:    str,
    llm_client,          # OpenAI client (connected to vLLM)
    env:         RetrievalEnvironment,
    model_name:  str,
    verbose:     bool = True,
) -> dict:
    """
    Search-R1 paradigm inference: model autonomously generates search queries,
    system intercepts and injects real retrieval results.

    Loop flow:
      round 1: model -> "<search_local>keyword" (stop at </search_local>)
               system -> execute local retrieval -> inject <information>...</information>
      round 2: model -> "<search_web>keyword" (stop at </search_web>)
               system -> execute web search -> inject <information>...</information>
      round 3: model -> "<answer>final answer</answer>"
               system -> detect <answer>, end

    Args:
        question:    User question
        llm_client:  OpenAI client (connected to vLLM)
        env:         Retrieval environment
        model_name:  Model name
        verbose:     Whether to print intermediate steps

    Returns:
        {
            "question":      str,
            "trajectory":    str,   # full trajectory
            "answer":        str,   # plain answer
            "reward":        float, # reward score
            "reward_detail": dict,  # reward breakdown
            "rounds":        int,   # actual generation rounds
            "search_calls":  dict,  # tool call statistics
        }
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    trajectory = ""
    rounds     = 0
    read_page_count = 0      # read_page call counter (for hop limit)
    total_search_count = 0   # all tool call counter

    for round_idx in range(MAX_GENERATE_ROUNDS):
        rounds += 1

        # ── Call model for generation ─────────────────────────
        # Don't use stop=["</answer>"] — vLLM would swallow </answer>,
        # breaking _RE_ANSWER detection and making is_done always False.
        # Only stop at search tag closures, intercept and inject real results.
        try:
            completion = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=MAX_TOKENS_PER_ROUND,
                temperature=0.3,
                top_p=0.9,
                stop=SEARCH_STOP_TOKENS,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )
            generated = completion.choices[0].message.content or ""
            finish_reason = completion.choices[0].finish_reason
        except Exception as e:
            logger.error("Generation failed: %s", e)
            break

        trajectory += generated
        if verbose:
            print(generated, end="", flush=True)

        # ── Check if model already generated <answer>...</answer> (natural end) ──
        if _RE_ANSWER.search(trajectory):
            if verbose:
                print()  # newline
            break

        # ── Check if search tag was triggered (stop hit) ─────
        hit_search = (finish_reason == "stop")

        if hit_search:
            # vLLM truncated at stop token; need to add back closing tag
            web_note = ""   # web search fallback hint (display-only, not in model context)

            if "<search_local>" in generated and "</search_local>" not in generated:
                # ── Total search count limit ──
                total_search_count += 1
                if total_search_count > MAX_SEARCH_STEPS:
                    info_block = (
                        "<information>已达到最大检索次数限制。</information>\n"
                        "<answer>根据已检索到的信息暂时无法给出完整答案，"
                        "建议访问小米汽车官网获取最新信息。</answer>"
                    )
                    trajectory += "</search_local>\n" + info_block + "\n"
                    messages.append({"role": "assistant", "content": generated + "</search_local>\n" + info_block + "\n"})
                    if verbose:
                        print(f"</search_local>\n{info_block}\n", end="", flush=True)
                    break

                trajectory += "</search_local>"
                if verbose:
                    print("</search_local>", end="", flush=True)
                # Extract search keyword
                local_match = _RE_SEARCH_LOCAL.findall(trajectory)
                query = local_match[-1].strip() if local_match else question

                # Execute local retrieval
                result_str, score = env.local_backend.search(query)
                info_block = f"<information>{result_str}</information>"

                # If local score is low, append hint
                if score < 0.35:
                    info_block = (
                        f"<information>{result_str}\n"
                        f"[提示：本地知识库相关性较低（{score:.2f}），"
                        f"如需更准确信息可调用网络搜索]</information>"
                    )

                # Time-sensitive keyword triggers web suggestion
                elif any(w in question for w in TIME_SENSITIVE_KEYWORDS):
                    info_block = (
                        f"<information>{result_str}\n"
                        f"[提示：本地知识库可能不含最新信息，"
                        f"如需最新版本/价格/上市等内容可调用网络搜索]</information>"
                    )

                close_tag = "</search_local>"

            elif "<search_web>" in generated and "</search_web>" not in generated:
                # ── Total search count limit ──
                total_search_count += 1
                if total_search_count > MAX_SEARCH_STEPS:
                    info_block = (
                        "<information>已达到最大检索次数限制。</information>\n"
                        "<answer>根据已检索到的信息暂时无法给出完整答案，"
                        "建议访问小米汽车官网获取最新信息。</answer>"
                    )
                    trajectory += "</search_web>\n" + info_block + "\n"
                    messages.append({"role": "assistant", "content": generated + "</search_web>\n" + info_block + "\n"})
                    if verbose:
                        print(f"</search_web>\n{info_block}\n", end="", flush=True)
                    break

                trajectory += "</search_web>"
                if verbose:
                    print("</search_web>", end="", flush=True)
                # Extract search keyword
                web_match = _RE_SEARCH_WEB.findall(trajectory)
                query = web_match[-1].strip() if web_match else f"小米SU7 {question}"

                # Execute web search
                result_str = env.web_backend.search(query)
                web_note = env.web_backend.last_note   # fallback hint (display only)
                info_block = f"<information>{result_str}</information>"

                close_tag = "</search_web>"

            elif "<read_page>" in generated and "</read_page>" not in generated:
                # ── read_page hop limit ──
                read_page_count += 1
                if read_page_count > MAX_READ_PAGE_HOPS:
                    info_block = "<information>已达到最大页面阅读次数限制，请基于已有信息作答。</information>"
                    trajectory += "</read_page>\n" + info_block + "\n"
                    messages.append({"role": "assistant", "content": generated + "</read_page>\n" + info_block + "\n"})
                    if verbose:
                        print(f"</read_page>\n{info_block}\n", end="", flush=True)
                    continue  # don't break, let model try to answer

                # ── Total search count limit ──
                total_search_count += 1
                if total_search_count > MAX_SEARCH_STEPS:
                    info_block = (
                        "<information>已达到最大检索次数限制。</information>\n"
                        "<answer>根据已检索到的信息暂时无法给出完整答案，"
                        "建议访问小米汽车官网获取最新信息。</answer>"
                    )
                    trajectory += "</read_page>\n" + info_block + "\n"
                    messages.append({"role": "assistant", "content": generated + "</read_page>\n" + info_block + "\n"})
                    if verbose:
                        print(f"</read_page>\n{info_block}\n", end="", flush=True)
                    break

                trajectory += "</read_page>"
                if verbose:
                    print("</read_page>", end="", flush=True)
                # Extract URL
                page_match = _RE_READ_PAGE.findall(trajectory)
                url = page_match[-1].strip() if page_match else ""

                # Execute deep page reading
                result_str = env.page_reader.fetch(url) if url else "无效的URL地址"
                info_block = f"<information>{result_str}</information>"

                close_tag = "</read_page>"

            else:
                # Other stop reason (shouldn't normally trigger), defensive handling
                messages.append({"role": "assistant", "content": generated})
                continue

            if info_block:
                # Inject retrieval result
                # web_note (e.g. fallback hint) only goes into display trajectory, not messages
                if web_note:
                    trajectory += "\n" + web_note + "\n" + info_block + "\n"
                else:
                    trajectory += "\n" + info_block + "\n"
                # Append current generation + retrieval result as assistant message
                # (web_note excluded from model context)
                messages.append({
                    "role": "assistant",
                    "content": generated + close_tag + "\n" + info_block + "\n",
                })
                if verbose:
                    print("\n" + info_block + "\n", end="", flush=True)
        else:
            # finish_reason != "stop" (e.g. length), model didn't trigger search this round
            messages.append({"role": "assistant", "content": generated})

            # Re-check: if this round's generation has <answer> but not closed
            if "<answer>" in generated and "</answer>" not in generated:
                # Model likely truncated by max_tokens, give another round to finish
                pass

    # ── Ensure trajectory format completeness ──────────────────
    # If <answer> not closed, add closing tag
    if "<answer>" in trajectory and "</answer>" not in trajectory:
        trajectory += "</answer>"
        if verbose:
            print("</answer>", end="", flush=True)

    if verbose:
        print()  # final newline

    # ── Extract results ────────────────────────────────────────
    answer       = RetrievalEnvironment.extract_answer(trajectory)
    search_calls = RetrievalEnvironment.count_search_calls(trajectory)
    reward       = compute_reward(question, trajectory)

    return {
        "question":      question,
        "trajectory":    trajectory,
        "answer":        answer,
        "reward":        reward["reward"],
        "reward_detail": reward,
        "rounds":        rounds,
        "search_calls":  search_calls,
    }


# ────────────────────────────────────────────────────────────
# Interactive Q&A main loop
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL-enhanced inference — Search-R1 generate-while-retrieving")
    parser.add_argument(
        "--vllm-url", type=str,
        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM service address",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (default from env RL_MODEL_NAME or 'qwen3-rl-su7')",
    )
    parser.add_argument(
        "--show-trajectory", action="store_true",
        help="Show full trajectory (including retrieval process)",
    )
    parser.add_argument(
        "--show-reward", action="store_true",
        help="Show reward score breakdown",
    )
    args = parser.parse_args()

    settings = get_settings()
    model_name = args.model or os.getenv("RL_MODEL_NAME", "qwen3-rl-su7")

    # ── Initialize ──────────────────────────────────────────
    print("=" * 80)
    print("Xiaomi SU7 RL-Enhanced Inference System (Search-R1 Paradigm)")
    print("=" * 80)
    print(f"  vLLM URL:  {args.vllm_url}")
    print(f"  Model:     {model_name}")
    print(f"  Paradigm:  Generate while retrieving (model decides retrieval timing)")
    print(f"  Tools:     <search_local> / <search_web> / <read_page>")
    print("=" * 80)
    print("  Enter a question to start, type 'quit' to exit")
    print("=" * 80)

    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] OpenAI SDK not installed. Install with: pip install openai")
        return

    llm_client = OpenAI(
        api_key="EMPTY",
        base_url=args.vllm_url,
    )

    print("\n[INFO] Loading retrieval environment...")
    env = RetrievalEnvironment()
    print("[INFO] Retrieval environment ready\n")

    # ── Interactive loop ────────────────────────────────────
    while True:
        try:
            question = input("User > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Stream output disabled; final summary only; full process via --show-trajectory
        print("\nAssistant >")
        start_time = time.time()
        result = run_rl_inference(
            question   = question,
            llm_client = llm_client,
            env        = env,
            model_name = model_name,
            verbose=False,
        )
        elapsed = time.time() - start_time

        # ── Output results ──────────────────────────────────
        print(f"  Answer: {result['answer']}")
        print(f"  Time: {elapsed:.1f}s | Rounds: {result['rounds']} | "
              f"Retrievals: local×{result['search_calls']['local']} "
              f"web×{result['search_calls']['web']} "
              f"read_page×{result['search_calls']['read_page']}")

        if args.show_reward:
            detail = result["reward_detail"]
            print(f"  Reward: {result['reward']:.3f} "
                  f"(format:{detail['format_score']:.2f} answer:{detail['answer_score']:.2f} "
                  f"tool:{detail['tool_score']:.2f} source:{detail['source_score']:.2f} "
                  f"domain:{detail['domain_score']:.2f} "
                  f"exploration:{detail.get('exploration_score', 0):.2f})")

        if args.show_trajectory:
            print(f"\n  Full trajectory:")
            print(f"  {'─' * 70}")
            for line in result["trajectory"].split("\n"):
                print(f"  {line}")
            print(f"  {'─' * 70}")

        print()


if __name__ == "__main__":
    main()
