# -*- coding: utf-8 -*-
"""
RL reward function — multi-dimensional trajectory quality scoring.

Provides reward signals for GRPO training, evaluating the quality of
model-generated tool-calling trajectories.

Reward dimensions (total 1.0):
  1. Format completeness    (0.05) — tags present and properly closed
  2. Answer quality         (0.40) — accuracy, informativeness, groundedness
  3. Tool usage quality     (0.15) — keyword precision, call ordering
  4. Source attribution     (0.10) — web sources properly cited
  5. Domain compliance      (0.15) — correct refusal for non-SU7 questions
  6. Exploration depth      (0.15) — local-sufficient stop / effective web+read_page

Supports two modes:
  - Rule mode (default): regex + heuristic scoring, no GPU needed
  - Model mode (optional): LLM-based semantic scoring, more precise but needs API

Usage:
  python app/rl/reward_model.py                                    # single demo
  python app/rl/reward_model.py --input data/rl_data/web_fallback_trajectories_grpo.jsonl
"""

import re
import os
import json
import argparse
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tag regex patterns ────────────────────────────────────────
_RE_ANSWER       = re.compile(r"<answer>(.*?)</answer>",            re.DOTALL)
_RE_SEARCH_LOCAL = re.compile(r"<search_local>(.*?)</search_local>", re.DOTALL)
_RE_SEARCH_WEB   = re.compile(r"<search_web>(.*?)</search_web>",   re.DOTALL)
_RE_READ_PAGE    = re.compile(r"<read_page>(.*?)</read_page>",     re.DOTALL)
_RE_INFORMATION  = re.compile(r"<information>(.*?)</information>",  re.DOTALL)

# ── SU7 domain keywords ──────────────────────────────────────
_SU7_KEYWORDS = {
    "小米SU7", "小米 SU7", "小米汽车", "SU7", "SU7 Max", "SU7 Pro",
    "HyperOS", "小米智驾", "澎湃", "米家", "小爱同学",
    "OTA", "弹射起步", "智能驾驶", "辅助驾驶", "领航辅助",
    "激光雷达", "纯电", "续航", "充电", "超级充电",
    "车机", "中控屏", "HUD", "座椅", "空调", "天窗",
    "自动驾驶", "自动泊车", "高速领航", "城市领航",
    # Vehicle maintenance / consumables (single hit = SU7 domain,
    # prevents maintenance questions from being misclassified as non-domain)
    "玻璃水", "洗涤液", "雨刮", "雨刮器", "雨刷", "防冻液", "机油",
    "刹车油", "冷却液", "胎压", "刹车片", "保养", "保修", "质保",
}

# ── Refusal keywords ─────────────────────────────────────────
_REFUSAL_PATTERNS = [
    "只能回答小米SU7相关问题",
    "无法回答此问题",
    "不在我的服务范围",
    "与小米SU7无关",
    "我只能回答",
    "建议您咨询",
]


# ────────────────────────────────────────────────────────────
# Dimension 1: Format completeness (0.0 ~ 0.05)
# ────────────────────────────────────────────────────────────

def score_format(trajectory: str) -> float:
    """
    Check whether the trajectory contains the complete tag structure.

    A complete trajectory should contain (at least one search method):
      Method 1: <search_local> -> <information> -> <answer>          (local search)
      Method 2: <search_web>   -> <information> -> <answer>          (web search)
      Method 3: <search_local> -> <search_web> -> <information> -> <answer> (mixed)
      Optional: <read_page> -> <information> (vertical search)

    Scoring (max 0.05):
      - At least one search method + information + answer = 0.03
      - Both search methods present -> +0.01
      - All present tags properly closed -> +0.01
      - Tag mismatch -> -0.01 per tag
    """
    has_local = bool(_RE_SEARCH_LOCAL.search(trajectory))
    has_web   = bool(_RE_SEARCH_WEB.search(trajectory))
    has_info  = bool(_RE_INFORMATION.search(trajectory))
    has_answer = bool(_RE_ANSWER.search(trajectory))

    # At least one search method required
    if not (has_local or has_web):
        return 0.0

    score = 0.0
    # Core structure: search + information + answer
    if has_local or has_web: score += 0.01
    if has_info:             score += 0.01
    if has_answer:           score += 0.01
    # Both search methods used -> more complete
    if has_local and has_web: score += 0.01

    # Tag closure check
    closure_ok = True
    for tag in ["search_local", "search_web", "read_page", "information", "answer"]:
        opens  = trajectory.count(f"<{tag}>")
        closes = trajectory.count(f"</{tag}>")
        if opens > 0 and opens != closes:
            closure_ok = False
            score -= 0.01  # tag mismatch penalty
    if closure_ok and (has_local or has_web):
        score += 0.01  # all correctly closed bonus

    return max(0.0, min(0.05, score))


# ────────────────────────────────────────────────────────────
# Dimension 2: Answer quality (0.0 ~ 0.40)
# ────────────────────────────────────────────────────────────

def score_answer_quality(trajectory: str, question: str = "") -> float:
    """
    Evaluate <answer> content quality (max 0.40).

    Core principle: detail rewards (length/numbers/terminology) are linked to
    groundedness. Penalises "detailed-and-fluent-but-unsupported" hallucinations.

    Scoring:
      - Non-empty base: 0.10
      - Length substantial (>30 / >100 chars): +0.04 each, scaled by grounded_ratio
      - Contains specific data/specs: +0.05, scaled by grounded_ratio
      - Contains technical terms: +0.04, scaled by grounded_ratio
      - Natural language fluency: +0.03
      - Topic relevance to question: +0.06
      - Groundedness bonus: +min(0.08, ratio*0.12)
      - Hallucination penalty: has numbers/terms but grounded_ratio==0 -> -0.08
    """
    answer_match = _RE_ANSWER.search(trajectory)
    if not answer_match:
        return 0.0

    answer = answer_match.group(1).strip()
    if not answer:
        return 0.0

    # ── Compute groundedness (whether answer is based on retrieved info) ──
    grounded_ratio = 0.0
    info_matches = _RE_INFORMATION.findall(trajectory)
    info_text = " ".join(m.strip() for m in info_matches if m.strip())
    if info_text:
        ans_phrases = re.findall(r"[一-鿿]{3,}", answer)
        if ans_phrases:
            grounded_count = sum(1 for p in ans_phrases if p in info_text)
            grounded_ratio = grounded_count / len(ans_phrases)

    # ── Specific claim detection ──
    has_number = bool(
        re.search(r"\d+[\.\d]*\s*(km|kW|N·m|V|Ah|mm|英寸|%|万元|秒|公里|马力)", answer)
    )
    tech_terms = [
        "激光雷达", "毫米波", "摄像头", "算力", "传感器",
        "电池", "电机", "逆变器", "减速器", "悬架",
        "制动", "转向", "扭矩", "功率", "续航",
    ]
    has_tech = any(t in answer for t in tech_terms)

    score = 0.10  # non-empty base

    # ── Hallucination penalty: specific claims with zero retrieval support ──
    if (has_number or has_tech) and grounded_ratio == 0.0:
        score -= 0.08

    # ── Detail rewards: scaled by grounded_ratio (need >50% phrase support for full) ──
    g = min(1.0, grounded_ratio * 2.0)
    if len(answer) > 30:
        score += 0.04 * g
    if len(answer) > 100:
        score += 0.04 * g
    if has_number:
        score += 0.05 * g
    if has_tech:
        score += 0.04 * g

    # Natural fluency (stylistic, low weight, not groundedness-linked)
    natural_markers = ["此外", "另外", "需要注意的是", "具体来说", "同时", "因此"]
    if any(m in answer for m in natural_markers):
        score += 0.03

    # Topic relevance: check if question 2-char fragments appear in answer
    if question:
        q_keywords = set(re.findall(r"[一-鿿]{2,}", question))
        if q_keywords:
            hits = sum(1 for kw in q_keywords if kw in answer)
            ratio = hits / len(q_keywords)
            score += min(0.06, ratio * 0.12)

    # Groundedness reinforcement
    score += min(0.08, grounded_ratio * 0.12)

    return max(0.0, min(0.40, score))


# ────────────────────────────────────────────────────────────
# Dimension 3: Tool usage quality (0.0 ~ 0.15)
# ────────────────────────────────────────────────────────────

def score_tool_usage(trajectory: str, question: str = "") -> float:
    """
    Evaluate the quality of tool calls.

    Scoring:
      - search_local appears before search_web: +0.06
      - Search keyword concise (<20 chars) and not redundant: +0.03
      - search_web keyword includes "小米SU7" prefix: +0.03
      - Total call count reasonable (1-3 local + 1 web): +0.03
    """
    score = 0.0

    # Check call ordering: local should be before web
    local_pos = trajectory.find("<search_local>")
    web_pos   = trajectory.find("<search_web>")
    if local_pos >= 0 and web_pos >= 0 and local_pos < web_pos:
        score += 0.06
    elif local_pos >= 0 and web_pos < 0:
        score += 0.05  # local-only is also reasonable

    # Local search keyword quality
    local_match = _RE_SEARCH_LOCAL.search(trajectory)
    if local_match:
        local_query = local_match.group(1).strip()
        if len(local_query) <= 20 and len(local_query) >= 2:
            score += 0.03
        elif len(local_query) > 20:
            score += 0.01  # too long penalty

    # Web search keyword includes "小米SU7" prefix
    web_match = _RE_SEARCH_WEB.search(trajectory)
    if web_match:
        web_query = web_match.group(1).strip()
        if "小米" in web_query or "SU7" in web_query:
            score += 0.03

    # Call count reasonableness
    local_count = len(_RE_SEARCH_LOCAL.findall(trajectory))
    web_count   = len(_RE_SEARCH_WEB.findall(trajectory))
    read_count  = len(_RE_READ_PAGE.findall(trajectory))
    if 1 <= local_count <= 3 and 1 <= web_count <= 2:
        score += 0.03
    elif local_count + web_count + read_count > 6:
        score -= 0.03  # excessive calls penalty

    return max(0.0, min(0.15, score))


# ────────────────────────────────────────────────────────────
# Dimension 4: Source attribution (0.0 ~ 0.10)
# ────────────────────────────────────────────────────────────

def score_source_attribution(trajectory: str) -> float:
    """
    Evaluate whether sources are properly attributed.

    Scoring:
      - When web search is used, answer cites "网络信息": +0.05
      - Page reference format used: +0.03
      - Disclaimer at end: +0.02
    """
    score = 0.0
    answer_match = _RE_ANSWER.search(trajectory)
    if not answer_match:
        return 0.0

    answer = answer_match.group(1).strip()
    has_web = bool(_RE_SEARCH_WEB.search(trajectory))

    # Web source attribution
    if has_web:
        if any(kw in answer for kw in ["网络信息", "来源于网络", "根据网络"]):
            score += 0.05
        elif "网络" in answer:
            score += 0.02

    # Page reference format: match 【1】 or 【1,3,5】
    if re.search(r"【\d+(?:,\d+)*】", answer) or re.search(r"第\d+页", answer):
        score += 0.03

    # Disclaimer
    disclaimer_keywords = ["请以小米官方", "以官方为准", "最新公告为准", "建议访问"]
    if any(kw in answer for kw in disclaimer_keywords):
        score += 0.02

    return max(0.0, min(0.10, score))


# ────────────────────────────────────────────────────────────
# Dimension 5: Domain compliance (0.0 ~ 0.15)
# ────────────────────────────────────────────────────────────

def score_domain_compliance(question: str, trajectory: str) -> float:
    """
    Evaluate domain compliance.

    For SU7-related questions: should answer normally with quality.
    For non-SU7 questions: should correctly refuse.

    Scoring:
      - SU7 question with substantive answer: +0.10
      - SU7 question without false refusal: +0.05
      - Non-SU7 question correctly refused: 0.15
      - Non-SU7 question not refused (hallucination risk): 0.00
    """
    answer_match = _RE_ANSWER.search(trajectory)
    if not answer_match:
        return 0.0

    answer = answer_match.group(1).strip()

    # Determine if question belongs to SU7 domain
    is_su7_related = _is_su7_question(question)

    if is_su7_related:
        score = 0.05  # base
        # No false refusal
        if not any(p in answer for p in _REFUSAL_PATTERNS):
            score += 0.05
        # Substantive answer (non-empty and reasonable length)
        if len(answer) > 20:
            score += 0.05
    else:
        # Non-SU7 question
        if any(p in answer for p in _REFUSAL_PATTERNS):
            score = 0.15  # correctly refused
        else:
            score = 0.00  # not refused, hallucination risk

    return max(0.0, min(0.15, score))


def _is_su7_question(question: str) -> bool:
    """Determine whether a question belongs to the SU7 domain."""
    # Direct SU7 keyword match
    for kw in _SU7_KEYWORDS:
        if kw in question:
            return True

    # Generic car keywords (likely SU7-related with >=2 hits)
    car_keywords = {
        "车辆", "驾驶", "充电", "续航", "电池", "轮胎", "刹车",
        "方向盘", "安全带", "空调", "座椅", "车门", "车窗",
        "后备箱", "仪表盘", "导航", "泊车", "灯光", "雨刷",
        "油门", "换挡", "启动", "熄火", "保养", "保险",
    }
    car_hits = sum(1 for kw in car_keywords if kw in question)
    if car_hits >= 2:
        return True

    return False


# ────────────────────────────────────────────────────────────
# Dimension 6: Exploration depth (0.0 ~ 0.15)
# ────────────────────────────────────────────────────────────

def _split_info_by_source(trajectory: str):
    """Classify <information> blocks by preceding search tag, return (local_text, web_text)."""
    local_parts, web_parts = [], []
    tokens = re.split(r"(<search_local>|<search_web>|<read_page>|<information>|</information>)", trajectory)
    source, in_info, buf = "local", False, []
    for tok in tokens:
        if tok == "<search_local>":
            source = "local"
        elif tok in ("<search_web>", "<read_page>"):
            source = "web"
        elif tok == "<information>":
            in_info, buf = True, []
        elif tok == "</information>":
            in_info = False
            text = "".join(buf).strip()
            if text:
                (local_parts if source == "local" else web_parts).append(text)
            buf = []
        elif in_info:
            buf.append(tok)
    return " ".join(local_parts), " ".join(web_parts)


def _grounded_ratio(answer: str, info_text: str) -> float:
    """Fraction of 3+ char Chinese phrases in answer that appear in info_text."""
    if not answer or not info_text:
        return 0.0
    phrases = re.findall(r"[一-鿿]{3,}", answer)
    if not phrases:
        return 0.0
    return sum(1 for p in phrases if p in info_text) / len(phrases)


def score_exploration_depth(trajectory: str) -> float:
    """
    Evaluate exploration depth (max 0.15).

    Principle: reward "effective exploration", not "exploration for its own sake".

    Distinguishes "should use web" from "shouldn't" by checking whether
    <answer> references web information (web-grounded):

      【No web (pure local)】
        - Answer grounded in local + substantial: 0.08 (ideal local-only)
        - Answer not supported by local (should have used web but didn't): 0.04
        - Local has no useful results / no answer: 0.02
      【With web】
        - Answer references web info (web was effective): 0.10 + read_page bonus (max 0.15)
        - Searched web but answer didn't use web info: 0.05 (worse than pure local 0.08)
        - Web returned nothing and answer unsupported: 0.03

    Net effect: local-only is best when local is sufficient (0.08 > wasteful web 0.05);
    using web is best when local is insufficient (0.10+ > forcing local 0.08).
    """
    has_local = bool(_RE_SEARCH_LOCAL.search(trajectory))
    has_web = bool(_RE_SEARCH_WEB.search(trajectory))
    read_matches = _RE_READ_PAGE.findall(trajectory)
    answer_match = _RE_ANSWER.search(trajectory)
    answer = answer_match.group(1).strip() if answer_match else ""

    local_info, web_info = _split_info_by_source(trajectory)
    ans_local_g = _grounded_ratio(answer, local_info)
    ans_web_g = _grounded_ratio(answer, web_info)

    # ── No web: pure local trajectory ───────────────────────
    if not has_web:
        if not has_local:
            return 0.0
        if answer and len(answer) > 20 and ans_local_g > 0:
            return 0.08   # local sufficient, answer grounded: ideal local-only
        if answer:
            return 0.04   # has answer but local unsupported (should have used web)
        return 0.02       # local produced nothing useful

    # ── With web ────────────────────────────────────────────
    # Web info referenced in answer -> web was effective, higher than pure local
    if ans_web_g > 0:
        score = 0.10
        valid_urls = sum(1 for u in read_matches if u.strip().startswith(("http://", "https://")))
        if valid_urls > 0:
            score += 0.03                      # read_page with valid URL
        if 1 <= len(read_matches) <= 2 and ans_web_g > 0.2:
            score += 0.02                      # reasonable depth + high web reference
        return max(0.0, min(0.15, score))

    # Searched web but answer didn't use web info: wasted web search
    if answer and ans_local_g > 0:
        return 0.05   # local was sufficient but searched web anyway, worse than pure local
    return 0.03       # web returned nothing, answer unsupported


# ────────────────────────────────────────────────────────────
# Composite reward function
# ────────────────────────────────────────────────────────────

def compute_reward(
    question:   str,
    trajectory: str,
    verbose:    bool = False,
) -> dict:
    """
    Compute the composite reward score.

    Args:
        question:   The user question
        trajectory: The full model-generated trajectory
        verbose:    Whether to print per-dimension scores

    Returns:
        {
            "reward":            float,  # total [0, 1]
            "format_score":      float,  # format completeness (0.05)
            "answer_score":      float,  # answer quality (0.40)
            "tool_score":        float,  # tool usage (0.15)
            "source_score":      float,  # source attribution (0.10)
            "domain_score":      float,  # domain compliance (0.15)
            "exploration_score": float,  # exploration depth (0.15)
        }
    """
    fmt        = score_format(trajectory)
    ans        = score_answer_quality(trajectory, question)
    tool       = score_tool_usage(trajectory, question)
    src        = score_source_attribution(trajectory)
    domain     = score_domain_compliance(question, trajectory)
    exploration = score_exploration_depth(trajectory)

    total = fmt + ans + tool + src + domain + exploration

    result = {
        "reward":            round(total,       4),
        "format_score":      round(fmt,         4),
        "answer_score":      round(ans,         4),
        "tool_score":        round(tool,        4),
        "source_score":      round(src,         4),
        "domain_score":      round(domain,      4),
        "exploration_score": round(exploration,  4),
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Question: {question[:50]}...")
        print(f"{'='*50}")
        print(f"  Format:       {fmt:.3f} / 0.05")
        print(f"  Answer:       {ans:.3f} / 0.40")
        print(f"  Tool:         {tool:.3f} / 0.15")
        print(f"  Source:       {src:.3f} / 0.10")
        print(f"  Domain:       {domain:.3f} / 0.15")
        print(f"  Exploration:  {exploration:.3f} / 0.15")
        print(f"  {'─'*40}")
        print(f"  Total:        {total:.3f} / 1.00")
        print(f"{'='*50}")

    return result


# ────────────────────────────────────────────────────────────
# Custom reward function entry (TRL GRPOTrainer compatible)
# ────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """
    Extract plain text from TRL's completion/prompt.

    TRL GRPOTrainer may pass:
      - str:                         "text content"
      - list[dict] (messages):       [{"role": "assistant", "content": "text"}]
      - list[str]:                   ["text content"]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # messages format [{"role": "assistant", "content": "..."}]
        for msg in content:
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
        # pure string list
        return " ".join(str(c) for c in content)
    return str(content)


def reward_fn(completions: list, **kwargs) -> list[float]:
    """
    GRPO custom reward function, compatible with TRL GRPOTrainer interface.

    Usage:
      # TRL GRPOTrainer
      trainer = GRPOTrainer(reward_funcs=reward_fn, ...)

    Args:
        completions: Model-generated trajectories (str / list[dict] / list[str])
        **kwargs:    TRL-provided prompts

    Returns:
        List of reward scores [0.0, 1.0] for each trajectory
    """
    prompts = kwargs.get("prompts", [""] * len(completions))
    rewards = []

    for prompt_text, completion in zip(prompts, completions):
        question   = _extract_question_from_prompt(_extract_text(prompt_text))
        trajectory = _extract_text(completion)
        result = compute_reward(question, trajectory)
        rewards.append(result["reward"])

    return rewards


def _extract_question_from_prompt(prompt_text: str) -> str:
    """Extract user question from GRPO prompt messages."""
    if isinstance(prompt_text, list):
        # messages format
        for msg in prompt_text:
            if msg.get("role") == "user":
                return msg.get("content", "")
    elif isinstance(prompt_text, str):
        return prompt_text
    return ""


# ────────────────────────────────────────────────────────────
# Batch evaluation CLI
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL reward function evaluation tool")
    parser.add_argument(
        "--input", type=str, default=None,
        help="GRPO JSONL file path for batch evaluation",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo examples",
    )
    args = parser.parse_args()

    if args.demo or not args.input:
        # ── Demo mode ──────────────────────────────────────
        print("=" * 60)
        print("Reward Function Demo")
        print("=" * 60)

        demos = [
            {
                "question": "小米SU7的续航里程是多少？",
                "trajectory": (
                    "<search_local>小米SU7 续航里程</search_local>\n"
                    "<information>[1] 小米SU7标准版CLTC续航里程为700km，"
                    "SU7 Pro版续航里程为830km，SU7 Max版续航里程为800km。</information>\n"
                    "<search_web>小米SU7 续航里程 最新数据</search_web>\n"
                    "<information>根据最新网络信息，小米SU7全系续航范围在700-830km之间。</information>\n"
                    "<answer>小米SU7根据不同版本的续航里程如下：\n"
                    "标准版：CLTC续航700km\n"
                    "Pro版：CLTC续航830km\n"
                    "Max版：CLTC续航800km\n"
                    "续航表现处于同级别纯电轿车前列水平。"
                    "（以上信息来源于网络，请以小米官方最新公告为准）</answer>"
                ),
            },
            {
                "question": "小米SU7最新OTA更新了什么功能？",
                "trajectory": (
                    "<search_local>小米SU7 OTA更新 功能</search_local>\n"
                    "<information>[提示：本地知识库相关性较低（0.18），"
                    "如需更准确信息可调用网络搜索]</information>\n"
                    "<search_web>小米SU7 最新OTA版本 2025 更新内容</search_web>\n"
                    "<information>"
                    "【小米SU7 OTA v2.4.0 发布公告】新增城市领航辅助、HUD自定义显示等12项更新\n"
                    "网址：https://www.xiaomi.com/ev/su7/ota\n"
                    "【车主社区】OTA v2.4.0 详细体验报告\n"
                    "网址：https://www.autohome.com.cn/news/202501/su7-ota</information>\n"
                    "<read_page>https://www.xiaomi.com/ev/su7/ota</read_page>\n"
                    "<information>[页面来源：www.xiaomi.com]\n"
                    "小米SU7 OTA v2.4.0 正式发布，本次更新包含12项功能升级：\n"
                    "1. 城市领航辅助（City NOA）正式上线\n"
                    "2. HUD抬头显示新增自定义模式\n"
                    "3. 语音助手升级，支持多轮对话\n"
                    "4. 座椅记忆功能优化...</information>\n"
                    "<answer>根据小米官方页面信息，小米SU7最新的OTA v2.4.0更新了以下主要功能：\n"
                    "1. 城市领航辅助（City NOA）正式上线\n"
                    "2. HUD抬头显示新增自定义模式\n"
                    "3. 语音助手升级，支持多轮对话\n"
                    "4. 座椅记忆功能优化\n"
                    "本次更新共包含12项功能升级。"
                    "（以上信息来源于www.xiaomi.com，请以小米官方最新公告为准）</answer>"
                ),
            },
            {
                "question": "今天天气怎么样？",
                "trajectory": (
                    "<search_local>天气</search_local>\n"
                    "<information>本地知识库中未检索到相关内容。</information>\n"
                    "<answer>很抱歉，我只能回答小米SU7相关问题。</answer>"
                ),
            },
        ]

        for d in demos:
            compute_reward(d["question"], d["trajectory"], verbose=True)
        return

    # ── Batch evaluation mode ────────────────────────────────
    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        return

    results = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question   = ""
            completion = item.get("completion", "")

            # Extract question from prompt
            prompt = item.get("prompt", "")
            if isinstance(prompt, list):
                for msg in prompt:
                    if msg.get("role") == "user":
                        question = msg.get("content", "")
            elif isinstance(prompt, str):
                question = prompt

            reward = compute_reward(question, completion)
            results.append(reward)

    # Statistics
    if not results:
        print("No data.")
        return

    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_fmt    = sum(r["format_score"] for r in results) / len(results)
    avg_ans    = sum(r["answer_score"] for r in results) / len(results)
    avg_tool   = sum(r["tool_score"] for r in results) / len(results)
    avg_src    = sum(r["source_score"] for r in results) / len(results)
    avg_domain = sum(r["domain_score"] for r in results) / len(results)
    avg_explore = sum(r.get("exploration_score", 0) for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"Batch Evaluation Report ({len(results)} items)")
    print("=" * 60)
    print(f"  Avg total reward:     {avg_reward:.4f} / 1.00")
    print(f"  Format completeness:  {avg_fmt:.4f} / 0.05")
    print(f"  Answer quality:       {avg_ans:.4f} / 0.40")
    print(f"  Tool usage:           {avg_tool:.4f} / 0.15")
    print(f"  Source attribution:   {avg_src:.4f} / 0.10")
    print(f"  Domain compliance:    {avg_domain:.4f} / 0.15")
    print(f"  Exploration depth:    {avg_explore:.4f} / 0.15")
    print("=" * 60)

    # Distribution
    brackets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for r in results:
        score = r["reward"]
        if score < 0.2:
            brackets["0.0-0.2"] += 1
        elif score < 0.4:
            brackets["0.2-0.4"] += 1
        elif score < 0.6:
            brackets["0.4-0.6"] += 1
        elif score < 0.8:
            brackets["0.6-0.8"] += 1
        else:
            brackets["0.8-1.0"] += 1

    print("\nReward distribution:")
    for bracket, count in brackets.items():
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bracket}: {count:>4d} ({pct:>5.1f}%) {bar}")


if __name__ == "__main__":
    main()
