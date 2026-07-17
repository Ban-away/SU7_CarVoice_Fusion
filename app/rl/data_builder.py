# -*- coding: utf-8 -*-
"""
RL training data builder — web fallback trajectory generator.

Functionality:
  1. Read questions from web_fallback_questions.json
  2. Each question: local retrieval first, determine if info is sufficient
  3. Trigger web search fallback when local info is insufficient
  4. Call LLM to generate complete tool-call trajectories
  5. Output GRPO training format JSON files

Output format: LLaMA-Factory SFT format (also serves as GRPO warm-up data)

Usage:
  python app/rl/data_builder.py
  python app/rl/data_builder.py --resume   # resume from checkpoint
  python app/rl/data_builder.py --dry-run  # process first 5 items only
"""

import os

# ── Disable tqdm before any tqdm-related imports ──
os.environ["TQDM_DISABLE"] = "1"

import re
import json
import time
import hashlib
import argparse
import threading
import logging
import concurrent.futures
from typing import Optional

from app.shared.config import get_settings
from app.knowledge.service import KnowledgeService
from app.rl.web_reader import WebPageReader
from app.rl.format_converter import SYSTEM_PROMPT, to_sft_target, to_sft_format, to_grpo_format

logger = logging.getLogger(__name__)

# ── Path configuration ──────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUESTIONS_PATH = os.path.join(BASE_DIR, "data/rl_data/web_fallback_questions.json")
OUTPUT_PATH    = os.path.join(BASE_DIR, "data/rl_data/web_fallback_trajectories.json")
CKPT_PATH      = os.path.join(BASE_DIR, "data/rl_data/web_fallback_ckpt.jsonl")

# ── Hyperparameters ─────────────────────────────────────────
LOCAL_TOPK          = 3      # local retrieval result count
RELEVANCE_THRESHOLD = 0.35   # below this score: "local info insufficient"
MAX_WORKERS         = 8      # concurrent threads
RETRY_TIMES         = 1      # per-item retry count


# ────────────────────────────────────────────────────────────
# Local search tool (wraps KnowledgeService)
# ────────────────────────────────────────────────────────────

class LocalSearchTool:
    """Encapsulates the local retrieval stack via KnowledgeService."""

    def __init__(self):
        logger.info("Loading local retrieval components...")
        settings = get_settings()
        self._ks = KnowledgeService(
            web_search_enabled=False,
            retriever_backend=settings.retriever_backend,
            reranker_backend=settings.reranker_backend,
            top_k=LOCAL_TOPK,
        )
        self._lock = threading.Lock()
        logger.info("Local retrieval components loaded")

    def search(self, query: str, topk: int = LOCAL_TOPK) -> tuple[list, float]:
        """
        Return (docs, max_score).
        max_score used to determine relevance: < RELEVANCE_THRESHOLD = insufficient.
        """
        with self._lock:
            docs = self._ks.search_local_docs(query, top_k=topk)

        if not docs:
            return [], 0.0

        # Estimate max relevance score
        import math
        max_score = 0.0
        if docs:
            raw = getattr(docs[0], "score", None)
            if raw is not None:
                try:
                    max_score = 1 / (1 + math.exp(-float(raw)))
                except (ValueError, OverflowError):
                    max_score = 0.5
            else:
                max_score = min(0.3 + len(docs) * 0.05, 0.6)
        return docs, max_score

    def format_result(self, docs: list) -> str:
        if not docs:
            return "本地知识库中未检索到相关内容。"
        parts = []
        for i, doc in enumerate(docs, 1):
            page = getattr(doc, "page", "")
            page_str = f"（第{page}页）" if page else ""
            content = getattr(doc, "content", "")
            parts.append(f"[{i}] {content[:300]}{page_str}")
        return "\n".join(parts)


# ────────────────────────────────────────────────────────────
# Web search module (multi-backend support)
# ────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Web search tool with multiple backends:
    - bing: Bing Search API (stable, Chinese-friendly)
    - serpapi: SerpAPI (needs SERPAPI_KEY env var)
    - serper: Serper (needs SERPER_API_KEY env var)
    - doubao: Simulated via Doubao LLM (no extra API needed, but not real crawl)
    """

    def __init__(self, backend: str = "auto"):
        self._backends = self._build_chain(backend)
        self._stats = {}          # backend -> success count
        self._fallback_used = 0   # succeeded only via fallback
        self._all_failed = 0      # all backends failed
        self._lock = threading.Lock()
        logger.info("Web search backend chain: %s", " → ".join(self._backends))

    def _build_chain(self, backend: str) -> list:
        """Build multi-backend fallback chain, consistent with environment.WebSearchBackend."""
        if backend != "auto":
            return [backend]
        chain = []
        if os.getenv("SERPAPI_KEY"):
            chain.append("serpapi")
        if os.getenv("SERPER_API_KEY"):
            chain.append("serper")
        if os.getenv("BING_SEARCH_KEY"):
            chain.append("bing")
        if not chain:
            chain.append("doubao")
        return chain

    def search(self, query: str) -> str:
        """Chained fallback: retry each backend RETRY_TIMES times, cascade on exhaustion."""
        failed = []
        for be in self._backends:
            for attempt in range(RETRY_TIMES):
                try:
                    if be == "bing":
                        result = self._search_bing(query)
                    elif be == "serpapi":
                        result = self._search_serpapi(query)
                    elif be == "serper":
                        result = self._search_serper(query)
                    else:
                        result = self._search_via_doubao(query)
                    with self._lock:
                        self._stats[be] = self._stats.get(be, 0) + 1
                        if failed:
                            self._fallback_used += 1
                    return result
                except Exception:
                    if attempt < RETRY_TIMES - 1:
                        time.sleep(2 ** attempt)
            failed.append(be)
        with self._lock:
            self._all_failed += 1
        return ""

    def summary(self) -> str:
        """Web search summary for end-of-run reporting."""
        parts = [f"{be}×{n}" for be, n in self._stats.items()]
        line = f"success {' '.join(parts) if parts else '0'}"
        if self._fallback_used:
            line += f" | fallback {self._fallback_used}"
        if self._all_failed:
            line += f" | all-failed {self._all_failed}"
        return line

    def _search_bing(self, query: str) -> str:
        import requests
        url     = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_SEARCH_KEY"]}
        params  = {"q": query, "mkt": "zh-CN", "count": 5, "responseFilter": "Webpages"}
        resp    = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("webPages", {}).get("value", [])
        if not results:
            return ""
        parts = []
        for r in results[:5]:
            parts.append(f"【{r['name']}】\n{r['snippet']}\n网址：{r['url']}")
        return "\n\n".join(parts)

    def _search_serpapi(self, query: str) -> str:
        import requests
        url    = "https://serpapi.com/search"
        params = {
            "q":       query,
            "hl":      "zh-cn",
            "gl":      "cn",
            "api_key": os.environ["SERPAPI_KEY"],
            "num":     5,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("organic_results", [])
        if not results:
            return ""
        parts = []
        for r in results[:5]:
            snippet = r.get("snippet", "")
            title   = r.get("title", "")
            link    = r.get("link", "")
            parts.append(f"【{title}】\n{snippet}\n网址：{link}")
        return "\n\n".join(parts)

    def _search_serper(self, query: str) -> str:
        """Serper (google.serper.dev) — fallback when SerpAPI quota exhausted."""
        import requests
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": os.environ["SERPER_API_KEY"],
                     "Content-Type": "application/json"},
            json={"q": query, "hl": "zh-cn", "gl": "cn", "num": 5},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("organic", [])
        if not results:
            return ""
        parts = []
        for r in results[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            parts.append(f"【{title}】\n{snippet}\n网址：{link}")
        return "\n\n".join(parts)

    def _search_via_doubao(self, query: str) -> str:
        """
        Use Doubao LLM knowledge base to simulate web search.
        Suitable when no search API key is available; results based on LLM training data.
        Real-time accuracy is limited; manual verification recommended for critical content.
        """
        try:
            from openai import OpenAI
        except ImportError:
            return ""
        client = OpenAI(
            api_key=os.environ["DOUBAO_API_KEY"],
            base_url=os.environ["DOUBAO_BASE_URL"],
        )
        prompt = (
            f"请以网络搜索结果的形式，提供关于以下问题的最新准确信息。"
            f"直接给出信息内容，不要有任何引导语，信息要具体、有数据支撑。\n\n"
            f"搜索问题：{query}\n\n"
            f"请提供3-5条有实质内容的搜索结果摘要："
        )
        completion = client.chat.completions.create(
            model=os.environ.get("DOUBAO_MODEL_NAME", "ep-20240601170316-5dhwt"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()


# ────────────────────────────────────────────────────────────
# Trajectory builder
# ────────────────────────────────────────────────────────────

class TrajectoryBuilder:
    """Assemble retrieval results into complete tool-call trajectories."""

    def __init__(self):
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(
                api_key=os.environ["DOUBAO_API_KEY"],
                base_url=os.environ["DOUBAO_BASE_URL"],
            )
        except ImportError:
            logger.warning("OpenAI SDK not available; trajectory builder will fail on answer synthesis")
            self.llm_client = None
        self.page_reader = WebPageReader()

    def build(
        self,
        question:     str,
        local_query:  str,
        local_docs:   list,
        web_query:    str,
        web_result:   str,
    ) -> str:
        """
        Generate complete assistant trajectory text.

        Key design: <search_local>/<information>/<search_web>/<information> are
        deterministically assembled by code (local info + web info always present),
        not left to LLM — prevents LLM from lazily omitting local <information>.
        LLM only synthesizes <answer> with strong constraint to include web info.
        """
        local_result_str = self._format_local_docs(local_docs)
        if not web_result.strip():
            web_result = "网络搜索暂时未获取到有效结果。"

        # Optional: read_page deep reading
        page_url, page_content = self._try_fetch_best_page(web_result)

        # LLM synthesizes answer based on web info
        answer = self._synthesize_answer(question, web_result, page_content)

        # Deterministic assembly: local info + web info always present, order fixed
        parts = [
            f"<search_local>{local_query}</search_local>",
            f"<information>{local_result_str}</information>",
            f"<search_web>{web_query}</search_web>",
            f"<information>{web_result}</information>",
        ]
        if page_url and page_content:
            parts.append(f"<read_page>{page_url}</read_page>")
            parts.append(f"<information>{page_content}</information>")
        parts.append(f"<answer>{answer}</answer>")
        return "\n".join(parts)

    def _synthesize_answer(self, question: str, web_result: str, page_content: str = "") -> str:
        """Synthesize <answer> from web info; must include concrete data from sources."""
        if self.llm_client is None:
            return "根据目前可获取的信息，暂时无法回答此问题，建议访问小米汽车官网获取最新信息。"

        context = web_result
        if page_content:
            context += "\n\n" + page_content
        prompt = (
            "根据以下网络检索信息回答用户问题。\n\n"
            f"问题：{question}\n\n"
            f"网络信息：\n{context}\n\n"
            "要求：\n"
            "1. 必须基于上述网络信息回答，提取关键数据/事实——版本号、数量、时间等具体内容必须给出。\n"
            "2. 信息中包含答案就直接回答，语言自然流畅，不要重复问题。\n"
            "3. 只有当信息完全不相关时，才输出：根据目前可获取的信息，暂时无法回答此问题，建议访问小米汽车官网。\n"
            "4. 结尾注明：（以上信息来源于网络，请以小米官方最新公告为准）\n\n"
            "直接输出答案正文（不要 <answer> 标签、不要任何前缀）："
        )
        completion = self.llm_client.chat.completions.create(
            model=os.environ.get("DOUBAO_MODEL_NAME", "ep-20240601170316-5dhwt"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()

    def _format_local_docs(self, docs: list) -> str:
        if not docs:
            return "本地知识库中未检索到相关内容。"
        parts = []
        for i, doc in enumerate(docs, 1):
            page = getattr(doc, "page", "")
            suffix = f"（第{page}页）" if page else ""
            content = getattr(doc, "content", "")
            parts.append(f"[{i}]{suffix} {content[:250]}")
        return "\n".join(parts)

    def _try_fetch_best_page(self, web_result: str) -> tuple[str, str]:
        """
        Extract the most valuable URL from web search results and fetch page content.
        Returns (url, content), or ("", "") on failure.
        """
        urls = re.findall(r"(?:网址|来源)[：:]\s*(https?://[^\s\n]+)", web_result)
        if not urls:
            return "", ""

        best_url = urls[0].strip()
        try:
            content = self.page_reader.fetch(best_url, max_chars=1500)
            if content.startswith("无法") or content.startswith("页面") or content.startswith("不支持"):
                return "", ""
            return best_url, content
        except Exception:
            return "", ""


# ────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────

def process_one(
    item:            dict,
    local_tool:      LocalSearchTool,
    web_tool:        WebSearchTool,
    traj_builder:    TrajectoryBuilder,
) -> Optional[dict]:
    """Process a single question, generating a complete trajectory."""

    question      = item["question"]
    local_query   = item.get("local_query_hint", question)
    web_query_raw = item.get("web_query_hint", f"小米SU7 {question}")

    # ── Step 1: Local retrieval ────────────────────────────
    local_docs, max_score = local_tool.search(local_query)

    # ── Step 2: Determine if web fallback needed ────────────
    # web_fallback_questions.json items are pre-selected as needing web fallback,
    # so unconditionally trigger web search; still retain local results in trajectory.
    need_web = True

    if not need_web:
        logger.info("Local sufficient (score=%.2f): %s...", max_score, question[:30])
        return None

    # ── Step 3: Web search ─────────────────────────────────
    web_result = web_tool.search(web_query_raw)

    # ── Step 4: Generate trajectory ────────────────────────
    trajectory = traj_builder.build(
        question=question,
        local_query=local_query,
        local_docs=local_docs,
        web_query=web_query_raw,
        web_result=web_result,
    )

    # ── Step 5: Assemble output ────────────────────────────
    unique_id = hashlib.md5(question.encode("utf-8")).hexdigest()
    return {
        "id":          item["id"],
        "unique_id":   unique_id,
        "category":    item.get("category", ""),
        "category_zh": item.get("category_zh", ""),
        "question":    question,
        "trajectory":  trajectory,
        "local_docs_count":   len(local_docs),
        "local_max_score":    round(max_score, 4),
        "web_search_used":    True,
        "sft_format":  to_sft_format(question, trajectory),
        "grpo_format": to_grpo_format(question, trajectory, item.get("category", "")),
    }


def load_checkpoint() -> set:
    """Load completed IDs for resume support."""
    done = set()
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["id"])
                except Exception:
                    pass
    return done


def save_checkpoint(result: dict):
    """Write checkpoint entry in real time."""
    with open(CKPT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": result["id"]}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Web fallback trajectory builder")
    parser.add_argument("--resume",  action="store_true", help="Resume from checkpoint, skip processed items")
    parser.add_argument("--dry-run", action="store_true", help="Process first 5 items only")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "bing", "serpapi", "doubao"],
                        help="Web search backend")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Concurrent thread count")
    args = parser.parse_args()

    # ── Load questions ────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    if args.dry_run:
        questions = questions[:5]
        print(f"[DRY-RUN] Processing first {len(questions)} items only")

    # Resume support
    done_ids = load_checkpoint() if args.resume else set()
    questions = [q for q in questions if q["id"] not in done_ids]
    print(f"[INFO] Pending: {len(questions)} items (skipped {len(done_ids)})")

    # ── Initialize tools ──────────────────────────────────
    local_tool   = LocalSearchTool()
    web_tool     = WebSearchTool(backend=args.backend)
    traj_builder = TrajectoryBuilder()

    # ── Concurrent processing ─────────────────────────────
    results     = []
    file_lock   = threading.Lock()

    def _process(item):
        for attempt in range(RETRY_TIMES):
            try:
                result = process_one(item, local_tool, web_tool, traj_builder)
                if result:
                    with file_lock:
                        save_checkpoint(result)
                return result
            except Exception as e:
                if attempt < RETRY_TIMES - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Processing failed (%s): %s", item["id"], e)
                    return None

    total = len(questions)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process, item): item for item in questions}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
            done_count += 1
            pct = done_count / total * 100
            bar_len = 40
            filled = int(bar_len * done_count / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\rGenerating trajectories: {pct:5.1f}%|{bar}| {done_count}/{total}", end="", flush=True)
    print()  # newline after progress bar

    # ── Save outputs ──────────────────────────────────────
    # 1. Full data (with debug info)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 2. SFT format (for LLaMA-Factory warm-up training)
    sft_path = OUTPUT_PATH.replace(".json", "_sft.json")
    sft_data = [r["sft_format"] for r in results]
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    # 3. GRPO format (for RL training)
    grpo_path = OUTPUT_PATH.replace(".json", "_grpo.jsonl")
    with open(grpo_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r["grpo_format"], ensure_ascii=False) + "\n")

    # ── Summary report ────────────────────────────────────
    category_counts = {}
    for r in results:
        cat = r.get("category_zh", "其他")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total trajectories: {len(results)}")
    print(f"Web search: {web_tool.summary()}")
    print(f"\nCategory distribution:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")
    print(f"\nOutput files:")
    print(f"  Full data: {OUTPUT_PATH}")
    print(f"  SFT format: {sft_path}")
    print(f"  GRPO format: {grpo_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
