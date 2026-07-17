# -*- coding: utf-8 -*-
"""
RL inference environment — tool-call router.

Responsibilities:
  1. Intercept model-generated tags during generation
  2. Pause generation on tag triggers, call corresponding search backend
  3. Inject search results as <information> blocks back into context
  4. Continue generation until <answer> appears or max steps reached
  5. Extract final answer for reward function

Usage:
  from app.rl.environment import RetrievalEnvironment
  env = RetrievalEnvironment()
  answer, trajectory = env.run(question)  # (run method not implemented here;
                                          #  see infer_rl.py for the loop)
"""

import os
import re
import math
import threading
import logging
from typing import Optional

from app.shared.config import get_settings
from app.knowledge.service import KnowledgeService
from app.knowledge.models import RetrievedDoc
from app.rl.web_reader import WebPageReader

logger = logging.getLogger(__name__)

# ── Relevance threshold ──────────────────────────────────────
RELEVANCE_THRESHOLD = 0.35
LOCAL_TOPK          = 3
MAX_SEARCH_STEPS    = 4   # max tool calls per single inference
MAX_READ_PAGE_HOPS  = 2   # max deep page reads (vertical search)

# ── Tool tag regex patterns ──────────────────────────────────
_RE_SEARCH_LOCAL = re.compile(r"<search_local>(.*?)</search_local>", re.DOTALL)
_RE_SEARCH_WEB   = re.compile(r"<search_web>(.*?)</search_web>",   re.DOTALL)
_RE_READ_PAGE    = re.compile(r"<read_page>(.*?)</read_page>",     re.DOTALL)
_RE_ANSWER       = re.compile(r"<answer>(.*?)</answer>",            re.DOTALL)
_RE_INFORMATION  = re.compile(r"<information>(.*?)</information>",  re.DOTALL)


# ────────────────────────────────────────────────────────────
# Local search backend (wraps KnowledgeService)
# ────────────────────────────────────────────────────────────

class LocalSearchBackend:
    """Thread-safe local search backend using KnowledgeService."""

    _instance = None
    _lock      = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        settings = get_settings()
        self._ks = KnowledgeService(
            web_search_enabled=False,
            retriever_backend=settings.retriever_backend,
            reranker_backend=settings.reranker_backend,
            top_k=LOCAL_TOPK,
        )
        self._search_lock = threading.Lock()

    def search(self, query: str) -> tuple[str, float]:
        """
        Return (formatted_result, relevance_score).

        score < RELEVANCE_THRESHOLD means local info is insufficient,
        web fallback should be considered.
        """
        with self._search_lock:
            docs = self._ks.search_local_docs(query, top_k=LOCAL_TOPK)

        if not docs:
            return "本地知识库中未检索到相关内容。", 0.0

        # Format results
        parts = []
        for i, doc in enumerate(docs, 1):
            page = getattr(doc, "page", "")
            suffix = f"【第{page}页】" if page else ""
            content = getattr(doc, "content", "")
            parts.append(f"[{i}]{suffix} {content[:400]}")
        result_str = "\n".join(parts)

        # Relevance score: use top doc score if available, else conservative estimate
        top_score = 0.0
        if docs:
            raw = getattr(docs[0], "score", None)
            if raw is not None:
                try:
                    top_score = 1 / (1 + math.exp(-float(raw)))
                except (ValueError, OverflowError):
                    top_score = 0.5
        # Conservative floor: if we got content, don't mark as low relevance
        content_floor = min(0.3 + len(docs) * 0.05, 0.6)
        if top_score < content_floor:
            top_score = content_floor
        return result_str, top_score


# ────────────────────────────────────────────────────────────
# Web search backend (multi-backend with fallback chain)
# ────────────────────────────────────────────────────────────

class WebSearchBackend:
    """Web search backend with automatic API detection."""

    # Xiaomi terms for query scoping
    _XIAOMI_TERMS = ("小米汽车", "小米SU7", "小米 SU7", "小米YU7", "小米 YU7",
                     "SU7", "YU7", "SU7 Ultra", "Xiaomi", "澎湃")

    def __init__(self):
        # Build fallback chain by priority and configured keys
        chain = []
        if os.getenv("SERPAPI_KEY"):
            chain.append("serpapi")
        if os.getenv("SERPER_API_KEY"):
            chain.append("serper")
        if os.getenv("BING_SEARCH_KEY"):
            chain.append("bing")
        if not chain:
            chain.append("doubao")  # use Doubao LLM as simulated search when no API key
        self._backends = chain
        self.backend = chain[0]  # primary backend name (for backward compat)
        self.last_note = ""      # most recent fallback hint (for display, not model context)

    def search(self, query: str) -> str:
        """Search the web, scoped to Xiaomi auto domain."""
        query = self._scope_to_xiaomi(query)
        self.last_note = ""
        last_err = None
        failed = []
        result = None
        for be in self._backends:
            try:
                if be == "serpapi":
                    result = self._serpapi(query)
                elif be == "serper":
                    result = self._serper(query)
                elif be == "bing":
                    result = self._bing(query)
                else:
                    result = self._doubao(query)
                break
            except Exception as e:
                last_err = e
                failed.append(be)
                continue
        if result is None:
            self.last_note = f"[检索提示：全部后端失败：{last_err}]"
            return f"网络搜索暂时不可用（尝试 {'→'.join(failed) or '所有'} 后端均失败：{last_err}）"
        if failed:
            self.last_note = f"[检索提示：{'→'.join(failed)} 失败，已顺延]"
        return result

    def _scope_to_xiaomi(self, query: str) -> str:
        """Prefix query with '小米SU7' if it lacks Xiaomi auto terms."""
        q = query.strip()
        if any(t in q for t in self._XIAOMI_TERMS):
            return q
        return f"小米SU7 {q}"

    def _bing(self, query: str) -> str:
        import requests
        headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_SEARCH_KEY"]}
        params  = {"q": query, "mkt": "zh-CN", "count": 5}
        resp    = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers=headers, params=params, timeout=10
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"
        results = resp.json().get("webPages", {}).get("value", [])
        if not results:
            return "网络搜索未找到相关结果。"
        parts = []
        for i, r in enumerate(results[:4], 1):
            parts.append(f"[{i}]【{r['name']}】\n{r['snippet']}\n网址：{r['url']}")
        return "\n\n".join(parts)

    def _serpapi(self, query: str) -> str:
        import requests
        params = {
            "q": query, "hl": "zh-cn", "gl": "cn",
            "api_key": os.environ["SERPAPI_KEY"], "num": 5,
        }
        resp    = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        results = resp.json().get("organic_results", [])
        if not results:
            return "网络搜索未找到相关结果。"
        parts = []
        for r in results[:4]:
            parts.append(
                f"【{r.get('title','')}】\n{r.get('snippet','')}\n网址：{r.get('link','')}"
            )
        return "\n\n".join(parts)

    def _serper(self, query: str) -> str:
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
        resp.encoding = "utf-8"
        results = resp.json().get("organic", [])
        if not results:
            return "网络搜索未找到相关结果。"
        parts = []
        for r in results[:4]:
            parts.append(
                f"【{r.get('title', '')}】\n{r.get('snippet', '')}\n网址：{r.get('link', '')}"
            )
        return "\n\n".join(parts)

    def _doubao(self, query: str) -> str:
        """Use Doubao LLM knowledge to simulate web search (no separate API needed)."""
        try:
            from openai import OpenAI
        except ImportError:
            return "网络搜索不可用：OpenAI SDK 未安装。"
        client = OpenAI(
            api_key=os.environ["DOUBAO_API_KEY"],
            base_url=os.environ["DOUBAO_BASE_URL"],
        )
        completion = client.chat.completions.create(
            model=os.environ.get("DOUBAO_MODEL_NAME", "ep-20240601170316-5dhwt"),
            messages=[{
                "role": "user",
                "content": (
                    f"请提供关于以下问题的准确网络信息，直接给出内容，"
                    f"不要有任何引导语：\n{query}"
                )
            }],
            max_tokens=600,
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()


# ────────────────────────────────────────────────────────────
# Core environment class
# ────────────────────────────────────────────────────────────

class RetrievalEnvironment:
    """
    Tool-call environment.

    During GRPO training, vLLM calls step() after each generated chunk;
    during inference, run_with_context() can be used with partial generation
    for routing decisions (see infer_rl.py for the full loop).
    """

    def __init__(self):
        self.local_backend = LocalSearchBackend.get_instance()
        self.web_backend   = WebSearchBackend()
        self.page_reader   = WebPageReader()

    # ── Core interface: process generated text, return injection ──
    def step(self, generated_text: str) -> tuple[Optional[str], bool]:
        """
        Detect pending tool-call tags at the end of generated_text.

        Returns:
          (information_block, is_done)
          - information_block: <information>...</information> string to inject,
                               None if no tool call pending
          - is_done: True if <answer> found, generation complete
        """
        # Check if already done
        if _RE_ANSWER.search(generated_text):
            return None, True

        # Detect tool calls
        local_match = _RE_SEARCH_LOCAL.findall(generated_text)
        web_match   = _RE_SEARCH_WEB.findall(generated_text)
        read_match  = _RE_READ_PAGE.findall(generated_text)

        # Existing information block count (to avoid re-processing)
        info_count   = len(_RE_INFORMATION.findall(generated_text))
        total_calls  = len(local_match) + len(web_match) + len(read_match)

        if total_calls <= info_count:
            # All existing calls already have corresponding information
            return None, False

        # Check read_page hop limit
        if len(read_match) > MAX_READ_PAGE_HOPS:
            return "<information>已达到最大页面阅读次数限制，请基于已有信息作答。</information>", False

        if total_calls > MAX_SEARCH_STEPS:
            # Exceeded max steps, force termination
            return (
                "<information>已达到最大检索次数限制。</information>\n"
                "<answer>根据已检索到的信息暂时无法给出完整答案，"
                "建议访问小米汽车官网获取最新信息。</answer>"
            ), True

        # Collect all calls ordered by position
        all_calls = []
        for m in _RE_SEARCH_LOCAL.finditer(generated_text):
            all_calls.append(("local", m.group(1).strip(), m.start()))
        for m in _RE_SEARCH_WEB.finditer(generated_text):
            all_calls.append(("web",   m.group(1).strip(), m.start()))
        for m in _RE_READ_PAGE.finditer(generated_text):
            all_calls.append(("read_page", m.group(1).strip(), m.start()))
        all_calls.sort(key=lambda x: x[2])  # sort by position

        # Take the info_count-th call (the next one awaiting response)
        call_type, query, _ = all_calls[info_count]

        if call_type == "local":
            result_str, score = self.local_backend.search(query)

            # Determine if web search upgrade is needed
            if score < RELEVANCE_THRESHOLD:
                info_block = (
                    f"<information>{result_str}\n"
                    f"[提示：本地知识库相关性较低（{score:.2f}），"
                    f"如需更准确信息可调用网络搜索]</information>"
                )
            else:
                info_block = f"<information>{result_str}</information>"

        elif call_type == "web":
            result_str = self.web_backend.search(query)
            info_block = f"<information>{result_str}</information>"

        else:  # read_page
            url = query.strip()
            page_content = self.page_reader.fetch(url)
            info_block = f"<information>{page_content}</information>"

        return info_block, False

    # ── Helper: extract final answer ──────────────────────────
    @staticmethod
    def extract_answer(full_text: str) -> str:
        match = _RE_ANSWER.search(full_text)
        return match.group(1).strip() if match else ""

    # ── Helper: determine if out-of-domain refusal ────────────
    @staticmethod
    def is_refusal(answer: str) -> bool:
        refusal_patterns = [
            "只能回答小米SU7相关问题",
            "无法回答此问题",
            "不在我的服务范围",
            "建议您咨询",
        ]
        return any(p in answer for p in refusal_patterns)

    # ── Helper: count tool calls (for reward function) ────────
    @staticmethod
    def count_search_calls(full_text: str) -> dict:
        return {
            "local":     len(_RE_SEARCH_LOCAL.findall(full_text)),
            "web":       len(_RE_SEARCH_WEB.findall(full_text)),
            "read_page": len(_RE_READ_PAGE.findall(full_text)),
        }
