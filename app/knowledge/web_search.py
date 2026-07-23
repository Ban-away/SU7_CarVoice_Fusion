"""Web search client — SerpAPI → Serper → Bing → Doubao chain with mock fallback."""

import logging, os, requests

from app.knowledge.models import RetrievedDoc
from app.shared.config import get_settings

logger = logging.getLogger(__name__)


class WebSearchClient:
    """Real web search: SerpAPI → Serper → Bing → Doubao cascade."""

    _FALLBACK = "垂直检索补充：建议参考小米汽车官方帮助中心获取最新信息。"

    def __init__(self, enabled: bool | None = None) -> None:
        self._enabled = enabled if enabled is not None else get_settings().web_search_enabled

    # ── Public API ──────────────────────────────────────────────────────

    def search(self, query: str) -> list[RetrievedDoc]:
        if not self._enabled:
            return []
        content = self._do_search(query)
        return [RetrievedDoc(content=content, source="web_search", score=0.7)]

    # ── Internal cascade ────────────────────────────────────────────────

    def _do_search(self, query: str) -> str:
        backends = []
        if os.getenv("SERPAPI_KEY"): backends.append("serpapi")
        if os.getenv("SERPER_API_KEY"): backends.append("serper")
        if os.getenv("BING_SEARCH_KEY"): backends.append("bing")
        backends.append("doubao")

        for be in backends:
            try:
                return getattr(self, f"_{be}")(query)
            except Exception:
                logger.warning("Web search backend %s failed", be)
        return self._FALLBACK

    def _serpapi(self, query: str) -> str:
        resp = requests.get("https://serpapi.com/search", params={
            "q": query, "hl": "zh-cn", "gl": "cn",
            "api_key": os.environ["SERPAPI_KEY"], "num": 5,
        }, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("organic_results", [])
        if not results: return ""
        return "\n\n".join(
            f"【{r.get('title','')}】\n{r.get('snippet','')}\n网址：{r.get('link','')}"
            for r in results[:4]
        )

    def _serper(self, query: str) -> str:
        resp = requests.post("https://google.serper.dev/search",
            headers={"X-API-KEY": os.environ["SERPER_API_KEY"], "Content-Type": "application/json"},
            json={"q": query, "hl": "zh-cn", "gl": "cn", "num": 5}, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("organic", [])
        if not results: return ""
        return "\n\n".join(
            f"【{r.get('title','')}】\n{r.get('snippet','')}\n网址：{r.get('link','')}"
            for r in results[:4]
        )

    def _bing(self, query: str) -> str:
        resp = requests.get("https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": os.environ["BING_SEARCH_KEY"]},
            params={"q": query, "mkt": "zh-CN", "count": 5}, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("webPages", {}).get("value", [])
        if not results: return ""
        return "\n\n".join(
            f"【{r.get('name','')}】\n{r.get('snippet','')}\n网址：{r.get('url','')}"
            for r in results[:4]
        )

    def _doubao(self, query: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["DOUBAO_API_KEY"], base_url=os.environ["DOUBAO_BASE_URL"])
        r = client.chat.completions.create(
            model=os.environ.get("DOUBAO_MODEL_NAME", ""),
            messages=[{"role": "user", "content": f"请提供关于以下问题的准确网络信息：\n{query}"}],
            max_tokens=512, temperature=0.3)
        return r.choices[0].message.content or ""

    @property
    def enabled(self) -> bool:
        return self._enabled
