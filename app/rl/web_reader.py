# -*- coding: utf-8 -*-
"""
Web page content fetcher — vertical search infrastructure.

Fetches web page content from URLs and converts to plain text,
used by the <read_page> tool in RL inference.

Usage:
  from app.rl.web_reader import WebPageReader
  reader = WebPageReader()
  content = reader.fetch("https://www.xiaomi.com/...", max_chars=2000)

Dependencies:
  - requests (HTTP requests)
  - inscriptis (HTML to plain text, good Chinese support, optional)
"""

import re
import logging
from typing import Optional
from urllib.parse import urlparse

import requests

from app.shared.config import get_settings

logger = logging.getLogger(__name__)

# ── Non-HTML suffix blacklist ─────────────────────────────────
_BINARY_EXTENSIONS = frozenset({
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dmg", ".iso", ".apk",
})

# ── Default configuration ────────────────────────────────────
DEFAULT_TIMEOUT      = 10    # seconds
DEFAULT_MAX_CHARS    = 2000  # max chars per page
DEFAULT_MAX_SIZE     = 2 * 1024 * 1024  # 2MB
DEFAULT_USER_AGENT   = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class WebPageReader:
    """
    Web page content fetcher, thread-safe.

    Responsibilities:
      1. Fetch HTML from a URL
      2. Filter non-HTML resources
      3. HTML -> plain text (strip tags, scripts, styles)
      4. Truncate to specified length
    """

    def __init__(
        self,
        timeout:   int = DEFAULT_TIMEOUT,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_size:  int = DEFAULT_MAX_SIZE,
    ):
        self.timeout   = timeout
        self.max_chars = max_chars
        self.max_size  = max_size
        self._session  = requests.Session()
        self._session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    def fetch(self, url: str, max_chars: Optional[int] = None) -> str:
        """
        Fetch and parse web page content.

        Args:
            url:       Target URL
            max_chars: Max returned characters (None uses default)

        Returns:
            Plain text content, or an error message string on failure
        """
        max_chars = max_chars or self.max_chars

        # URL format validation
        if not self.validate_url(url):
            return f"无法读取页面：URL地址无效或格式不支持（{url[:100]}）"

        try:
            return self._do_fetch(url, max_chars)
        except requests.Timeout:
            return f"页面读取超时：{self._short_url(url)}"
        except requests.ConnectionError:
            return f"无法连接到页面：{self._short_url(url)}"
        except requests.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            return f"页面返回错误（HTTP {code}）：{self._short_url(url)}"
        except Exception as e:
            return f"页面读取失败：{str(e)[:100]}"

    def _do_fetch(self, url: str, max_chars: int) -> str:
        """Actual fetch logic."""
        resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()

        # Check Content-Type
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"不支持的内容类型：{content_type[:50]}"

        # Check size
        content_length = len(resp.content)
        if content_length > self.max_size:
            return f"页面内容过大（{content_length // 1024}KB），已跳过"

        # Encoding detection: requests defaults to ISO-8859-1 for
        # text/html without declared charset, causing garbled Chinese.
        # Prefer chardet-detected apparent_encoding, then fallback
        # through common Chinese encodings (gb18030 covers GBK).
        content = resp.content
        encoding = resp.encoding
        if not encoding or encoding.lower() in ("iso-8859-1", "latin-1"):
            encoding = resp.apparent_encoding or "utf-8"
        html = None
        for enc in (encoding, "utf-8", "gb18030", "gbk"):
            try:
                html = content.decode(enc, errors="strict")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if html is None:
            html = content.decode("utf-8", errors="replace")

        # HTML -> plain text
        text = self._html_to_text(html)

        # Extract domain as source label
        domain = urlparse(url).netloc

        # Truncate
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return f"[页面来源：{domain}]\n{text}"

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text, preferring inscriptis with regex fallback."""
        try:
            import inscriptis
            text = inscriptis.get_text(html)
        except ImportError:
            logger.debug("inscriptis not available, using regex fallback for HTML stripping")
            text = self._simple_html_strip(html)

        # Clean excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _simple_html_strip(html: str) -> str:
        """Simple HTML tag stripping (no third-party dependency fallback)."""
        # Remove script/style blocks
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>",  "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common HTML entities
        for entity, char in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')]:
            text = text.replace(entity, char)
        return text

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format for fetch-ability."""
        if not url or not isinstance(url, str):
            return False

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return False

        try:
            parsed = urlparse(url)
        except Exception:
            return False

        if not parsed.netloc:
            return False

        # Filter non-HTML suffixes
        path_lower = parsed.path.lower()
        for ext in _BINARY_EXTENSIONS:
            if path_lower.endswith(ext):
                return False

        return True

    @staticmethod
    def _short_url(url: str) -> str:
        """Shorten URL for error messages."""
        if len(url) > 80:
            return url[:77] + "..."
        return url


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    reader = WebPageReader()

    test_urls = [
        "https://www.mi.com",                    # Xiaomi official
        "https://www.xiaomiev.com",              # Xiaomi EV
        "https://invalid-test-12345.com",         # Non-existent domain
        "https://example.com/test.pdf",           # Non-HTML
    ]

    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        result = reader.fetch(url)
        print(result[:500])
        print(f"  ... ({len(result)} chars)")
