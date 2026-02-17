"""Search Worker — consolidated web search via DuckDuckGo.

Provides three search modes:
1. Instant Answer (JSON API) — structured answers for factual queries
2. HTML Search — general web results via DuckDuckGo HTML
3. Page Fetch — retrieve and strip a single URL

No API key required.  Fully free and self-contained.

Usage:
    worker = SearchWorker()
    result = await worker.search("台北天氣")
    page   = await worker.fetch_page("https://example.com")
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

import httpx
from loguru import logger

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_TIMEOUT = 10.0


class SearchWorker:
    """DuckDuckGo Instant Answer + HTML search + page fetch."""

    def __init__(self) -> None:
        self.name = "search"

    # ── Worker interface ──────────────────────────────────────────

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Worker interface used by ReactExecutor.

        Args:
            task: Search query string.
            **kwargs: max_results (int, default 5).

        Returns:
            dict with ``result`` (text) and ``source``.
        """
        max_results = kwargs.get("max_results", 5)
        return await self.search(task, max_results=max_results)

    # ── Public API ────────────────────────────────────────────────

    async def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Run instant-answer + HTML search and merge results.

        Returns:
            dict with keys: result (str), instant (dict|None),
            results (list[dict]), source, worker.
        """
        instant = await self._ddg_instant(query)
        results = await self._ddg_search(query, max_results)

        # Build human-readable text
        parts: list[str] = []
        if instant and instant.get("abstract"):
            parts.append(f"📌 {instant['abstract']}")
            if instant.get("url"):
                parts.append(f"   來源: {instant['url']}")
            parts.append("")

        for i, r in enumerate(results, 1):
            parts.append(f"{i}. {r['title']}")
            if r.get("snippet"):
                parts.append(f"   {r['snippet']}")
            if r.get("url"):
                parts.append(f"   {r['url']}")

        text = "\n".join(parts) if parts else f"未找到「{query}」的搜尋結果。"

        return {
            "result": text,
            "instant": instant,
            "results": results,
            "source": "search",
            "worker": self.name,
        }

    async def fetch_page(self, url: str, max_chars: int = 5000) -> dict[str, Any]:
        """Fetch a URL and return stripped text content.

        Args:
            url: Full URL to fetch.
            max_chars: Truncate body text to this length.

        Returns:
            dict with ``result`` (str), ``url``, ``source``.
        """
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_TIMEOUT),
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return {
                        "error": f"HTTP {resp.status_code}",
                        "url": url,
                        "source": "search",
                        "worker": self.name,
                    }

                text = self._strip_html(resp.text)[:max_chars]
                return {
                    "result": text,
                    "url": url,
                    "source": "search",
                    "worker": self.name,
                }
        except Exception as e:
            logger.debug(f"SearchWorker fetch_page failed: {e}")
            return {
                "error": str(e),
                "url": url,
                "source": "search",
                "worker": self.name,
            }

    # ── Internal: DuckDuckGo Instant Answer API ───────────────────

    async def _ddg_instant(self, query: str) -> dict[str, Any] | None:
        """Query DuckDuckGo Instant Answer JSON API.

        Returns dict with abstract, url, answer, or None on failure.
        """
        url = f"https://api.duckduckgo.com/?q={quote_plus(query[:120])}&format=json&no_html=1"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_TIMEOUT),
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                abstract = data.get("AbstractText", "")
                answer = data.get("Answer", "")
                if not abstract and not answer:
                    return None
                return {
                    "abstract": abstract or answer,
                    "url": data.get("AbstractURL", ""),
                    "source_name": data.get("AbstractSource", ""),
                    "answer": answer,
                }
        except Exception as e:
            logger.debug(f"SearchWorker instant answer failed: {e}")
            return None

    # ── Internal: DuckDuckGo HTML search ─────────────────────────

    async def _ddg_search(
        self, query: str, max_results: int = 5,
    ) -> list[dict[str, str]]:
        """Scrape DuckDuckGo HTML search for general results."""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query[:120])}"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_TIMEOUT),
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return []
                return self._parse_ddg_html(resp.text, max_results)
        except Exception as e:
            logger.debug(f"SearchWorker HTML search failed: {e}")
            return []

    def _parse_ddg_html(self, html: str, max_results: int) -> list[dict[str, str]]:
        """Extract titles, snippets, and URLs from DuckDuckGo HTML."""
        results: list[dict[str, str]] = []

        # Title + URL pairs
        title_pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        # Snippets
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>',
            re.DOTALL,
        )

        titles = title_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (url, title) in enumerate(titles[:max_results]):
            entry: dict[str, str] = {
                "title": self._strip_html(title).strip(),
                "url": url.strip(),
            }
            if i < len(snippets):
                entry["snippet"] = self._strip_html(snippets[i]).strip()
            results.append(entry)

        return results

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags and collapse whitespace."""
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    async def close(self) -> None:
        """Cleanup (no persistent resources)."""
