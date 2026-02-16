"""Browser Worker — web fetch and browser automation.

Provides two capabilities:
1. HTTP fetch — retrieve web page content via httpx (headless, fast)
2. Chrome open — launch URL in isolated Chrome profile (visual)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# Isolated Chrome profile — never touches user's main profile
JARVIS_BROWSER_PROFILE = "./data/chrome_profile_jarvis"

# Sensible limits
MAX_RESPONSE_SIZE = 1_000_000  # 1 MB text limit
FETCH_TIMEOUT = 20.0


class BrowserWorker:
    """Worker for web tasks — HTTP fetch and Chrome automation.

    Usage:
        worker = BrowserWorker()
        result = await worker.execute("查詢台北今日天氣", url="https://...")
    """

    def __init__(
        self,
        security_gate: Any = None,
        chrome_path: str | None = None,
        user_data_dir: str = JARVIS_BROWSER_PROFILE,
    ):
        self.security = security_gate
        self.chrome_path = chrome_path or os.environ.get(
            "BROWSER", r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        )
        self.user_data_dir = user_data_dir
        self.name = "browser"
        self._http_client: httpx.AsyncClient | None = None

        # Ensure isolated profile directory exists
        Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(FETCH_TIMEOUT),
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                },
            )
        return self._http_client

    async def fetch_url(self, url: str) -> dict[str, Any]:
        """Fetch a URL and return its text content.

        Returns:
            dict with status, url, content (text body), content_type, etc.
        """
        # Security check
        if self.security:
            verdict = self.security.check_browser_url(url)
            if verdict == "block":
                return {"error": f"URL blocked by security gate: {url}"}

        client = await self._get_client()
        try:
            resp = await client.get(url)
            content_type = resp.headers.get("content-type", "")

            # Only return text-based content
            if "text" in content_type or "json" in content_type or "xml" in content_type:
                body = resp.text[:MAX_RESPONSE_SIZE]
            else:
                body = f"[Binary content: {content_type}, {len(resp.content)} bytes]"

            logger.info(
                f"BrowserWorker fetched: {url} "
                f"({resp.status_code}, {len(body)} chars)"
            )
            return {
                "status": "ok",
                "url": str(resp.url),
                "http_status": resp.status_code,
                "content_type": content_type,
                "content": body,
                "worker": self.name,
            }

        except httpx.TimeoutException:
            logger.warning(f"Fetch timeout: {url}")
            return {"error": f"Request timed out: {url}", "worker": self.name}
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return {"error": str(e), "worker": self.name}

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a browser task.

        If a url is provided, fetches it. Otherwise describes capability.

        Args:
            task: description of the web task
            **kwargs: url, action, etc.

        Returns:
            dict with result and metadata
        """
        url = kwargs.get("url")

        if url:
            return await self.fetch_url(url)

        # No URL provided — inform CEO what we can do
        logger.info(f"BrowserWorker: {task[:60]}...")
        return {
            "status": "ready",
            "task": task,
            "note": (
                "Provide a URL via url= to fetch web content. "
                "I can retrieve and read web pages."
            ),
            "worker": self.name,
        }

    async def open_url(self, url: str) -> dict[str, Any]:
        """Open a URL in Chrome with isolated profile."""
        import subprocess

        # Security check
        if self.security:
            verdict = self.security.check_browser_url(url)
            if verdict == "block":
                return {"error": f"URL blocked by security gate: {url}"}

        try:
            subprocess.Popen([
                self.chrome_path,
                f"--user-data-dir={os.path.abspath(self.user_data_dir)}",
                url,
            ])
            return {"status": "opened", "url": url}
        except Exception as e:
            logger.error(f"Failed to open URL: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
