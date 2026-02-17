"""Browser Worker — web fetch and browser automation.

Provides three capabilities:
1. HTTP fetch — retrieve web page content via httpx (headless, fast)
2. Chrome open — launch URL in isolated Chrome profile (visual)
3. Playwright automation — interact with web pages (Google Maps search, etc.)
   Uses screenshot → Vision LLM for page understanding.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

try:
    from playwright.async_api import async_playwright, BrowserContext
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

# Chrome profile for JARVIS — uses the main Chrome profile
# (user confirmed Chrome is dedicated for JARVIS use)
JARVIS_BROWSER_PROFILE = os.path.expanduser(
    "~/AppData/Local/Google/Chrome/User Data"
)

# Sensible limits
MAX_RESPONSE_SIZE = 1_000_000  # 1 MB text limit
FETCH_TIMEOUT = 20.0

# Vision prompt for extracting restaurant info from Maps screenshot
_MAPS_EXTRACT_PROMPT = """你看到的是 Google Maps 的搜尋結果頁面截圖。
請提取以下資訊，以 JSON 格式回覆（找不到的欄位填 null）：
{
  "name": "店名",
  "phone": "電話號碼",
  "address": "完整地址",
  "rating": "評分（如 4.8）",
  "booking_url": "訂位/預約連結（如果頁面上有的話）"
}
只回覆 JSON，不要其他文字。"""


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using stdlib only.

    Strips <script>, <style>, <nav>, <header>, <footer> blocks,
    then removes remaining tags and collapses whitespace.
    """
    # Remove script/style/nav blocks entirely
    text = re.sub(
        r'<\s*(script|style|nav|header|footer|noscript|svg)[^>]*>.*?</\s*\1\s*>',
        ' ', html, flags=re.DOTALL | re.IGNORECASE,
    )
    # Replace <br>, <p>, <div>, <li>, <tr> with newlines for readability
    text = re.sub(r'<\s*(?:br|/p|/div|/li|/tr|/h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    import html as _html
    text = _html.unescape(text)
    # Collapse whitespace (keep newlines for structure)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


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
        model_router: Any = None,
        headless: bool = False,
    ):
        self.security = security_gate
        self.chrome_path = chrome_path or os.environ.get(
            "BROWSER", r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        )
        self.user_data_dir = user_data_dir
        self.name = "browser"
        self._headless = headless
        self._http_client: httpx.AsyncClient | None = None
        self._router = model_router  # for screenshot → Vision LLM

        # Playwright state (lazy init)
        self._pw: Any = None
        self._pw_context: Any = None
        self._browser: Any = None  # CDP browser handle
        self._pw_available: bool | None = None  # cache Playwright/CDP status

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
                raw = resp.text[:MAX_RESPONSE_SIZE]
                # Convert HTML to plain text for LLM consumption
                if "html" in content_type:
                    body = _html_to_text(raw)
                else:
                    body = raw
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

    # ── Playwright automation ────────────────────────────────────

    async def _ensure_context(self) -> Any:
        """Lazy-init: connect to user's Chrome via CDP or launch fresh.

        Strategy:
        1. Try CDP connection to localhost:9222 (if Chrome started with debugging)
        2. If no debugger, kill Chrome → relaunch with user profile + debugging
        3. Connect via CDP → get the existing browser context with all cookies
        """
        if not _HAS_PLAYWRIGHT:
            raise RuntimeError("playwright not installed")
        if self._pw_available is False:
            raise RuntimeError("Playwright previously failed, skipping")
        if self._pw_context:
            return self._pw_context

        self._pw = await async_playwright().start()

        cdp_url = "http://localhost:9222"

        # Try connecting to existing Chrome with debugging
        try:
            self._browser = await self._pw.chromium.connect_over_cdp(cdp_url)
            self._pw_context = self._browser.contexts[0]
            self._pw_available = True
            logger.info("Connected to existing Chrome via CDP")
            return self._pw_context
        except Exception:
            logger.debug("No Chrome debugger found, will launch Chrome")

        # Kill existing Chrome and relaunch with remote debugging
        import subprocess
        subprocess.run(
            ["taskkill", "/F", "/IM", "chrome.exe"],
            capture_output=True,
        )
        await asyncio.sleep(2)

        # Launch Chrome with user profile + remote debugging
        chrome_args = [
            self.chrome_path,
            f"--user-data-dir={self.user_data_dir}",
            "--remote-debugging-port=9222",
            "--restore-last-session",
            "--disable-blink-features=AutomationControlled",
        ]
        subprocess.Popen(chrome_args)

        # Wait for Chrome to be ready with retries
        for attempt in range(10):
            await asyncio.sleep(2)
            try:
                self._browser = await self._pw.chromium.connect_over_cdp(cdp_url)
                self._pw_context = self._browser.contexts[0]
                self._pw_available = True
                logger.info(f"Connected to Chrome via CDP (attempt {attempt + 1})")
                return self._pw_context
            except Exception:
                logger.debug(f"CDP connect attempt {attempt + 1} failed, retrying...")

        self._pw_available = False
        raise RuntimeError("Failed to connect to Chrome after 10 attempts")

    async def _screenshot_to_data_url(self, page: Any) -> str:
        """Take a screenshot and return as base64 data URL for Vision LLM."""
        png_bytes = await page.screenshot(type="png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    async def _vision_extract(self, data_url: str, prompt: str) -> dict[str, Any]:
        """Send screenshot to Vision LLM and parse JSON response."""
        if not self._router:
            return {"error": "no model_router for vision"}
        resp = await self._router.vision_analyze(image_url=data_url, prompt=prompt)
        text = resp.content.strip()
        # Strip markdown code fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Vision LLM returned non-JSON: {text[:200]}")
            return {"error": f"Vision parse failed: {text[:200]}"}

    async def search_google_maps(self, query: str) -> dict[str, Any]:
        """Search Google Maps → {name, phone, address, booking_url, rating}.

        Strategy: navigate → wait for page load → screenshot → Vision LLM extracts info.
        Falls back to aria-label parsing if Vision unavailable.
        """
        if not _HAS_PLAYWRIGHT:
            return {"error": "playwright not installed", "worker": self.name}

        try:
            ctx = await self._ensure_context()
        except Exception as e:
            logger.warning(f"Playwright context init failed: {e}")
            return {"error": str(e), "worker": self.name}

        page = await ctx.new_page()
        try:
            from urllib.parse import quote_plus as _qp
            url = f"https://www.google.com/maps/search/{_qp(query)}"
            await page.goto(url, timeout=20000)

            # Wait for either feed (multiple results) or detail panel (single result)
            try:
                await page.wait_for_selector(
                    '[role="feed"], [data-item-id="address"], h1',
                    timeout=10000,
                )
            except Exception:
                pass  # proceed anyway, let vision handle it

            # If feed view, click first result to get detail
            feed = page.locator('[role="feed"] > div')
            if await feed.count() > 0:
                await feed.first.click()
                await page.wait_for_timeout(2500)
            else:
                await page.wait_for_timeout(2000)

            # Save debug screenshot
            try:
                await page.screenshot(path="./data/maps_debug.png")
                logger.debug("Maps debug screenshot saved to data/maps_debug.png")
            except Exception:
                pass

            # ── Strategy 1: Vision LLM (robust, adapts to any layout) ──
            info: dict[str, Any] = {}
            if self._router:
                try:
                    data_url = await self._screenshot_to_data_url(page)
                    info = await self._vision_extract(data_url, _MAPS_EXTRACT_PROMPT)
                    if "error" in info:
                        logger.warning(f"Vision extract failed: {info.get('error')}")
                        info = {}
                    else:
                        logger.info(f"Maps vision extract: {info.get('name')} / {info.get('phone')}")
                except Exception as e:
                    logger.warning(f"Vision extract error: {e}")

            # ── Strategy 2: Semantic selectors fill gaps ──
            if not info.get("name"):
                h1 = page.locator("h1").first
                if await h1.count() > 0:
                    info["name"] = (await h1.inner_text()).strip()

            if not info.get("phone"):
                phone_el = page.locator('[data-item-id^="phone:tel:"]').first
                if await phone_el.count() > 0:
                    aria = await phone_el.get_attribute("aria-label") or ""
                    info["phone"] = re.sub(r'^[^:]+:\s*', '', aria).strip() or None

            if not info.get("address"):
                addr_el = page.locator('[data-item-id="address"]').first
                if await addr_el.count() > 0:
                    aria = await addr_el.get_attribute("aria-label") or ""
                    info["address"] = re.sub(r'^[^:]+:\s*', '', aria).strip() or None

            if not info.get("rating"):
                rating_el = page.locator('span[role="img"][aria-label]').first
                if await rating_el.count() > 0:
                    aria = await rating_el.get_attribute("aria-label") or ""
                    m = re.search(r'(\d\.\d)', aria)
                    if m:
                        info["rating"] = m.group(1)

            # ── Booking URL detection (with scroll) ──
            reserve_selectors = [
                'a:has-text("訂位")',
                'a:has-text("預約")',
                'a:has-text("預訂")',
                'a:has-text("Reserve")',
                'a:has-text("線上預訂")',
                'a:has-text("線上訂位")',
                'a:has-text("Reserve a table")',
                '[data-item-id="booking"] a',
                'a[href*="inline.app"]',
                'a[href*="eztable"]',
                'a[href*="booking"]',
                'a[href*="reserve"]',
                'button:has-text("訂位")',
                'button:has-text("預訂")',
            ]

            async def _try_booking_selectors() -> str | None:
                for sel in reserve_selectors:
                    try:
                        btn = page.locator(sel).first
                        if await btn.count() > 0:
                            href = await btn.get_attribute("href")
                            if href:
                                logger.info(f"Found booking URL via '{sel}': {href}")
                                return href
                    except Exception:
                        continue
                return None

            if not info.get("booking_url"):
                info["booking_url"] = await _try_booking_selectors()

            # Scroll LEFT PANEL down to reveal booking section
            # Maps detail panel is in the left container
            if not info.get("booking_url"):
                # Position mouse over left panel (x=200 is well within the panel)
                await page.mouse.move(200, 400)
                for scroll_attempt in range(3):
                    await page.mouse.wheel(0, 500)
                    await page.wait_for_timeout(1200)
                    url = await _try_booking_selectors()
                    if url:
                        info["booking_url"] = url
                        break

                # Save debug screenshot after scrolling
                if not info.get("booking_url"):
                    try:
                        await page.screenshot(path="./data/maps_debug_scrolled.png")
                        logger.debug("Maps scrolled screenshot saved")
                    except Exception:
                        pass

            # Look for official website link
            if not info.get("website"):
                web_el = page.locator('[data-item-id="authority"]').first
                if await web_el.count() > 0:
                    aria = await web_el.get_attribute("aria-label") or ""
                    info["website"] = re.sub(r'^[^:]+:\s*', '', aria).strip() or None

            info["worker"] = self.name
            return info

        except Exception as e:
            logger.warning(f"Google Maps search failed: {e}")
            return {"error": str(e), "worker": self.name}
        finally:
            await page.close()

    async def find_booking_url(self, restaurant_name: str) -> str | None:
        """Search DuckDuckGo for restaurant booking URL (httpx, no Playwright).

        Looks for inline.app, eztable, opentable, or other booking platforms.
        DuckDuckGo wraps URLs in redirect links, so we decode those too.

        Returns:
            booking URL string or None
        """
        from urllib.parse import quote_plus as _qp, unquote as _uq

        query = f"{restaurant_name} 訂位"
        url = f"https://html.duckduckgo.com/html/?q={_qp(query)}"
        client = await self._get_client()
        try:
            resp = await client.get(url)
            html = resp.text

            # DuckDuckGo wraps URLs: uddg=https%3A%2F%2Finline.app%2F...
            # First extract all uddg URLs, then check for booking domains
            uddg_urls = re.findall(r'uddg=([^&"]+)', html)
            decoded_urls = [_uq(u) for u in uddg_urls]

            # Priority order of booking platforms
            booking_domains = [
                "inline.app",
                "eztable.com",
                "opentable.com",
                "eatogether.com",
                "autoreserve.com",
            ]
            for domain in booking_domains:
                for decoded in decoded_urls:
                    if domain in decoded:
                        logger.info(f"Found booking URL via web search: {decoded}")
                        return decoded

            # Also try direct regex on HTML for any we missed
            for domain in booking_domains:
                pat = rf'(https?://[^\s"\'<>&]*{re.escape(domain)}[^\s"\'<>&]*)'
                matches = re.findall(pat, html)
                if matches:
                    url_found = _uq(matches[0])
                    logger.info(f"Found booking URL via regex: {url_found}")
                    return url_found

            logger.debug(f"No booking URL found in web search for '{restaurant_name}'")
            return None

        except Exception as e:
            logger.warning(f"Booking URL search failed: {e}")
            return None

    async def navigate_and_click(
        self, url: str, selector: str,
    ) -> dict[str, Any]:
        """Navigate to a URL and click an element, return page text."""
        if not _HAS_PLAYWRIGHT:
            return {"error": "playwright not installed", "worker": self.name}

        try:
            ctx = await self._ensure_context()
        except Exception as e:
            return {"error": str(e), "worker": self.name}

        page = await ctx.new_page()
        try:
            await page.goto(url, timeout=15000)
            await page.wait_for_selector(selector, timeout=8000)
            await page.locator(selector).first.click()
            await page.wait_for_timeout(2000)
            content = await page.inner_text("body")
            return {
                "status": "ok",
                "url": page.url,
                "content": content[:MAX_RESPONSE_SIZE],
                "worker": self.name,
            }
        except Exception as e:
            logger.warning(f"navigate_and_click failed: {e}")
            return {"error": str(e), "worker": self.name}
        finally:
            await page.close()

    # ── Vision Agent Loop ────────────────────────────────────────

    async def vision_agent_step(
        self, page: Any, goal: str, history: list[str],
    ) -> dict[str, Any]:
        """One step of the vision agent loop.

        Takes screenshot → asks Vision LLM what to do → returns action dict.
        Actions: click, fill, navigate, scroll, done, fail
        """
        if not self._router:
            return {"action": "fail", "reason": "no model_router for vision"}

        data_url = await self._screenshot_to_data_url(page)
        current_url = page.url

        history_text = "\n".join(f"  Step {i+1}: {h}" for i, h in enumerate(history))

        prompt = (
            f"## 任務目標\n{goal}\n\n"
            f"## 目前網址\n{current_url}\n\n"
        )
        if history_text:
            prompt += f"## 已執行步驟\n{history_text}\n\n"
        prompt += (
            "## 指令\n"
            "看這張截圖，決定下一步該做什麼來完成任務。\n"
            "用 JSON 格式回覆一個動作：\n\n"
            '- 點擊: {"action":"click","x":數字,"y":數字,"desc":"點了什麼"}\n'
            '- 填寫: {"action":"fill","x":數字,"y":數字,"text":"要填的文字","desc":"填了什麼"}\n'
            '- 打開網址: {"action":"navigate","url":"完整網址","desc":"為什麼要去"}\n'
            '- 向下捲動: {"action":"scroll","desc":"為什麼要捲"}\n'
            '- 任務完成: {"action":"done","result":"結果摘要"}\n'
            '- 無法完成: {"action":"fail","reason":"失敗原因"}\n\n'
            "只回覆 JSON，不要其他文字。"
        )

        try:
            resp = await self._router.vision_analyze(image_url=data_url, prompt=prompt)
            text = resp.content.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Vision agent non-JSON: {resp.content[:200]}")
            return {"action": "fail", "reason": f"LLM parse error: {resp.content[:100]}"}
        except Exception as e:
            logger.warning(f"Vision agent error: {e}")
            return {"action": "fail", "reason": str(e)}

    async def _execute_agent_action(
        self, page: Any, action: dict[str, Any],
    ) -> str:
        """Execute a vision agent action on the page. Returns description."""
        act = action.get("action", "")
        desc = action.get("desc", "")

        if act == "click":
            x, y = action.get("x", 0), action.get("y", 0)
            await page.mouse.click(x, y)
            await page.wait_for_timeout(2000)
            return f"Clicked ({x},{y}): {desc}"

        elif act == "fill":
            x, y = action.get("x", 0), action.get("y", 0)
            text = action.get("text", "")
            await page.mouse.click(x, y)
            await page.wait_for_timeout(500)
            await page.keyboard.type(text, delay=50)
            await page.wait_for_timeout(500)
            return f"Filled ({x},{y}) with '{text}': {desc}"

        elif act == "navigate":
            url = action.get("url", "")
            await page.goto(url, timeout=15000)
            await page.wait_for_timeout(2000)
            return f"Navigated to {url}: {desc}"

        elif act == "scroll":
            await page.mouse.wheel(0, 400)
            await page.wait_for_timeout(1500)
            return f"Scrolled down: {desc}"

        return f"Unknown action: {act}"

    async def complete_booking(
        self,
        restaurant_info: dict[str, Any],
        booking_details: dict[str, Any],
        max_steps: int = 12,
    ) -> dict[str, Any]:
        """Complete a restaurant booking end-to-end using vision agent.

        Args:
            restaurant_info: {name, phone, address, booking_url, website, ...}
            booking_details: {date, time, people, name, phone}
            max_steps: max agent loop iterations

        Returns:
            dict with status, result/error
        """
        if not _HAS_PLAYWRIGHT:
            return {"error": "playwright not installed", "worker": self.name}
        if not self._router:
            return {"error": "no model_router for vision agent", "worker": self.name}

        # Decide starting URL
        start_url = (
            restaurant_info.get("booking_url")
            or restaurant_info.get("website")
        )
        # Ensure URL has protocol
        if start_url and not start_url.startswith(("http://", "https://")):
            start_url = f"https://{start_url}"
        if not start_url:
            return {
                "error": "no_booking_url",
                "phone": restaurant_info.get("phone"),
                "message": "找不到訂位連結或官網，請用電話訂位",
                "worker": self.name,
            }

        goal = (
            f"在這個網站完成餐廳訂位。\n"
            f"餐廳: {restaurant_info.get('name', '未知')}\n"
            f"訂位資訊: {booking_details.get('date', '')} "
            f"{booking_details.get('time', '')} "
            f"{booking_details.get('people', '')}人\n"
            f"訂位人: {booking_details.get('name', 'Ted')}\n"
            f"電話: {booking_details.get('phone', '')}\n"
            "找到訂位表單 → 填寫 → 送出 → 確認完成。"
        )

        try:
            ctx = await self._ensure_context()
        except Exception as e:
            return {"error": str(e), "worker": self.name}

        page = await ctx.new_page()
        history: list[str] = []

        try:
            await page.goto(start_url, timeout=15000)
            await page.wait_for_timeout(2000)
            history.append(f"Opened {start_url}")

            for step in range(max_steps):
                logger.info(f"Booking agent step {step + 1}/{max_steps}")
                action = await self.vision_agent_step(page, goal, history)
                act = action.get("action", "")

                if act == "done":
                    result = action.get("result", "訂位完成")
                    logger.info(f"Booking agent done: {result}")
                    # Take final screenshot as proof
                    screenshot_path = f"./data/booking_done_{restaurant_info.get('name', 'unknown')}.png"
                    await page.screenshot(path=screenshot_path)
                    return {
                        "status": "booked",
                        "result": result,
                        "screenshot": screenshot_path,
                        "steps": len(history),
                        "worker": self.name,
                    }

                if act == "fail":
                    reason = action.get("reason", "unknown")
                    logger.warning(f"Booking agent failed: {reason}")
                    # If CAPTCHA/verification, return booking_url for user
                    is_captcha = any(kw in reason.lower() for kw in [
                        "captcha", "verification", "human", "robot",
                        "驗證", "機器人", "verify",
                    ])
                    return {
                        "error": reason,
                        "captcha": is_captcha,
                        "phone": restaurant_info.get("phone"),
                        "booking_url": start_url if is_captcha else None,
                        "steps": len(history),
                        "worker": self.name,
                    }

                # Execute the action
                step_desc = await self._execute_agent_action(page, action)
                history.append(step_desc)
                logger.debug(f"Booking agent: {step_desc}")

            return {
                "error": "max_steps_exceeded",
                "phone": restaurant_info.get("phone"),
                "steps": len(history),
                "worker": self.name,
            }

        except Exception as e:
            logger.warning(f"Booking agent error: {e}")
            return {"error": str(e), "worker": self.name}
        finally:
            await page.close()

    async def close_playwright(self) -> None:
        """Disconnect from Chrome (don't close the browser itself)."""
        # For CDP mode, we just disconnect — Chrome stays running for user
        if self._browser:
            try:
                self._browser.close()
            except Exception as e:
                logger.debug(f"Browser disconnect error: {e}")
            self._browser = None
        self._pw_context = None
        if self._pw:
            try:
                await self._pw.stop()
            except Exception as e:
                logger.debug(f"Playwright stop error: {e}")
            self._pw = None

    async def close(self) -> None:
        """Close the HTTP client and Playwright."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
        await self.close_playwright()
