"""Assist Worker — 做到 90% 給選項.

核心哲學：
  不是「做不到，這是連結」
  而是「我幫你做到這裡了，剩下你選一個我接著處理」

When full automation fails (browser, API), this worker:
1. Searches the web for relevant info (phone, address, hours, etc.)
2. Checks memory for past context
3. Uses LLM to extract structured data from search results
4. Builds a helpful message with actionable options

Usage:
    worker = AssistWorker(model_router=router, memory_search=search)
    result = await worker.execute("幫我訂 Niku Mura", task_type="booking")
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote_plus

import httpx
from loguru import logger


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_SEARCH_TIMEOUT = 10.0


class AssistWorker:
    """When automation fails, do 90% of the work and give user options."""

    def __init__(
        self,
        model_router: Any = None,
        memory_search: Any = None,
        gog_worker: Any = None,
    ):
        self.router = model_router
        self.memory_search = memory_search
        self.gog = gog_worker
        self.name = "assist"

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Worker interface — dispatches based on task_type.

        Args:
            task: Task description (e.g. "幫我訂 Niku Mura 明天 7 點 2 位").
            **kwargs:
                task_type: "booking", "ticket", or "generic".
                task_context: dict with parsed info (restaurant, date, time, people).
                failed_attempts: list of prior failed attempt dicts.

        Returns:
            dict with result, is_partial=True, and worker name.
        """
        task_type = kwargs.get("task_type", "generic")
        task_context = kwargs.get("task_context", {})
        failed_attempts = kwargs.get("failed_attempts", [])

        if task_type == "booking":
            return await self._assist_booking(task, task_context, failed_attempts)
        elif task_type == "ticket":
            return await self._assist_ticket(task, task_context, failed_attempts)
        else:
            return await self._assist_generic(task, failed_attempts)

    # ── Booking assist ───────────────────────────────────────────

    async def _assist_booking(
        self,
        task: str,
        context: dict,
        failed: list[dict],
    ) -> dict[str, Any]:
        """Booking failed → collect restaurant info + give options."""
        restaurant = context.get("restaurant") or self._extract_restaurant(task)
        date = context.get("date", "")
        time_ = context.get("time", "")
        people = context.get("people", "2")

        # 1. Web search for restaurant info
        search_text = await self._web_search(f"{restaurant} 電話 地址 營業時間 訂位")

        # 2. Memory search
        memory_text = self._search_memory(restaurant)

        # 3. LLM extraction
        extracted = await self._extract_info(restaurant, search_text, memory_text)

        # 4. Build message with options
        phone = extracted.get("phone", "未找到")
        address = extracted.get("address", "未找到")
        hours = extracted.get("hours", "未找到")
        booking_url = extracted.get("booking_url")

        parts = [
            f"Sir，{restaurant} 的自動訂位需要驗證，我沒辦法自動完成。",
            f"但我已經幫你查好了：",
            "",
            f"📞 電話: {phone}",
            f"📍 地址: {address}",
            f"🕐 營業: {hours}",
            f"📋 訂位資訊: {people} 位 / {date} {time_}",
            "",
            "你要怎麼處理？",
        ]

        options = []
        label = "A"
        if booking_url:
            options.append(f"{label}. 開訂位頁面，你只需填驗證碼 → {booking_url}")
            label = chr(ord(label) + 1)
        if phone and phone != "未找到":
            options.append(f"{label}. 你打電話，我把要講的話準備好給你")
            label = chr(ord(label) + 1)
        options.append(f"{label}. 幫你查其他類似餐廳可以線上訂的")

        parts.extend(options)

        return {
            "result": "\n".join(parts),
            "is_partial": True,
            "source": "assist",
            "worker": self.name,
            "restaurant_info": extracted,
            "phone": phone if phone != "未找到" else None,
            "booking_url": booking_url,
        }

    # ── Ticket assist ────────────────────────────────────────────

    async def _assist_ticket(
        self,
        task: str,
        context: dict,
        failed: list[dict],
    ) -> dict[str, Any]:
        """Ticket booking failed → search for schedule info + give options."""
        search_text = await self._web_search(task)
        memory_text = self._search_memory(task)

        failed_methods = ", ".join(
            a.get("worker", "?") for a in failed
        ) if failed else "自動查票"

        msg = await self._llm_generate(
            f"用戶想要: {task}\n"
            f"自動查票失敗了（{failed_methods} 都試過了）。\n"
            f"搜尋到的資訊: {search_text or '無'}\n"
            f"記憶: {memory_text or '無'}\n\n"
            f"請用 JARVIS 的語氣（稱呼 Sir）回覆：\n"
            f"1. 你查到的班次/票價/時間資訊\n"
            f"2. 給 2-3 個選項讓用戶選擇怎麼處理\n"
            f"3. 語氣自然，結論先行"
        )

        return {
            "result": msg,
            "is_partial": True,
            "source": "assist",
            "worker": self.name,
        }

    # ── Generic assist ───────────────────────────────────────────

    async def _assist_generic(
        self,
        task: str,
        failed: list[dict],
    ) -> dict[str, Any]:
        """Generic task failed → provide info + options."""
        search_text = await self._web_search(task)
        memory_text = self._search_memory(task)

        failed_methods = ", ".join(
            a.get("worker", "?") for a in failed
        ) if failed else "自動處理"

        msg = await self._llm_generate(
            f"用戶要求: {task}\n"
            f"已嘗試: {failed_methods}，都失敗了。\n"
            f"搜尋到的資訊: {search_text or '無'}\n"
            f"記憶: {memory_text or '無'}\n\n"
            f"用 JARVIS 的語氣回覆：\n"
            f"1. 承認自動完成失敗，但不只是道歉\n"
            f"2. 提供你知道的相關資訊\n"
            f"3. 給 2-3 個具體可行的選項\n"
            f"4. 選項要是你能接著處理的（不是丟回去叫用戶自己來）"
        )

        return {
            "result": msg,
            "is_partial": True,
            "source": "assist",
            "worker": self.name,
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _extract_restaurant(self, task: str) -> str:
        """Extract restaurant name from task description."""
        cleaned = re.sub(
            r"幫我訂|訂位|預約|預定|明天|今天|後天|晚上|中午|早上|下午"
            r"|\d+點|\d+個人|\d+位",
            "", task,
        ).strip()
        return cleaned or task[:20]

    async def _web_search(self, query: str) -> str:
        """Quick DuckDuckGo search via httpx."""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query[:80])}"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_SEARCH_TIMEOUT),
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    # Extract text snippets from DuckDuckGo HTML
                    text = resp.text
                    snippets = re.findall(
                        r'class="result__snippet"[^>]*>(.*?)</a>',
                        text, re.DOTALL,
                    )
                    if snippets:
                        clean = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:5]]
                        return "\n".join(clean)
                    # Fallback: strip all HTML
                    return re.sub(r"<[^>]+>", " ", text)[:2000]
        except Exception as e:
            logger.debug(f"AssistWorker web search failed: {e}")
        return ""

    def _search_memory(self, query: str) -> str:
        """Search memory for relevant context."""
        if not self.memory_search:
            return ""
        try:
            results = self.memory_search.search(query, top_k=3)
            if results:
                return "\n".join(r["text"][:200] for r in results)
        except Exception as e:
            logger.debug(f"AssistWorker memory search failed: {e}")
        return ""

    async def _extract_info(
        self, restaurant: str, search_text: str, memory_text: str,
    ) -> dict[str, str | None]:
        """Use LLM to extract structured restaurant info from search results."""
        if not self.router:
            return {}

        prompt = (
            f"從以下搜尋結果中提取餐廳資訊，用 JSON 回覆：\n"
            f"搜尋結果: {search_text[:1500] if search_text else '無'}\n"
            f"記憶: {memory_text[:500] if memory_text else '無'}\n"
            f"餐廳名稱: {restaurant}\n\n"
            f"提取: phone, address, hours, booking_url\n"
            f"找不到的欄位填 null。只回覆 JSON，不要其他文字。"
        )

        try:
            from clients.base_client import ChatMessage
            from core.model_router import ModelRole

            response = await self.router.chat(
                [ChatMessage(role="user", content=prompt)],
                role=ModelRole.CEO,
                max_tokens=200,
                temperature=0.1,
            )
            # Parse JSON from response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = re.sub(r"```\w*\n?", "", content).strip()
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"AssistWorker info extraction failed: {e}")
            return {}

    async def _llm_generate(self, prompt: str) -> str:
        """Generate text via LLM."""
        if not self.router:
            return "Sir，系統暫時無法處理。請稍後再試。"

        try:
            from clients.base_client import ChatMessage
            from core.model_router import ModelRole

            response = await self.router.chat(
                [ChatMessage(role="user", content=prompt)],
                role=ModelRole.CEO,
                max_tokens=500,
            )
            return response.content
        except Exception as e:
            logger.warning(f"AssistWorker LLM generate failed: {e}")
            return "Sir，系統暫時無法處理。請稍後再試。"

    async def close(self) -> None:
        """Cleanup (no persistent resources)."""
        pass
