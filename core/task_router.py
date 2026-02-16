"""Task Router — regex-based task classification (no LLM cost).

Classifies user messages into task types and determines whether the
CEO LLM is needed or if the task can be dispatched directly to a worker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RoutedTask:
    """A classified sub-task ready for dispatch."""

    task_type: str
    worker: str | None  # None means CEO handles it
    needs_llm: bool
    text: str  # original user text
    depends_on: str | None = None


# ── Pattern registry ────────────────────────────────────────────

_TASK_PATTERNS: list[dict] = [
    {
        "type": "weather",
        "patterns": [r"天氣|氣溫|下雨|帶傘|溫度|weather|降雨"],
        "worker": "weather",
        "needs_llm": False,
    },
    {
        "type": "calendar",
        "patterns": [r"行程|會議|日曆|calendar|幾點.*(?:開|有)|schedule|待辦"],
        "worker": "gog",
        "needs_llm": False,
    },
    {
        "type": "email",
        "patterns": [r"email|信箱|寄信|收信|gmail|郵件|mail"],
        "worker": "gog",
        "needs_llm": False,
    },
    {
        "type": "selfie",
        "patterns": [r"自拍|照片|穿搭|selfie|拍.*?照"],
        "worker": "selfie",
        "needs_llm": False,
    },
    {
        "type": "voice",
        "patterns": [r"語音|說一下|用說的|唸.*給我聽"],
        "worker": "voice",
        "needs_llm": False,
    },
    {
        "type": "restaurant_booking",
        "patterns": [
            r"訂位|預約.*(?:餐廳|火鍋|飯|店)|幫我.*訂.*(?:位|餐|火鍋|飯|店)",
            r"幫我訂|預定.*(?:餐廳|火鍋|飯|店)",
        ],
        "worker": "browser",
        "needs_llm": True,
    },
    {
        "type": "web_search",
        "patterns": [
            r"幫我[找看查搜]|查一下|搜尋|搜索|搜一下",
            r"上網.*?(?:查|看|搜|找)",
            r"(?:今天|今日|現在|目前|最新|最近).*?(?:新聞|消息|行情|價格|報導)",
            r"(?:股價|匯率|比特幣|bitcoin|btc|eth).*?(?:多少|現在|今天)?",
            r"多少錢|哪裡買|怎麼去|幾點.*?(?:開|關|營業)",
        ],
        "worker": "browser",
        "needs_llm": True,
    },
    {
        "type": "web_browse",
        "patterns": [r"https?://\S+", r"打開.*網站|開啟.*網頁"],
        "worker": "browser",
        "needs_llm": True,
    },
    {
        "type": "code",
        "patterns": [r"寫.*程式|寫.*code|debug|修.*bug|跑.*腳本|執行.*script"],
        "worker": "code",
        "needs_llm": True,
    },
]

# Compiled patterns for performance
_COMPILED: list[tuple[dict, list[re.Pattern]]] = [
    (cfg, [re.compile(p, re.IGNORECASE) for p in cfg["patterns"]])
    for cfg in _TASK_PATTERNS
]


class TaskRouter:
    """Classify user messages into sub-tasks using regex patterns.

    No LLM tokens are consumed — pure Python string matching.
    """

    def classify(self, message: str) -> list[RoutedTask]:
        """Return a list of matched task types for *message*.

        If multiple patterns match, multiple RoutedTask objects are
        returned (e.g. "查天氣順便看日曆" → [weather, calendar]).
        If nothing matches, a single ``conversation`` task is returned.
        """
        tasks: list[RoutedTask] = []
        seen_types: set[str] = set()

        for cfg, patterns in _COMPILED:
            for pat in patterns:
                if pat.search(message) and cfg["type"] not in seen_types:
                    seen_types.add(cfg["type"])
                    tasks.append(RoutedTask(
                        task_type=cfg["type"],
                        worker=cfg["worker"],
                        needs_llm=cfg["needs_llm"],
                        text=message,
                    ))
                    break  # one match per task type is enough

        if not tasks:
            tasks.append(RoutedTask(
                task_type="conversation",
                worker=None,
                needs_llm=True,
                text=message,
            ))

        return tasks

    @staticmethod
    def build_ceo_context(
        tasks: list[RoutedTask],
        results: list[dict],
    ) -> str:
        """Summarise worker results for the CEO prompt (compact)."""
        parts: list[str] = []
        for task, result in zip(tasks, results):
            if result.get("success") or result.get("status") == "ok":
                summary = result.get("summary") or result.get("content", "")
                parts.append(f"[{task.task_type}] {str(summary)[:200]}")
            else:
                err = result.get("error", "未知錯誤")
                parts.append(f"[{task.task_type}] 失敗: {err}")
        return "\n".join(parts)
