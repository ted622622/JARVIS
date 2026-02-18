"""Agent SDK Executor — JARVIS Phase 2 "real Agent mode".

Wraps claude-agent-sdk for complex tasks, letting GLM drive tool-use
loops autonomously (web search, bash, file read/write).

Usage:
    executor = AgentExecutor()
    result = await executor.run(
        task="幫我訂明天 Niku Mura 2 位",
        tier="complex",
        persona="jarvis",
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Environment preparation ─────────────────────────────────────


def _prepare_env() -> None:
    """Set env vars for Agent SDK, clear nested session markers."""
    # Prevent nested Claude Code session error
    for key in ("CLAUDECODE", "CLAUDE_CODE_ENTRY_POINT",
                "CLAUDE_CODE_SESSION_ID"):
        os.environ.pop(key, None)

    # Map ZHIPU_API_KEY → ANTHROPIC_API_KEY (Agent SDK expects this)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        zhipu_key = os.environ.get("ZHIPU_API_KEY", "")
        if zhipu_key:
            os.environ["ANTHROPIC_API_KEY"] = zhipu_key

    if not os.environ.get("ANTHROPIC_BASE_URL"):
        os.environ["ANTHROPIC_BASE_URL"] = (
            "https://open.bigmodel.cn/api/anthropic"
        )

    os.environ.setdefault("API_TIMEOUT_MS", "3000000")

    # Model tier env vars for Agent SDK
    os.environ.setdefault(
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        os.environ.get("ZHIPU_CEO_MODEL", "glm-4.6v"),
    )
    os.environ.setdefault(
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        os.environ.get("ZHIPU_LITE_MODEL", "glm-4.5-air"),
    )

    # Windows UTF-8 safety
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ─── Tier configuration ──────────────────────────────────────────

TIER_CONFIG: dict[str, dict[str, Any]] = {
    "simple": {
        "max_turns": 5,
        "timeout": 30,
        "allowed_tools": ["WebSearch", "WebFetch"],
    },
    "medium": {
        "max_turns": 15,
        "timeout": 120,
        "allowed_tools": ["WebSearch", "WebFetch", "Bash", "Read"],
    },
    "complex": {
        "max_turns": 40,
        "timeout": 420,
        "allowed_tools": [
            "WebSearch", "WebFetch", "Bash", "Read", "Write",
        ],
    },
}

# ─── Bash security ───────────────────────────────────────────────

BASH_ALLOWED_PREFIXES = [
    "gog ",             # Google Workspace CLI
    "curl ",            # HTTP requests
    "python ",          # Python scripts
    "cat ", "ls ",      # Read operations
    "head ", "tail ",   # Read operations
    "grep ", "find ",   # Search
    "echo ",            # Output
    "type ",            # Windows cat
    "dir ",             # Windows ls
]

BASH_BLOCKED = [
    "rm -rf", "del /s", "rmdir /s",
    "sudo", "chmod 777", "chown",
    "format", "mkfs", "fdisk",
    "shutdown", "reboot", "halt",
    "net user", "net localgroup",
    "reg delete", "reg add",
    "powershell -enc",
]

# ─── Token tracking file ────────────────────────────────────────

_TOKEN_USAGE_PATH: Path | None = None


def _get_token_path(root: str) -> Path:
    global _TOKEN_USAGE_PATH
    if _TOKEN_USAGE_PATH is None:
        _TOKEN_USAGE_PATH = Path(root) / "data" / "token_usage.json"
    return _TOKEN_USAGE_PATH


def _load_token_usage(root: str) -> dict[str, Any]:
    path = _get_token_path(root)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "last_reset": time.strftime("%Y-%m-%d"),
        "daily_history": [],
    }


def _save_token_usage(root: str, data: dict[str, Any]) -> None:
    path = _get_token_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Main executor ───────────────────────────────────────────────


class AgentExecutor:
    """Wraps claude-agent-sdk for autonomous tool-use tasks.

    Spawns a Claude Code CLI subprocess (backed by ZhipuAI GLM)
    that can use WebSearch, WebFetch, Bash, Read, Write.
    """

    DAILY_LIMIT = 200_000  # tokens per day

    def __init__(self, jarvis_root: str | None = None):
        _prepare_env()
        self._root = jarvis_root or str(
            Path(__file__).parent.parent.resolve()
        )
        self._daily_tokens = 0
        self._daily_reset_date = time.strftime("%Y-%m-%d")

        # Load persisted usage
        usage = _load_token_usage(self._root)
        if usage.get("last_reset") == self._daily_reset_date:
            # Sum today's estimated tokens from history
            for entry in usage.get("daily_history", []):
                if entry.get("date") == self._daily_reset_date:
                    self._daily_tokens = entry.get("tokens", 0)
                    break

    async def run(
        self,
        task: str,
        tier: str = "medium",
        persona: str = "jarvis",
        extra_context: str = "",
    ) -> dict[str, Any]:
        """Execute a task via Agent SDK.

        Returns:
            {
                "success": bool,
                "response": str,
                "tool_calls": int,
                "duration": float,
                "error": str | None,
            }
        """
        # Daily reset check
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_tokens = 0
            self._daily_reset_date = today

        if self._daily_tokens > self.DAILY_LIMIT:
            return {
                "success": False,
                "response": "今日 Agent 額度已用完，明天再試。",
                "tool_calls": 0,
                "duration": 0,
                "error": "daily_limit_exceeded",
            }

        config = TIER_CONFIG.get(tier, TIER_CONFIG["medium"])
        system = self._build_system_prompt(persona)

        full_prompt = task
        if extra_context:
            full_prompt = f"{extra_context}\n\n用戶請求：{task}"

        start = time.time()
        response_text = ""
        tool_count = 0
        error = None

        try:
            from claude_agent_sdk import (
                query,
                ClaudeAgentOptions,
                AssistantMessage,
                ResultMessage,
                TextBlock,
                ToolUseBlock,
            )

            options = ClaudeAgentOptions(
                allowed_tools=config["allowed_tools"],
                permission_mode="bypassPermissions",
                system_prompt=system,
                max_turns=config["max_turns"],
                cwd=self._root,
            )

            async def _run() -> None:
                nonlocal response_text, tool_count
                async for msg in query(prompt=full_prompt, options=options):
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                response_text += block.text
                            elif isinstance(block, ToolUseBlock):
                                tool_count += 1
                    elif isinstance(msg, ResultMessage):
                        if hasattr(msg, "result") and msg.result:
                            response_text += str(msg.result)

            await asyncio.wait_for(_run(), timeout=config["timeout"])

        except asyncio.TimeoutError:
            error = f"超時（{config['timeout']}秒）"
            if not response_text:
                response_text = (
                    "Sir，這個任務比預期複雜，"
                    "我目前找到的資訊如下，但還沒完全整理好。"
                )

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:300]}"
            logger.error(f"Agent SDK error: {error}")
            if not response_text:
                response_text = ""

        duration = time.time() - start

        # Estimate tokens (rough: 1.5 token/char + 500/tool call)
        est_tokens = int(len(response_text) * 1.5) + (tool_count * 500)
        self._daily_tokens += est_tokens

        # Persist usage
        self._persist_usage(est_tokens)

        # Log to JSONL
        self._log_execution(task, tier, bool(response_text.strip()),
                            tool_count, duration, est_tokens, error)

        logger.info(
            f"Agent SDK done: tier={tier}, tools={tool_count}, "
            f"time={duration:.1f}s, est_tokens={est_tokens}"
        )

        return {
            "success": bool(response_text.strip()),
            "response": response_text.strip(),
            "tool_calls": tool_count,
            "duration": round(duration, 1),
            "error": error,
        }

    def get_daily_usage(self) -> dict[str, Any]:
        """Return current daily token usage for monitoring."""
        return {
            "daily_tokens": self._daily_tokens,
            "daily_limit": self.DAILY_LIMIT,
            "usage_pct": round(self._daily_tokens / self.DAILY_LIMIT * 100, 1),
            "date": self._daily_reset_date,
        }

    def get_usage_line(self) -> str:
        """One-line summary for morning brief / status report."""
        usage = self.get_daily_usage()
        return (
            f"🤖 Agent SDK: {usage['daily_tokens']:,}/{usage['daily_limit']:,} "
            f"tokens ({usage['usage_pct']}%)"
        )

    def is_quota_low(self) -> bool:
        """True when daily usage exceeds 80%."""
        return self._daily_tokens > self.DAILY_LIMIT * 0.8

    def _persist_usage(self, new_tokens: int) -> None:
        """Save token usage to data/token_usage.json."""
        try:
            usage = _load_token_usage(self._root)
            today = self._daily_reset_date
            usage["last_reset"] = today

            # Update or add today's entry
            history = usage.get("daily_history", [])
            found = False
            for entry in history:
                if entry.get("date") == today:
                    entry["tokens"] = self._daily_tokens
                    found = True
                    break
            if not found:
                history.append({"date": today, "tokens": self._daily_tokens})

            # Keep last 30 days
            if len(history) > 30:
                history = history[-30:]
            usage["daily_history"] = history

            _save_token_usage(self._root, usage)
        except Exception as e:
            logger.debug(f"Failed to persist token usage: {e}")

    def _log_execution(
        self, task: str, tier: str, success: bool,
        tool_calls: int, duration: float, est_tokens: int,
        error: str | None,
    ) -> None:
        """Append execution log to data/agent_sdk_log.jsonl."""
        try:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task": task[:100],
                "tier": tier,
                "success": success,
                "tool_calls": tool_calls,
                "duration": round(duration, 1),
                "est_tokens": est_tokens,
                "daily_total": self._daily_tokens,
                "error": error,
            }
            log_path = Path(self._root) / "data" / "agent_sdk_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log Agent SDK execution: {e}")

    def _build_system_prompt(self, persona: str) -> str:
        """Build system prompt for the Agent SDK session."""
        if persona == "clawra":
            return (
                "你是 Clawra，Ted 的女朋友。"
                "用繁體中文、台灣口語回覆。"
                "簡短自然，像在傳訊息。"
                "如果需要查資訊，直接查，不用問他。"
            )

        return (
            "你是 JARVIS，Ted 的 AI 管家。\n"
            "稱呼他為 Sir。用繁體中文回覆。\n"
            "你的工作是高效解決問題。\n\n"
            "可用工具：\n"
            "- Bash: 可以執行 gog（Google Workspace CLI）、"
            "curl、python 等指令\n"
            "  - gog 路徑: ./bin/gog.exe\n"
            "  - gog calendar events primary --from <iso> --to <iso> --json\n"
            "  - gog calendar create primary --summary '標題' "
            "--from <iso> --to <iso>\n"
            "  - gog gmail search 'newer_than:7d' --max 10 --json\n"
            "- WebSearch: 搜尋網路\n"
            "- WebFetch: 讀取網頁\n"
            "- Read: 讀取檔案\n\n"
            "重要：\n"
            "- 做到 90% 就給選項，不要空手而歸\n"
            "- 找不到完美答案也要給最佳替代方案\n"
            "- 電話號碼和連結要單獨一行，方便點擊\n"
            "- 回覆簡潔，不要廢話\n"
        )
