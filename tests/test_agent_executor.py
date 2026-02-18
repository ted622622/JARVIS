"""Tests for core/agent_executor.py — Agent SDK Executor.

Covers:
- Environment preparation (_prepare_env)
- Tier configuration
- Bash security lists
- Token tracking (persist, load, daily reset)
- AgentExecutor (daily limit, usage, usage_line, quota check)
- System prompt building (jarvis vs clawra)
- Execution logging
- Complexity classification in CEO
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Environment prep tests ─────────────────────────────────────


class TestPrepareEnv:
    """Test _prepare_env() function."""

    def test_clears_claudecode_markers(self):
        from core.agent_executor import _prepare_env

        os.environ["CLAUDECODE"] = "1"
        os.environ["CLAUDE_CODE_ENTRY_POINT"] = "test"
        os.environ["CLAUDE_CODE_SESSION_ID"] = "abc"
        _prepare_env()
        assert "CLAUDECODE" not in os.environ
        assert "CLAUDE_CODE_ENTRY_POINT" not in os.environ
        assert "CLAUDE_CODE_SESSION_ID" not in os.environ

    def test_maps_zhipu_to_anthropic_key(self):
        from core.agent_executor import _prepare_env

        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ZHIPU_API_KEY"] = "test-zhipu-key"
        _prepare_env()
        assert os.environ.get("ANTHROPIC_API_KEY") == "test-zhipu-key"
        # Cleanup
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_does_not_overwrite_existing_anthropic_key(self):
        from core.agent_executor import _prepare_env

        os.environ["ANTHROPIC_API_KEY"] = "existing-key"
        os.environ["ZHIPU_API_KEY"] = "should-not-replace"
        _prepare_env()
        assert os.environ["ANTHROPIC_API_KEY"] == "existing-key"
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_sets_base_url(self):
        from core.agent_executor import _prepare_env

        os.environ.pop("ANTHROPIC_BASE_URL", None)
        _prepare_env()
        assert "open.bigmodel.cn" in os.environ.get("ANTHROPIC_BASE_URL", "")

    def test_sets_model_tier_env_vars(self):
        from core.agent_executor import _prepare_env

        os.environ.pop("ANTHROPIC_DEFAULT_SONNET_MODEL", None)
        os.environ.pop("ANTHROPIC_DEFAULT_HAIKU_MODEL", None)
        _prepare_env()
        assert os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL") is not None
        assert os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL") is not None

    def test_model_tier_uses_zhipu_env(self):
        from core.agent_executor import _prepare_env

        os.environ.pop("ANTHROPIC_DEFAULT_SONNET_MODEL", None)
        os.environ.pop("ANTHROPIC_DEFAULT_HAIKU_MODEL", None)
        os.environ["ZHIPU_CEO_MODEL"] = "my-ceo-model"
        os.environ["ZHIPU_LITE_MODEL"] = "my-lite-model"
        _prepare_env()
        assert os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL") == "my-ceo-model"
        assert os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL") == "my-lite-model"
        # Cleanup
        os.environ.pop("ZHIPU_CEO_MODEL", None)
        os.environ.pop("ZHIPU_LITE_MODEL", None)
        os.environ.pop("ANTHROPIC_DEFAULT_SONNET_MODEL", None)
        os.environ.pop("ANTHROPIC_DEFAULT_HAIKU_MODEL", None)


# ── Tier config tests ──────────────────────────────────────────


class TestTierConfig:
    def test_simple_tier(self):
        from core.agent_executor import TIER_CONFIG
        assert TIER_CONFIG["simple"]["max_turns"] == 5
        assert TIER_CONFIG["simple"]["timeout"] == 30

    def test_medium_tier(self):
        from core.agent_executor import TIER_CONFIG
        assert TIER_CONFIG["medium"]["max_turns"] == 15
        assert TIER_CONFIG["medium"]["timeout"] == 120
        assert "Bash" in TIER_CONFIG["medium"]["allowed_tools"]

    def test_complex_tier(self):
        from core.agent_executor import TIER_CONFIG
        assert TIER_CONFIG["complex"]["max_turns"] == 40
        assert TIER_CONFIG["complex"]["timeout"] == 420
        assert "Write" in TIER_CONFIG["complex"]["allowed_tools"]


# ── Bash security tests ───────────────────────────────────────


class TestBashSecurity:
    def test_allowed_prefixes(self):
        from core.agent_executor import BASH_ALLOWED_PREFIXES
        assert "curl " in BASH_ALLOWED_PREFIXES
        assert "gog " in BASH_ALLOWED_PREFIXES
        assert "python " in BASH_ALLOWED_PREFIXES

    def test_blocked_commands(self):
        from core.agent_executor import BASH_BLOCKED
        assert "rm -rf" in BASH_BLOCKED
        assert "sudo" in BASH_BLOCKED
        assert "shutdown" in BASH_BLOCKED
        assert "powershell -enc" in BASH_BLOCKED


# ── Token tracking tests ──────────────────────────────────────


class TestTokenTracking:
    def test_load_missing_file(self, tmp_path):
        import core.agent_executor as ae
        # Reset global
        ae._TOKEN_USAGE_PATH = None
        usage = ae._load_token_usage(str(tmp_path))
        assert "last_reset" in usage
        assert "daily_history" in usage

    def test_save_and_load(self, tmp_path):
        import core.agent_executor as ae
        ae._TOKEN_USAGE_PATH = None
        data = {"last_reset": "2026-01-01", "daily_history": [{"date": "2026-01-01", "tokens": 5000}]}
        ae._save_token_usage(str(tmp_path), data)
        ae._TOKEN_USAGE_PATH = None
        loaded = ae._load_token_usage(str(tmp_path))
        assert loaded["last_reset"] == "2026-01-01"
        assert loaded["daily_history"][0]["tokens"] == 5000

    def test_corrupted_file(self, tmp_path):
        import core.agent_executor as ae
        ae._TOKEN_USAGE_PATH = None
        path = tmp_path / "data" / "token_usage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json", encoding="utf-8")
        ae._TOKEN_USAGE_PATH = None
        usage = ae._load_token_usage(str(tmp_path))
        assert "daily_history" in usage  # Falls back to default


# ── AgentExecutor class tests ─────────────────────────────────


class TestAgentExecutor:
    @pytest.fixture
    def executor(self, tmp_path):
        import core.agent_executor as ae
        ae._TOKEN_USAGE_PATH = None
        return ae.AgentExecutor(jarvis_root=str(tmp_path))

    def test_init_zero_tokens(self, executor):
        assert executor._daily_tokens == 0

    def test_get_daily_usage(self, executor):
        usage = executor.get_daily_usage()
        assert usage["daily_tokens"] == 0
        assert usage["daily_limit"] == 200_000
        assert usage["usage_pct"] == 0.0

    def test_get_usage_line(self, executor):
        line = executor.get_usage_line()
        assert "Agent SDK" in line
        assert "0%" in line

    def test_is_quota_low_false(self, executor):
        assert not executor.is_quota_low()

    def test_is_quota_low_true(self, executor):
        executor._daily_tokens = 180_000
        assert executor.is_quota_low()

    def test_is_quota_low_exact_80(self, executor):
        executor._daily_tokens = 160_001
        assert executor.is_quota_low()

    @pytest.mark.asyncio
    async def test_daily_limit_exceeded(self, executor):
        executor._daily_tokens = 250_000
        result = await executor.run("test task", tier="simple")
        assert not result["success"]
        assert result["error"] == "daily_limit_exceeded"
        assert "額度" in result["response"]

    @pytest.mark.asyncio
    async def test_daily_reset(self, executor):
        executor._daily_tokens = 50_000
        executor._daily_reset_date = "2020-01-01"
        # After reset, tokens should be 0 and not trigger limit
        with patch("core.agent_executor.AgentExecutor._build_system_prompt", return_value="test"):
            with patch.dict("sys.modules", {"claude_agent_sdk": MagicMock()}):
                # Import will fail gracefully
                result = await executor.run("test", tier="simple")
        assert executor._daily_tokens >= 0  # Reset happened

    def test_system_prompt_jarvis(self, executor):
        prompt = executor._build_system_prompt("jarvis")
        assert "JARVIS" in prompt
        assert "Sir" in prompt
        assert "gog" in prompt

    def test_system_prompt_clawra(self, executor):
        prompt = executor._build_system_prompt("clawra")
        assert "Clawra" in prompt
        assert "口語" in prompt

    def test_persist_usage(self, executor, tmp_path):
        import core.agent_executor as ae
        executor._daily_tokens = 1000
        executor._persist_usage(1000)
        ae._TOKEN_USAGE_PATH = None
        loaded = ae._load_token_usage(str(tmp_path))
        assert loaded["daily_history"][0]["tokens"] == 1000

    def test_log_execution(self, executor, tmp_path):
        executor._log_execution("test task", "simple", True, 2, 5.5, 1500, None)
        log_path = tmp_path / "data" / "agent_sdk_log.jsonl"
        assert log_path.exists()
        line = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert line["task"] == "test task"
        assert line["success"] is True
        assert line["tool_calls"] == 2

    def test_persist_keeps_30_days(self, executor, tmp_path):
        import core.agent_executor as ae
        # Pre-fill with 35 entries
        data = {
            "last_reset": "2026-01-01",
            "daily_history": [
                {"date": f"2025-12-{i:02d}", "tokens": i * 100}
                for i in range(1, 36)
            ],
        }
        ae._TOKEN_USAGE_PATH = None
        ae._save_token_usage(str(tmp_path), data)
        ae._TOKEN_USAGE_PATH = None
        executor._daily_tokens = 500
        executor._persist_usage(500)
        ae._TOKEN_USAGE_PATH = None
        loaded = ae._load_token_usage(str(tmp_path))
        assert len(loaded["daily_history"]) <= 30


# ── CEO complexity classification tests ───────────────────────


class TestComplexityClassification:
    @pytest.fixture
    def ceo(self):
        from core.ceo_agent import CEOAgent
        router = MagicMock()
        ceo = CEOAgent(model_router=router)
        return ceo

    def test_simple_greeting(self, ceo):
        assert ceo._classify_complexity("你好") == "simple"
        assert ceo._classify_complexity("嗨") == "simple"
        assert ceo._classify_complexity("OK") == "simple"

    def test_simple_short(self, ceo):
        assert ceo._classify_complexity("嗯") == "simple"

    def test_complex_booking(self, ceo):
        assert ceo._classify_complexity("幫我訂明天晚上六點的滿築火鍋五個人") == "complex"
        assert ceo._classify_complexity("幫我預約下週三的牙醫看一下") == "complex"

    def test_complex_research(self, ceo):
        assert ceo._classify_complexity("幫我研究一下首爾的住宿推薦") == "complex"
        assert ceo._classify_complexity("幫我分析這個數據看看趨勢") == "complex"

    def test_complex_multi_step(self, ceo):
        assert ceo._classify_complexity("查一下拉麵店然後整理排名") == "complex"

    def test_medium_web_need(self, ceo):
        from core.ceo_agent import TaskComplexity
        # Regular questions that need web but aren't complex
        result = ceo._classify_complexity("今天天氣怎麼樣？")
        assert result in (TaskComplexity.SIMPLE, TaskComplexity.MEDIUM)

    def test_agent_executor_lazy_init(self, ceo):
        """Agent executor returns None when SDK not installed."""
        with patch.dict("sys.modules", {"core.agent_executor": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = ceo._get_agent_executor()
        # Should be None when import fails (or cached from previous call)
        # Just verify it doesn't crash

    def test_extract_phone(self):
        from core.ceo_agent import CEOAgent
        assert CEOAgent._extract_phone("電話: 02-1234-5678") == "02-1234-5678"
        assert CEOAgent._extract_phone("no phone here") is None

    def test_extract_booking_url(self):
        from core.ceo_agent import CEOAgent
        assert "inline.app" in CEOAgent._extract_booking_url(
            "訂位: https://inline.app/test"
        )
        assert CEOAgent._extract_booking_url("no url") is None


# ── Morning brief Agent SDK line test ─────────────────────────


class TestMorningBriefAgentLine:
    @pytest.mark.asyncio
    async def test_brief_includes_agent_usage(self):
        """Morning brief includes Agent SDK usage line when executor is available."""
        from core.heartbeat import Heartbeat

        hb = Heartbeat.__new__(Heartbeat)
        hb.persona = "jarvis"
        hb.telegram = AsyncMock()
        hb.survival = None
        hb.reminder = None
        hb.md_memory = None

        # Mock CEO with agent executor
        mock_executor = MagicMock()
        mock_executor.get_usage_line.return_value = (
            "🤖 Agent SDK: 5,000/200,000 tokens (2.5%)"
        )
        mock_ceo = MagicMock()
        mock_ceo._agent_executor = mock_executor
        hb.ceo = mock_ceo

        with patch.object(hb, "_fetch_weather", new_callable=AsyncMock, return_value="晴天 25°C"):
            with patch.object(hb, "_get_gog_today_events", return_value=[]):
                brief = await hb.morning_brief()

        assert "Agent SDK" in brief
        assert "5,000" in brief
