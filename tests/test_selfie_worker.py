"""Tests for workers.selfie_worker — Patch T+: Framing system."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from workers.selfie_worker import (
    detect_framing,
    build_framing_prompt,
    detect_mode,
    build_prompt,
    _FRAMING_PATTERNS,
    _FRAMING_PROMPTS,
    _CLOSEUP_EXPRESSIONS,
    SelfieWorker,
)


# ── detect_framing ───────────────────────────────────────────────


class TestDetectFraming:
    """Framing detection from user context."""

    def test_mirror_keywords(self):
        assert detect_framing("鏡子前拍穿搭") == "mirror"
        assert detect_framing("mirror selfie in outfit") == "mirror"
        assert detect_framing("穿搭照") == "mirror"
        assert detect_framing("拍洋裝") == "mirror"

    def test_full_body_keywords(self):
        assert detect_framing("全身照") == "full_body"
        assert detect_framing("全身拍一張") == "full_body"
        assert detect_framing("full body shot") == "full_body"
        assert detect_framing("站著拍") == "full_body"

    def test_closeup_keywords(self):
        assert detect_framing("近照") == "closeup"
        assert detect_framing("close up selfie") == "closeup"
        assert detect_framing("拍臉") == "closeup"
        assert detect_framing("大頭照") == "closeup"

    def test_medium_keywords(self):
        assert detect_framing("在咖啡廳") == "medium"
        assert detect_framing("cafe selfie") == "medium"
        assert detect_framing("拍張照") == "medium"
        assert detect_framing("早安自拍") == "medium"

    def test_default_is_medium(self):
        """No keyword match → default to medium."""
        assert detect_framing("拍一張") == "medium"
        assert detect_framing("some random scene") == "medium"
        assert detect_framing("") == "medium"

    def test_priority_mirror_over_medium(self):
        """Mirror keywords should win over medium keywords."""
        assert detect_framing("穿搭在咖啡廳") == "mirror"

    def test_priority_full_body_over_medium(self):
        assert detect_framing("全身照在公園") == "full_body"


# ── build_framing_prompt ─────────────────────────────────────────


class TestBuildFramingPrompt:
    """Framing-specific prompt generation."""

    def test_mirror_prompt(self):
        prompt = build_framing_prompt("紅色洋裝", "mirror")
        assert "mirror selfie" in prompt
        assert "紅色洋裝" in prompt

    def test_full_body_prompt(self):
        prompt = build_framing_prompt("漢江邊", "full_body")
        assert "full-body" in prompt
        assert "漢江邊" in prompt

    def test_medium_prompt(self):
        prompt = build_framing_prompt("咖啡廳", "medium")
        assert "half-body" in prompt
        assert "咖啡廳" in prompt

    def test_closeup_prompt(self):
        prompt = build_framing_prompt("golden hour", "closeup")
        assert "close-up" in prompt
        assert "golden hour" in prompt
        # Should have an expression inserted
        has_expression = any(e in prompt for e in _CLOSEUP_EXPRESSIONS)
        assert has_expression, f"No expression in closeup prompt: {prompt}"

    def test_unknown_framing_falls_back_to_medium(self):
        prompt = build_framing_prompt("test", "unknown")
        assert "half-body" in prompt

    def test_all_framings_produce_non_empty(self):
        for framing in ("mirror", "full_body", "medium", "closeup"):
            prompt = build_framing_prompt("test scene", framing)
            assert len(prompt) > 20, f"{framing} prompt too short"


# ── Backward compatibility ───────────────────────────────────────


class TestBackwardCompat:
    """detect_mode() and build_prompt() still work."""

    def test_detect_mode_mirror(self):
        assert detect_mode("穿搭照") == "mirror"

    def test_detect_mode_direct(self):
        assert detect_mode("cafe selfie") == "direct"
        assert detect_mode("some scene") == "direct"

    def test_build_prompt_mirror(self):
        prompt = build_prompt("紅洋裝", "mirror")
        assert "mirror selfie" in prompt

    def test_build_prompt_direct(self):
        prompt = build_prompt("cafe", "direct")
        assert "half-body" in prompt  # direct maps to medium


# ── Data integrity ───────────────────────────────────────────────


class TestFramingData:
    """Verify framing data structures."""

    def test_all_four_patterns_exist(self):
        assert set(_FRAMING_PATTERNS.keys()) == {"mirror", "full_body", "closeup", "medium"}

    def test_all_four_prompts_exist(self):
        assert set(_FRAMING_PROMPTS.keys()) == {"mirror", "full_body", "medium", "closeup"}

    def test_closeup_expressions_count(self):
        assert len(_CLOSEUP_EXPRESSIONS) == 10

    def test_prompts_have_scene_placeholder(self):
        for framing, template in _FRAMING_PROMPTS.items():
            assert "{scene}" in template, f"{framing} template missing {{scene}}"


# ── SelfieWorker.execute ─────────────────────────────────────────


class TestSelfieWorkerExecute:
    @pytest.mark.asyncio
    async def test_no_skills_returns_error(self):
        worker = SelfieWorker(skill_registry=None)
        result = await worker.execute("test")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_framing_kwarg_passed(self):
        mock_registry = AsyncMock()
        mock_registry.invoke = AsyncMock(return_value={"image_url": "http://x", "success": True})
        worker = SelfieWorker(skill_registry=mock_registry)

        result = await worker.execute("test", framing="closeup")
        assert result["framing"] == "closeup"
        # invoke should have been called with framing kwarg
        call_kwargs = mock_registry.invoke.call_args
        assert call_kwargs[1]["framing"] == "closeup"

    @pytest.mark.asyncio
    async def test_auto_detect_framing(self):
        mock_registry = AsyncMock()
        mock_registry.invoke = AsyncMock(return_value={"image_url": "http://x", "success": True})
        worker = SelfieWorker(skill_registry=mock_registry)

        result = await worker.execute("鏡子前穿搭")
        assert result["framing"] == "mirror"

    @pytest.mark.asyncio
    async def test_legacy_mode_kwarg(self):
        mock_registry = AsyncMock()
        mock_registry.invoke = AsyncMock(return_value={"image_url": "http://x", "success": True})
        worker = SelfieWorker(skill_registry=mock_registry)

        result = await worker.execute("test", mode="mirror")
        assert result["framing"] == "mirror"
        assert result["mode"] == "mirror"  # backward compat

    @pytest.mark.asyncio
    async def test_result_has_compat_mode(self):
        mock_registry = AsyncMock()
        mock_registry.invoke = AsyncMock(return_value={"image_url": "http://x"})
        worker = SelfieWorker(skill_registry=mock_registry)

        result = await worker.execute("咖啡廳自拍")
        assert result["framing"] == "medium"
        assert result["mode"] == "direct"  # non-mirror → "direct"
