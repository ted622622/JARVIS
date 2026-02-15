"""Tests for Soul, EmotionClassifier, and SelfieSkill.

Run: pytest tests/test_soul_emotion.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.base_client import ChatMessage, ChatResponse
from core.emotion import EMOTION_LABELS, EmotionClassifier
from core.soul import CORE_DNA_PROMPT, Soul
from memory.memos_manager import MemOS
from skills.selfie.main import SelfieResult, SelfieSkill


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
async def memos(tmp_path):
    db_path = str(tmp_path / "test.db")
    m = MemOS(db_path)
    await m.init()
    yield m
    await m.close()


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.chat = AsyncMock(return_value=ChatResponse(content="normal", model="kimi"))
    return router


# ── Soul Tests ──────────────────────────────────────────────────


class TestSoul:
    def test_load_from_file(self):
        soul = Soul("./config/SOUL.md")
        soul.load()
        assert soul.is_loaded
        assert soul._raw_content != ""

    def test_load_missing_file(self, tmp_path):
        soul = Soul(str(tmp_path / "nonexistent.md"))
        soul.load()
        assert soul.is_loaded  # Should still mark as loaded with defaults

    def test_jarvis_prompt(self):
        soul = Soul()
        soul.load()
        prompt = soul.build_system_prompt("jarvis")
        assert "J.A.R.V.I.S." in prompt
        assert "100% 誠實" in prompt
        assert "結論先行" in prompt

    def test_clawra_prompt(self):
        soul = Soul()
        soul.load()
        prompt = soul.build_system_prompt("clawra")
        assert "Clawra" in prompt
        assert "100% 誠實" in prompt
        assert "活潑開朗" in prompt

    def test_extra_context_injected(self):
        soul = Soul()
        soul.load()
        prompt = soul.build_system_prompt("jarvis", extra_context="用戶情緒: tired")
        assert "用戶情緒: tired" in prompt
        assert "當前上下文" in prompt

    def test_selfie_prompt_composition(self):
        soul = Soul()
        soul.load()
        prompt = soul.get_selfie_prompt("holding coffee, afternoon light")
        assert CORE_DNA_PROMPT in prompt
        assert "holding coffee" in prompt

    def test_core_dna_prompt_immutable(self):
        assert "Korean girl" in CORE_DNA_PROMPT
        assert "aegyo-sal" in CORE_DNA_PROMPT
        assert "warm" in CORE_DNA_PROMPT


# ── EmotionClassifier Tests ─────────────────────────────────────


class TestEmotionClassifier:
    @pytest.mark.asyncio
    async def test_classify_returns_valid_label(self, mock_router, memos):
        mock_router.chat = AsyncMock(
            return_value=ChatResponse(content="tired", model="kimi")
        )
        classifier = EmotionClassifier(mock_router, memos)
        label = await classifier.classify("我好累喔，今天加班到很晚")
        assert label == "tired"
        assert label in EMOTION_LABELS

    @pytest.mark.asyncio
    async def test_classify_writes_to_memos(self, mock_router, memos):
        mock_router.chat = AsyncMock(
            return_value=ChatResponse(content="happy", model="kimi")
        )
        classifier = EmotionClassifier(mock_router, memos)
        await classifier.classify("太棒了！今天超順利")

        emotion = await memos.working_memory.get("user_emotion")
        assert emotion == "happy"

    @pytest.mark.asyncio
    async def test_classify_defaults_to_normal_on_error(self):
        classifier = EmotionClassifier(model_router=None, memos=None)
        label = await classifier.classify("some message")
        assert label == "normal"

    @pytest.mark.asyncio
    async def test_classify_extracts_from_verbose_response(self, mock_router, memos):
        mock_router.chat = AsyncMock(
            return_value=ChatResponse(content="The emotion is: anxious", model="kimi")
        )
        classifier = EmotionClassifier(mock_router, memos)
        label = await classifier.classify("我很擔心明天的考試")
        assert label == "anxious"

    @pytest.mark.asyncio
    async def test_classify_unknown_defaults_normal(self, mock_router, memos):
        mock_router.chat = AsyncMock(
            return_value=ChatResponse(content="confused_and_lost", model="kimi")
        )
        classifier = EmotionClassifier(mock_router, memos)
        label = await classifier.classify("test")
        assert label == "normal"

    @pytest.mark.asyncio
    async def test_get_current_emotion(self, mock_router, memos):
        await memos.working_memory.set("user_emotion", "excited", agent_id="test")
        classifier = EmotionClassifier(mock_router, memos)
        emotion = await classifier.get_current_emotion()
        assert emotion == "excited"

    @pytest.mark.asyncio
    async def test_get_current_emotion_default(self, mock_router, memos):
        classifier = EmotionClassifier(mock_router, memos)
        emotion = await classifier.get_current_emotion()
        assert emotion == "unknown"

    @pytest.mark.asyncio
    async def test_all_valid_labels(self):
        """Ensure the label set is complete."""
        expected = {"anxious", "tired", "sad", "frustrated", "normal", "happy", "excited"}
        assert set(EMOTION_LABELS) == expected


# ── SelfieSkill Tests ──────────────────────────────────────────


class TestSelfieSkill:
    def test_init(self):
        skill = SelfieSkill(fal_api_key="test-key")
        assert skill._fal_key == "test-key"

    @pytest.mark.asyncio
    async def test_generate_without_any_key_fails(self):
        skill = SelfieSkill(fal_api_key="", google_api_key="")
        result = await skill.generate("test scene")
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_generate_via_fal_success(self):
        skill = SelfieSkill(fal_api_key="test-key")

        from clients.fal_client import FalImageResponse
        fake_response = FalImageResponse(
            url="https://fal.ai/result.jpg", width=1024, height=1024
        )

        skill._generate_via_fal = AsyncMock(return_value=fake_response)
        result = await skill.generate("holding coffee", verify=False)

        assert result.success
        assert result.image_url == "https://fal.ai/result.jpg"
        assert result.width == 1024
        assert result.cost_usd == 0.04
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_consistency_check_retry(self):
        skill = SelfieSkill(fal_api_key="test-key")
        skill.CONSISTENCY_THRESHOLD = 0.7

        from clients.fal_client import FalImageResponse

        call_count = 0
        async def mock_gen(prompt):
            nonlocal call_count
            call_count += 1
            return FalImageResponse(url=f"https://fal.ai/{call_count}.jpg", width=1024, height=1024)

        async def mock_check(url):
            if call_count <= 1:
                return 0.3
            return 0.8

        skill._generate_via_fal = mock_gen
        skill._check_consistency = mock_check
        skill.router = MagicMock()

        result = await skill.generate("test scene", verify=True)
        assert result.success
        assert result.attempts == 2
        assert result.consistency_score == 0.8

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        skill = SelfieSkill(fal_api_key="test-key")
        skill.CONSISTENCY_THRESHOLD = 0.9
        skill.MAX_RETRIES = 1

        from clients.fal_client import FalImageResponse
        fake = FalImageResponse(url="https://fal.ai/x.jpg", width=512, height=512)
        skill._generate_via_fal = AsyncMock(return_value=fake)
        skill._check_consistency = AsyncMock(return_value=0.2)
        skill.router = MagicMock()

        result = await skill.generate("test", verify=True)
        assert not result.success
        assert result.attempts == 2
        assert "too low" in result.error.lower()

    def test_core_dna_in_prompt(self):
        soul = Soul()
        soul.load()
        prompt = soul.get_selfie_prompt("at a cafe")
        assert "Korean girl" in prompt
        assert "aegyo-sal" in prompt
        assert "at a cafe" in prompt


# ── Live Test (requires fal.ai API key) ──────────────────────


@pytest.mark.live
class TestSelfieLive:
    @pytest.mark.asyncio
    async def test_fal_kontext_generation(self):
        """Requires FAL_KEY in .env"""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        key = os.environ.get("FAL_KEY", "")
        if not key or key == "your-fal-key-here":
            pytest.skip("FAL_KEY not set")

        skill = SelfieSkill(fal_api_key=key)
        result = await skill.generate(
            "holding a coffee cup, warm afternoon light, casual white t-shirt",
            verify=False,
        )
        assert result.success
        assert result.image_url is not None
