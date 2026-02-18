"""Tests for Patch P: CEO Long-content chunking + Telegram split."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from clients.base_client import ChatMessage, ChatResponse
from core.ceo_agent import (
    CEOAgent,
    _CHUNK_SIZE,
    _LONG_CONTENT_THRESHOLD,
    _LONG_WEB_THRESHOLD,
    _STRUCTURED_THRESHOLD,
    _ANALYSIS_KEYWORDS,
    _STRUCTURED_MARKERS,
    _TASK_TEMPLATE_PATTERN,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_router():
    router = AsyncMock()
    router.chat.return_value = ChatResponse(
        content="Test response", model="test-model", usage={}
    )
    return router


@pytest.fixture
def ceo(mock_router):
    return CEOAgent(model_router=mock_router)


# ── _split_long_content ────────────────────────────────────────


class TestSplitLongContent:
    def test_split_long_content_short(self, ceo):
        """Text shorter than _CHUNK_SIZE returns a single chunk."""
        text = "短文本" * 100  # 300 chars
        chunks = ceo._split_long_content(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_content_basic(self, ceo):
        """Text >3000 chars is split into multiple chunks."""
        text = "a" * 7000
        chunks = ceo._split_long_content(text)
        assert len(chunks) >= 2
        # Rejoined text should match original
        assert "".join(chunks) == text

    def test_split_long_content_sentence_boundary(self, ceo):
        """Chunks prefer to split at sentence boundaries (。)."""
        # Build text with sentence markers
        sentence = "這是一個句子。"  # 7 chars
        # Repeat enough to exceed _CHUNK_SIZE
        count = (_CHUNK_SIZE // len(sentence)) + 10
        text = sentence * count
        chunks = ceo._split_long_content(text)
        assert len(chunks) >= 2
        # First chunk should end at a 。 boundary
        assert chunks[0].endswith("。")
        # All text preserved
        assert "".join(chunks) == text


# ── _handle_long_content ───────────────────────────────────────


class TestHandleLongContent:
    @pytest.mark.asyncio
    async def test_handle_long_content_two_stage(self, ceo, mock_router):
        """Stage 1 uses task_type='template', Stage 2 uses CEO model."""
        text = "x" * 7000  # Will produce multiple chunks
        calls = []

        async def track_chat(messages, *, role=None, max_tokens=None, task_type=None):
            calls.append({"task_type": task_type, "max_tokens": max_tokens})
            return ChatResponse(content=f"summary_{len(calls)}", model="test", usage={})

        mock_router.chat.side_effect = track_chat

        result = await ceo._handle_long_content(text, "整理重點", "jarvis")

        # At least 2 Stage-1 calls (template) + 1 Stage-2 call (None = CEO)
        stage1_calls = [c for c in calls if c["task_type"] == "template"]
        stage2_calls = [c for c in calls if c["task_type"] is None]
        assert len(stage1_calls) >= 2, f"Expected >=2 Stage-1 calls, got {len(stage1_calls)}"
        assert len(stage2_calls) == 1, f"Expected 1 Stage-2 call, got {len(stage2_calls)}"
        # Stage 1 uses small max_tokens
        assert all(c["max_tokens"] == 800 for c in stage1_calls)
        # Stage 2 uses full max_tokens
        assert stage2_calls[0]["max_tokens"] == 4096
        assert result.startswith("summary_")


# ── Trigger logic in _process_message ──────────────────────────


class TestLongContentTrigger:
    @pytest.fixture
    def ceo_with_deps(self, mock_router):
        """CEO with enough mocks for _process_message to reach trigger logic."""
        ceo = CEOAgent(model_router=mock_router)
        ceo.emotion = AsyncMock()
        ceo.emotion.classify = AsyncMock(return_value="normal")
        ceo.memos = AsyncMock()
        ceo.memos.log_message = AsyncMock()
        ceo.memos.get_conversation = AsyncMock(return_value=[])
        return ceo

    @pytest.mark.asyncio
    async def test_trigger_user_msg_long_with_keyword(self, ceo_with_deps, mock_router):
        """User message >2000 chars with analysis keyword triggers chunking."""
        long_msg = "請幫我整理以下內容：\n" + "重要資訊。" * 500  # >2000 chars
        assert len(long_msg) > _LONG_CONTENT_THRESHOLD
        assert _ANALYSIS_KEYWORDS.search(long_msg[:500])

        with patch.object(ceo_with_deps, "_handle_long_content", new_callable=AsyncMock) as mock_hlc:
            mock_hlc.return_value = "chunked_reply"
            with patch.object(ceo_with_deps, "_try_skill_match", new_callable=AsyncMock) as mock_skill:
                mock_skill.return_value = None
                # Skip Agent SDK dispatch so long content handler is reached
                with patch.object(ceo_with_deps, "_get_agent_executor", return_value=None):
                    result = await ceo_with_deps._process_message(
                        long_msg, "jarvis", "jarvis_default", None, False
                    )
        mock_hlc.assert_called_once()
        assert result == "chunked_reply"

    @pytest.mark.asyncio
    async def test_trigger_web_ctx_long(self, ceo_with_deps, mock_router):
        """Web context >5000 chars triggers chunking."""
        user_msg = "查一下這個網頁"
        web_ctx = "x" * 6000
        context = {"網路搜尋結果": web_ctx}

        with patch.object(ceo_with_deps, "_handle_long_content", new_callable=AsyncMock) as mock_hlc:
            mock_hlc.return_value = "web_chunked_reply"
            with patch.object(ceo_with_deps, "_try_skill_match", new_callable=AsyncMock) as mock_skill:
                mock_skill.return_value = None
                with patch.object(ceo_with_deps, "_proactive_web_search", new_callable=AsyncMock) as mock_ws:
                    mock_ws.return_value = None
                    result = await ceo_with_deps._process_message(
                        user_msg, "jarvis", "jarvis_default", context, False
                    )
        mock_hlc.assert_called_once()
        assert result == "web_chunked_reply"

    @pytest.mark.asyncio
    async def test_no_trigger_short_msg(self, ceo_with_deps, mock_router):
        """Short message without analysis keywords does NOT trigger chunking."""
        short_msg = "今天天氣如何？"
        mock_router.chat.return_value = ChatResponse(
            content="天氣很好", model="test", usage={}
        )

        with patch.object(ceo_with_deps, "_handle_long_content", new_callable=AsyncMock) as mock_hlc:
            with patch.object(ceo_with_deps, "_try_skill_match", new_callable=AsyncMock) as mock_skill:
                mock_skill.return_value = None
                with patch.object(ceo_with_deps, "_proactive_web_search", new_callable=AsyncMock) as mock_ws:
                    mock_ws.return_value = None
                    result = await ceo_with_deps._process_message(
                        short_msg, "jarvis", "jarvis_default", None, False
                    )
        mock_hlc.assert_not_called()
        assert result == "天氣很好"


# ── Telegram _send_long_text ───────────────────────────────────


class TestTelegramSendLongText:
    @pytest.mark.asyncio
    async def test_telegram_split_long_message(self):
        """Messages >4000 chars are split into multiple reply_text calls."""
        from clients.telegram_client import TelegramClient

        client = TelegramClient()
        update = MagicMock()
        update.message.reply_text = AsyncMock()

        long_text = "x" * 9000  # Will need 3 chunks (4000+4000+1000)
        await client._send_long_text(update, long_text)

        assert update.message.reply_text.call_count == 3
        # Verify all text is covered
        sent = "".join(call.args[0] for call in update.message.reply_text.call_args_list)
        assert sent == long_text

    @pytest.mark.asyncio
    async def test_telegram_short_message_no_split(self):
        """Messages <=4000 chars are sent as a single reply_text call."""
        from clients.telegram_client import TelegramClient

        client = TelegramClient()
        update = MagicMock()
        update.message.reply_text = AsyncMock()

        short_text = "Hello, world!"
        await client._send_long_text(update, short_text)

        update.message.reply_text.assert_called_once_with(short_text)


# ── Structured content detection ───────────────────────────────


class TestStructuredContentDetection:
    def test_structured_markers_match_markdown(self):
        """Markdown with headers/blockquotes/separators triggers structured detection."""
        md = (
            "# Title\n\n"
            "> Source info\n\n"
            "---\n\n"
            "## Section 1\n"
            "Content here.\n\n"
            "### Section 1.1\n"
            "More content.\n"
        )
        markers = _STRUCTURED_MARKERS.findall(md)
        assert len(markers) >= 3

    def test_plain_text_no_markers(self):
        """Plain text without MD markers does not match."""
        text = "今天天氣如何？我想出門走走。"
        markers = _STRUCTURED_MARKERS.findall(text)
        assert len(markers) < 3

    @pytest.fixture
    def ceo_with_deps(self, mock_router):
        ceo = CEOAgent(model_router=mock_router)
        ceo.emotion = AsyncMock()
        ceo.emotion.classify = AsyncMock(return_value="normal")
        ceo.memos = AsyncMock()
        ceo.memos.log_message = AsyncMock()
        ceo.memos.get_conversation = AsyncMock(return_value=[])
        return ceo

    @pytest.mark.asyncio
    async def test_trigger_structured_md(self, ceo_with_deps, mock_router):
        """Structured MD >500 chars with >=3 markers triggers chunking."""
        md_msg = (
            "# Clawra 開源設定\n\n"
            "> 來源：SumeLabs/clawra\n\n"
            "---\n\n"
            "## Part 1\n"
            "設定內容在此。" * 50 + "\n\n"
            "## Part 2\n"
            "更多內容。" * 50 + "\n"
        )
        assert len(md_msg) > _STRUCTURED_THRESHOLD
        assert len(_STRUCTURED_MARKERS.findall(md_msg)) >= 3

        with patch.object(ceo_with_deps, "_handle_long_content", new_callable=AsyncMock) as mock_hlc:
            mock_hlc.return_value = "structured_reply"
            with patch.object(ceo_with_deps, "_try_skill_match", new_callable=AsyncMock) as mock_skill:
                mock_skill.return_value = None
                result = await ceo_with_deps._process_message(
                    md_msg, "jarvis", "jarvis_default", None, False
                )
        mock_hlc.assert_called_once()
        assert result == "structured_reply"

    @pytest.mark.asyncio
    async def test_no_trigger_short_md(self, ceo_with_deps, mock_router):
        """Short MD (< 500 chars) does NOT trigger even with markers."""
        short_md = "# Title\n\n> Quote\n\n---\n\nHello"
        assert len(short_md) < _STRUCTURED_THRESHOLD
        mock_router.chat.return_value = ChatResponse(
            content="normal reply", model="test", usage={}
        )

        with patch.object(ceo_with_deps, "_handle_long_content", new_callable=AsyncMock) as mock_hlc:
            with patch.object(ceo_with_deps, "_try_skill_match", new_callable=AsyncMock) as mock_skill:
                mock_skill.return_value = None
                with patch.object(ceo_with_deps, "_proactive_web_search", new_callable=AsyncMock) as mock_ws:
                    mock_ws.return_value = None
                    result = await ceo_with_deps._process_message(
                        short_md, "jarvis", "jarvis_default", None, False
                    )
        mock_hlc.assert_not_called()
        assert result == "normal reply"

    @pytest.mark.asyncio
    async def test_no_trigger_task_template(self, ceo_with_deps, mock_router):
        """MD with （完整內容） placeholders is a task template — skips chunking."""
        template_msg = (
            "# Clawra 開源設定完整參考\n\n"
            "> 來源：SumeLabs/clawra\n\n"
            "---\n\n"
            "## Part 1 — 基礎人格\n"
            "（完整內容）\n\n"
            "## Part 2 — 對話風格\n"
            "（完整內容）\n\n"
            "## Part 3 — 記憶設定\n"
            "（完整內容）\n\n"
            "## Part 4 — 進階\n"
            "（完整內容）\n\n"
            "## Part 5 — 附錄\n"
            "（完整內容）\n"
        )
        # Has MD markers + >500 chars, but is a task template
        assert len(_STRUCTURED_MARKERS.findall(template_msg)) >= 3
        assert _TASK_TEMPLATE_PATTERN.search(template_msg)

        mock_router.chat.return_value = ChatResponse(
            content="我來幫您查看這些設定。[FETCH:https://raw.githubusercontent.com/...]",
            model="test", usage={},
        )

        with patch.object(ceo_with_deps, "_handle_long_content", new_callable=AsyncMock) as mock_hlc:
            with patch.object(ceo_with_deps, "_try_skill_match", new_callable=AsyncMock) as mock_skill:
                mock_skill.return_value = None
                with patch.object(ceo_with_deps, "_proactive_web_search", new_callable=AsyncMock) as mock_ws:
                    mock_ws.return_value = None
                    with patch.object(ceo_with_deps, "_execute_tool_call", new_callable=AsyncMock) as mock_tool:
                        mock_tool.return_value = "file content here"
                        result = await ceo_with_deps._process_message(
                            template_msg, "jarvis", "jarvis_default", None, False
                        )
        # Should NOT go through chunking
        mock_hlc.assert_not_called()


class TestFetchGithubRepos:
    @pytest.mark.asyncio
    async def test_fetch_github_repos_extracts_and_fetches(self, mock_router):
        """Detects owner/repo patterns and fetches them."""
        ceo = CEOAgent(model_router=mock_router)
        msg = (
            "# Clawra 設定\n"
            "> 來源：SumeLabs/clawra, clawra-dev/clawra-anime\n"
            "（完整內容）\n"
        )
        with patch.object(ceo, "_execute_tool_call", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = "# README\n" + "Content here. " * 30  # >200 chars
            result = await ceo._fetch_github_repos(msg)
        assert mock_fetch.call_count == 2
        # Verify URLs constructed correctly
        calls = [c.args[0] for c in mock_fetch.call_args_list]
        assert "https://github.com/SumeLabs/clawra" in calls
        assert "https://github.com/clawra-dev/clawra-anime" in calls
        assert result is not None
        assert "SumeLabs/clawra" in result

    @pytest.mark.asyncio
    async def test_fetch_github_repos_no_repos(self, mock_router):
        """No repo patterns → returns None."""
        ceo = CEOAgent(model_router=mock_router)
        result = await ceo._fetch_github_repos("今天天氣如何？（完整內容）")
        assert result is None


class TestTaskTemplatePattern:
    def test_matches_full_content_placeholder(self):
        """（完整內容） matches task template pattern."""
        assert _TASK_TEMPLATE_PATTERN.search("## Part 1\n（完整內容）")

    def test_matches_fill_in_placeholder(self):
        """（填入設定） matches task template pattern."""
        assert _TASK_TEMPLATE_PATTERN.search("（填入設定）")

    def test_matches_mustache_template(self):
        """{{variable}} matches task template pattern."""
        assert _TASK_TEMPLATE_PATTERN.search("Name: {{user_name}}")

    def test_no_match_plain_text(self):
        """Plain text does not match template pattern."""
        assert not _TASK_TEMPLATE_PATTERN.search("今天天氣如何？")

    def test_no_match_analysis_text(self):
        """Analysis text with keywords does not match template pattern."""
        assert not _TASK_TEMPLATE_PATTERN.search("請幫我整理以下內容：重要資訊。")


# ── Conversation history poison filter ─────────────────────────


class TestHistoryPoisonFilter:
    @pytest.mark.asyncio
    async def test_filters_poisoned_assistant_replies(self):
        """Assistant replies with '無法克隆' are filtered from history."""
        mock_router = AsyncMock()
        mock_router.chat.return_value = ChatResponse(
            content="Clean reply", model="test", usage={}
        )
        ceo = CEOAgent(model_router=mock_router)
        ceo.memos = AsyncMock()
        ceo.memos.get_conversation = AsyncMock(return_value=[
            {"role": "user", "content": "處理這個 MD"},
            {"role": "assistant", "content": "我無法克隆 GitHub 仓库"},
            {"role": "user", "content": "你可以的"},
            {"role": "assistant", "content": "我确实无法访问文件系统"},
        ])

        messages = await ceo._build_messages("system prompt", "新問題")

        # Should have: system + "處理這個 MD" + "你可以的" + "新問題" = 4
        # Poisoned assistant replies should be filtered out
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 0
        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) == 3  # 2 from history + 1 new
