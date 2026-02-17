"""Tests for TranscribeWorker — long audio ASR + two-stage summary.

Run: pytest tests/test_transcribe_worker.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.base_client import ChatResponse
from workers.transcribe_worker import TranscribeWorker, SUPPORTED_FORMATS, CHUNK_SIZE


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.chat = AsyncMock(
        return_value=ChatResponse(
            content="📋 會議摘要：測試摘要內容\n✅ 決議一\n📌 行動項目",
            model="test",
        )
    )
    return router


@pytest.fixture
def worker(mock_router):
    return TranscribeWorker(model_router=mock_router, zhipu_key="test-key")


@pytest.fixture
def worker_no_key():
    return TranscribeWorker()


@pytest.fixture
def fake_audio(tmp_path):
    """Create a fake audio file for testing."""
    audio_file = tmp_path / "test_meeting.m4a"
    audio_file.write_bytes(b"fake audio content")
    return audio_file


# ── Basic properties ─────────────────────────────────────────────


class TestBasic:
    def test_name(self, worker):
        assert worker.name == "transcribe"

    def test_supported_formats(self):
        assert ".ogg" in SUPPORTED_FORMATS
        assert ".mp3" in SUPPORTED_FORMATS
        assert ".m4a" in SUPPORTED_FORMATS
        assert ".wav" in SUPPORTED_FORMATS
        assert ".flac" in SUPPORTED_FORMATS
        assert ".opus" in SUPPORTED_FORMATS

    @pytest.mark.asyncio
    async def test_close(self, worker):
        await worker.close()  # Should not raise


# ── Execute interface ────────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_delegates_to_process_audio(self, worker):
        with patch.object(worker, "process_audio", new_callable=AsyncMock) as mock:
            mock.return_value = {"result": "summary", "worker": "transcribe"}
            result = await worker.execute("/path/to/audio.m4a", context="週會")
            mock.assert_awaited_once_with("/path/to/audio.m4a", context="週會")

    @pytest.mark.asyncio
    async def test_execute_default_context(self, worker):
        with patch.object(worker, "process_audio", new_callable=AsyncMock) as mock:
            mock.return_value = {"result": "summary", "worker": "transcribe"}
            await worker.execute("/path/to/audio.m4a")
            mock.assert_awaited_once_with("/path/to/audio.m4a", context="")


# ── Process audio ─────────────────────────────────────────────────


class TestProcessAudio:
    @pytest.mark.asyncio
    async def test_file_not_found(self, worker):
        result = await worker.process_audio("/nonexistent/file.m4a")
        assert "error" in result
        assert "不存在" in result["error"]

    @pytest.mark.asyncio
    async def test_unsupported_format(self, worker, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not audio")
        result = await worker.process_audio(str(txt_file))
        assert "error" in result
        assert "不支援" in result["error"]

    @pytest.mark.asyncio
    async def test_no_zhipu_key(self, worker_no_key, fake_audio):
        result = await worker_no_key.process_audio(str(fake_audio))
        assert "error" in result
        assert "ZHIPU_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_transcription(self, worker, fake_audio):
        with patch.object(worker, "_transcribe", new_callable=AsyncMock, return_value="測試逐字稿內容"):
            result = await worker.process_audio(str(fake_audio), context="週會")
        assert "result" in result
        assert result["transcript"] == "測試逐字稿內容"
        assert result["source"] == "transcribe"
        assert result["worker"] == "transcribe"

    @pytest.mark.asyncio
    async def test_transcription_failure(self, worker, fake_audio):
        with patch.object(worker, "_transcribe", new_callable=AsyncMock, return_value=""):
            result = await worker.process_audio(str(fake_audio))
        assert "error" in result
        assert "轉錄失敗" in result["error"]


# ── Split transcript ──────────────────────────────────────────────


class TestSplitTranscript:
    def test_short_text_single_chunk(self, worker):
        chunks = worker._split_transcript("短文")
        assert len(chunks) == 1
        assert chunks[0] == "短文"

    def test_long_text_splits(self, worker):
        long_text = "這是一句話。" * 1000  # ~6000 chars
        chunks = worker._split_transcript(long_text)
        assert len(chunks) >= 2
        # All text should be covered
        joined = "".join(chunks)
        assert len(joined) == len(long_text)

    def test_splits_at_sentence_boundary(self, worker):
        text = "第一句。" * 500 + "第二句。" * 500
        chunks = worker._split_transcript(text)
        # Each chunk except possibly the last should end at a sentence boundary
        for chunk in chunks[:-1]:
            assert chunk.endswith("。") or chunk.endswith(".") or chunk.endswith("\n") or chunk.endswith("，") or chunk.endswith(",")

    def test_exact_chunk_size(self, worker):
        text = "x" * CHUNK_SIZE
        chunks = worker._split_transcript(text)
        assert len(chunks) == 1


# ── Two-stage summary ────────────────────────────────────────────


class TestTwoStageSummary:
    @pytest.mark.asyncio
    async def test_summary_no_router(self):
        worker = TranscribeWorker(zhipu_key="test")
        result = await worker._two_stage_summary("test transcript", "")
        # Falls back to truncated raw text
        assert result == "test transcript"

    @pytest.mark.asyncio
    async def test_summary_with_router(self, worker):
        result = await worker._two_stage_summary("逐字稿內容", "週會")
        assert "會議摘要" in result

    @pytest.mark.asyncio
    async def test_summary_calls_router_twice_for_single_chunk(self, worker):
        """Single chunk: stage 1 (chunk summary) + stage 2 (final merge)."""
        await worker._two_stage_summary("短逐字稿", "")
        assert worker.router.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_summary_calls_router_more_for_long_text(self, worker):
        """Multiple chunks: stage 1 per chunk + stage 2 merge."""
        long_text = "這是逐字稿。" * 1000
        await worker._two_stage_summary(long_text, "test")
        # Should be called once per chunk + once for final merge
        chunks = worker._split_transcript(long_text)
        expected_calls = len(chunks) + 1
        assert worker.router.chat.call_count == expected_calls


# ── LLM generate helper ──────────────────────────────────────────


class TestLLMGenerate:
    @pytest.mark.asyncio
    async def test_generate_no_router(self):
        worker = TranscribeWorker()
        result = await worker._llm_generate("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_success(self, worker):
        result = await worker._llm_generate("summarize this")
        assert "會議摘要" in result

    @pytest.mark.asyncio
    async def test_generate_error(self, mock_router):
        mock_router.chat = AsyncMock(side_effect=Exception("API error"))
        worker = TranscribeWorker(model_router=mock_router, zhipu_key="test")
        result = await worker._llm_generate("test")
        assert result == ""
