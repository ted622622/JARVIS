"""Tests for VoiceWorker (TTS), GroqSTTClient (STT), and Telegram voice handler."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.groq_stt_client import GroqSTTClient, GroqSTTError
from workers.voice_worker import (
    VoiceWorker, VoiceError, VoiceTextCleaner,
    VOICE_MAP, VOICE_RATE, VOICE_STYLE, ZHIPU_VOICE_MAP,
)


# ── VoiceWorker ──────────────────────────────────────────────────


class TestVoiceWorkerInit:
    def test_defaults(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))
        assert worker.name == "voice"
        assert worker.cache_dir == tmp_path
        assert worker.azure_key == ""

    def test_with_azure(self, tmp_path):
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            azure_key="testkey",
            azure_region="eastasia",
        )
        assert worker.azure_key == "testkey"
        assert worker.azure_region == "eastasia"

    def test_creates_cache_dir(self, tmp_path):
        cache = tmp_path / "sub" / "cache"
        worker = VoiceWorker(cache_dir=str(cache))
        assert cache.exists()

    def test_voice_map(self):
        assert "clawra" in VOICE_MAP
        assert "jarvis" in VOICE_MAP
        assert "Neural" in VOICE_MAP["clawra"]
        assert "Neural" in VOICE_MAP["jarvis"]

    def test_voice_rate(self):
        assert VOICE_RATE["clawra"] == "+0%"
        assert VOICE_RATE["jarvis"] == "+5%"

    def test_voice_style(self):
        assert VOICE_STYLE["clawra"] == "chat"
        assert VOICE_STYLE["jarvis"] == "chat"


class TestSSML:
    def test_ssml_clawra_has_express_as(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path), azure_key="k", azure_region="r")
        ssml = worker._build_ssml("哈囉", "clawra")
        assert "express-as" in ssml
        assert "style='chat'" in ssml
        assert VOICE_MAP["clawra"] in ssml
        assert "哈囉" in ssml

    def test_ssml_jarvis_has_express_as(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path), azure_key="k", azure_region="r")
        ssml = worker._build_ssml("Hello Sir", "jarvis")
        assert "express-as" in ssml
        assert "style='chat'" in ssml
        assert VOICE_MAP["jarvis"] in ssml
        assert "Hello Sir" in ssml

    def test_ssml_escapes_html(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path), azure_key="k", azure_region="r")
        ssml = worker._build_ssml("A & B <test>", "jarvis")
        assert "&amp;" in ssml
        assert "&lt;" in ssml

    def test_ssml_inserts_breaks(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path), azure_key="k", azure_region="r")
        ssml = worker._build_ssml("你好。我是JARVIS，很高興認識你！", "jarvis")
        assert '<break time="350ms"/>' in ssml  # after 。
        assert '<break time="180ms"/>' in ssml  # after ，
        assert '<break time="350ms"/>' in ssml  # after ！

    def test_insert_breaks_static(self):
        result = VoiceWorker._insert_breaks("甲。乙，丙！丁？")
        assert result.count("<break") == 4


class TestAzureTTS:
    @pytest.mark.asyncio
    async def test_azure_success(self, tmp_path):
        worker = VoiceWorker(
            cache_dir=str(tmp_path), azure_key="testkey", azure_region="eastasia"
        )
        out_path = tmp_path / "test.mp3"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\xff\xfb\x90\x00" * 100

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker._azure_tts("測試", "clawra", out_path)
        assert result is True
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Verify SSML was sent
        call_kwargs = mock_client.post.call_args
        assert "eastasia.tts.speech.microsoft.com" in call_kwargs[0][0]

    @pytest.mark.asyncio
    async def test_azure_no_key(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))
        out_path = tmp_path / "test.mp3"
        result = await worker._azure_tts("test", "jarvis", out_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_azure_api_error(self, tmp_path):
        worker = VoiceWorker(
            cache_dir=str(tmp_path), azure_key="bad", azure_region="eastasia"
        )
        out_path = tmp_path / "test.mp3"

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker._azure_tts("test", "jarvis", out_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_azure_network_error(self, tmp_path):
        worker = VoiceWorker(
            cache_dir=str(tmp_path), azure_key="k", azure_region="eastasia"
        )
        out_path = tmp_path / "test.mp3"

        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("timeout")
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker._azure_tts("test", "jarvis", out_path)
        assert result is False


class TestVoiceWorkerTTS:
    @pytest.mark.asyncio
    async def test_tts_azure_primary(self, tmp_path):
        """Azure is tried first when key is configured."""
        worker = VoiceWorker(
            cache_dir=str(tmp_path), azure_key="k", azure_region="eastasia"
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\xff\xfb\x90\x00" * 100

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        path = await worker.text_to_speech("Hello", persona="jarvis")
        assert Path(path).exists()
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_tts_fallback_to_edge(self, tmp_path):
        """Falls back to edge-tts when Azure fails."""
        worker = VoiceWorker(cache_dir=str(tmp_path))  # no azure key

        mock_communicate = AsyncMock()

        async def fake_save(path):
            Path(path).write_bytes(b"\xff\xfb\x90\x00" * 100)

        mock_communicate.save = fake_save

        with patch("workers.voice_worker.edge_tts") as mock_edge:
            mock_edge.Communicate.return_value = mock_communicate
            path = await worker.text_to_speech("Hello world", persona="jarvis")

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
        mock_edge.Communicate.assert_called_once_with(
            "Hello world", voice=VOICE_MAP["jarvis"], rate=VOICE_RATE["jarvis"],
        )

    @pytest.mark.asyncio
    async def test_tts_cache_hit(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))

        cache_path = worker._cache_path("cached text", "jarvis")
        cache_path.write_bytes(b"\xff\xfb\x90\x00" * 50)

        with patch("workers.voice_worker.edge_tts") as mock_edge:
            path = await worker.text_to_speech("cached text", persona="jarvis")

        mock_edge.Communicate.assert_not_called()
        assert path == str(cache_path)

    @pytest.mark.asyncio
    async def test_tts_empty_text_error(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))

        with pytest.raises(VoiceError, match="Empty text"):
            await worker.text_to_speech("", persona="jarvis")

    @pytest.mark.asyncio
    async def test_tts_all_engines_fail(self, tmp_path):
        """When both Azure and edge-tts fail, raise VoiceError."""
        worker = VoiceWorker(cache_dir=str(tmp_path))  # no azure

        mock_communicate = AsyncMock()

        async def fake_save_empty(path):
            Path(path).write_bytes(b"")

        mock_communicate.save = fake_save_empty

        with patch("workers.voice_worker.edge_tts") as mock_edge:
            mock_edge.Communicate.return_value = mock_communicate
            with pytest.raises(VoiceError, match="All TTS engines failed"):
                await worker.text_to_speech("test", persona="jarvis")

    @pytest.mark.asyncio
    async def test_deterministic_cache_path(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))
        p1 = worker._cache_path("hello", "jarvis")
        p2 = worker._cache_path("hello", "jarvis")
        p3 = worker._cache_path("hello", "clawra")
        assert p1 == p2
        assert p1 != p3


class TestVoiceWorkerExecute:
    @pytest.mark.asyncio
    async def test_execute_success(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))

        mock_communicate = AsyncMock()

        async def fake_save(path):
            Path(path).write_bytes(b"\xff\xfb\x90\x00" * 50)

        mock_communicate.save = fake_save

        with patch("workers.voice_worker.edge_tts") as mock_edge:
            mock_edge.Communicate.return_value = mock_communicate
            result = await worker.execute("Say something", persona="jarvis")

        assert result["worker"] == "voice"
        assert "audio_path" in result

    @pytest.mark.asyncio
    async def test_execute_error(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))

        result = await worker.execute("", persona="jarvis")
        assert "error" in result
        assert result["worker"] == "voice"

    @pytest.mark.asyncio
    async def test_close(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))
        mock_client = AsyncMock()
        mock_client.is_closed = False
        worker._http_client = mock_client

        await worker.close()
        mock_client.aclose.assert_called_once()


# ── GroqSTTClient ────────────────────────────────────────────────


class TestGroqSTTClientInit:
    def test_defaults(self):
        client = GroqSTTClient(api_key="gsk_test")
        assert client.api_key == "gsk_test"
        assert client.model == "whisper-large-v3-turbo"

    def test_custom_model(self):
        client = GroqSTTClient(api_key="gsk_test", model="whisper-large-v3")
        assert client.model == "whisper-large-v3"


class TestGroqSTTTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_success(self, tmp_path):
        client = GroqSTTClient(api_key="gsk_test")

        audio_file = tmp_path / "test.oga"
        audio_file.write_bytes(b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "你好世界"

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.transcribe(str(audio_file))
        assert result == "你好世界"

        call_args = mock_http.post.call_args
        assert call_args[0][0] == "https://api.groq.com/openai/v1/audio/transcriptions"

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self, tmp_path):
        client = GroqSTTClient(api_key="gsk_test")

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Hello world"

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.transcribe(str(audio_file), language="en")
        assert result == "Hello world"

        call_kwargs = mock_http.post.call_args[1]
        assert call_kwargs["data"]["language"] == "en"

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self):
        client = GroqSTTClient(api_key="gsk_test")

        with pytest.raises(GroqSTTError, match="not found"):
            await client.transcribe("/nonexistent/audio.oga")

    @pytest.mark.asyncio
    async def test_transcribe_api_error(self, tmp_path):
        client = GroqSTTClient(api_key="gsk_test")

        audio_file = tmp_path / "test.oga"
        audio_file.write_bytes(b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(GroqSTTError, match="401"):
            await client.transcribe(str(audio_file))

    @pytest.mark.asyncio
    async def test_transcribe_http_error(self, tmp_path):
        import httpx

        client = GroqSTTClient(api_key="gsk_test")

        audio_file = tmp_path / "test.oga"
        audio_file.write_bytes(b"\x00" * 100)

        mock_http = AsyncMock()
        mock_http.post.side_effect = httpx.ConnectError("timeout")
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(GroqSTTError, match="HTTP error"):
            await client.transcribe(str(audio_file))

    @pytest.mark.asyncio
    async def test_close(self):
        client = GroqSTTClient(api_key="gsk_test")
        mock_http = AsyncMock()
        mock_http.is_closed = False
        client._client = mock_http

        await client.close()
        mock_http.aclose.assert_called_once()


# ── Telegram Voice Handler ───────────────────────────────────────


class TestTelegramVoiceHandler:
    """Test the Telegram voice message flow (download → STT → CEO → TTS → reply voice)."""

    def _make_client(self):
        from clients.telegram_client import TelegramClient

        client = TelegramClient(
            jarvis_token="fake-jarvis-token",
            clawra_token="fake-clawra-token",
            chat_id=12345,
        )
        client._token_to_persona = {"fake-jarvis-token": "jarvis"}
        return client

    def _make_update(self, voice_data=True, user_id=None):
        update = MagicMock()
        update.message.chat_id = 12345
        update.message.message_id = 99
        update.message.from_user.id = user_id or 1
        update.message.from_user.first_name = "TestUser"
        update.message.reply_text = AsyncMock()

        if voice_data:
            mock_file = AsyncMock()
            mock_file.download_to_drive = AsyncMock()
            update.message.voice.get_file = AsyncMock(return_value=mock_file)
        else:
            update.message.voice = None

        return update

    def _make_context(self, token="fake-jarvis-token"):
        ctx = MagicMock()
        ctx.bot.token = token
        return ctx

    @pytest.mark.asyncio
    async def test_no_voice_in_update(self):
        client = self._make_client()
        update = self._make_update(voice_data=False)
        context = self._make_context()

        await client._handle_voice_message(update, context)
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_user(self):
        client = self._make_client()
        client._allowed_user_ids = {999}
        update = self._make_update(user_id=1)
        context = self._make_context()

        await client._handle_voice_message(update, context)
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_stt_client_jarvis(self):
        client = self._make_client()
        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)
        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "GROQ_API_KEY" in call_text

    @pytest.mark.asyncio
    async def test_no_stt_client_clawra(self):
        client = self._make_client()
        client._token_to_persona["fake-jarvis-token"] = "clawra"
        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)
        call_text = update.message.reply_text.call_args[0][0]
        assert "聽不懂語音" in call_text

    @pytest.mark.asyncio
    async def test_full_voice_flow(self, tmp_path):
        client = self._make_client()
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = "你好"
        client.stt_client = mock_stt

        # Create fake reply audio file so send_voice can open it
        reply_mp3 = tmp_path / "reply.mp3"
        reply_mp3.write_bytes(b"\xff\xfb\x90\x00" * 50)

        mock_voice = AsyncMock()
        mock_voice.text_to_speech.return_value = str(reply_mp3)
        client.voice_worker = mock_voice

        async def fake_handler(text, chat_id, persona):
            return "你好！有什麼我能幫你的嗎？"

        client.set_message_handler(fake_handler)

        mock_bot = MagicMock()
        mock_bot.send_voice = AsyncMock(return_value=MagicMock(message_id=100))
        client._jarvis_bot = mock_bot

        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)

        # STT was called
        mock_stt.transcribe.assert_called_once()

        # Voice reply only — no text reply for CEO response
        update.message.reply_text.assert_not_called()

        # TTS was generated and voice was sent
        mock_voice.text_to_speech.assert_called_once_with(
            "你好！有什麼我能幫你的嗎？", persona="jarvis", emotion=None,
        )
        mock_bot.send_voice.assert_called_once()

    @pytest.mark.asyncio
    async def test_stt_failure_jarvis(self, tmp_path):
        client = self._make_client()
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.side_effect = Exception("STT error")
        client.stt_client = mock_stt

        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)

        calls = update.message.reply_text.call_args_list
        assert any("語音辨識失敗" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_empty_transcription(self, tmp_path):
        client = self._make_client()
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = ""
        client.stt_client = mock_stt

        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)

        calls = update.message.reply_text.call_args_list
        assert any("沒聽到" in str(c) or "沒有聽到" in str(c) or "偵測到" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_tts_failure_falls_back_to_text(self, tmp_path):
        """If TTS fails, the text reply should be sent as fallback."""
        client = self._make_client()
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = "測試"
        client.stt_client = mock_stt

        mock_voice = AsyncMock()
        mock_voice.text_to_speech.side_effect = Exception("TTS failed")
        client.voice_worker = mock_voice

        async def fake_handler(text, chat_id, persona):
            return "回覆"

        client.set_message_handler(fake_handler)

        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)

        calls = update.message.reply_text.call_args_list
        assert any("回覆" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_no_typing_delay_for_voice(self, tmp_path):
        """Voice replies should NOT trigger typing delay."""
        client = self._make_client()
        client._token_to_persona["fake-jarvis-token"] = "clawra"
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = "嗨"
        client.stt_client = mock_stt

        reply_mp3 = tmp_path / "reply.mp3"
        reply_mp3.write_bytes(b"\xff\xfb\x90\x00" * 50)

        mock_voice = AsyncMock()
        mock_voice.text_to_speech.return_value = str(reply_mp3)
        client.voice_worker = mock_voice

        async def fake_handler(text, chat_id, persona):
            return "嗨～"

        client.set_message_handler(fake_handler)

        mock_bot = MagicMock()
        mock_bot.send_voice = AsyncMock(return_value=MagicMock(message_id=100))
        mock_bot.send_chat_action = AsyncMock()
        client._clawra_bot = mock_bot

        update = self._make_update()
        context = self._make_context()

        with patch.object(client, "_simulate_typing") as mock_typing:
            await client._handle_voice_message(update, context)
            mock_typing.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_text_reply_when_user_requests(self, tmp_path):
        """If user says '用文字回我' in voice, should reply with text, not TTS."""
        client = self._make_client()
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = "今天天氣怎樣 用文字回我"
        client.stt_client = mock_stt

        mock_voice = AsyncMock()
        client.voice_worker = mock_voice

        async def fake_handler(text, chat_id, persona):
            return "今天台北 25 度，晴天。"

        client.set_message_handler(fake_handler)

        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)

        # Should use text reply, NOT TTS
        update.message.reply_text.assert_called()
        call_text = update.message.reply_text.call_args[0][0]
        assert "25 度" in call_text

        # TTS should NOT be called
        mock_voice.text_to_speech.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_text_reply_with_typing_keyword(self, tmp_path):
        """'打字回' variant should also trigger text reply."""
        client = self._make_client()
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = "你打字回我好了"
        client.stt_client = mock_stt

        mock_voice = AsyncMock()
        client.voice_worker = mock_voice

        async def fake_handler(text, chat_id, persona):
            return "好的，Sir。"

        client.set_message_handler(fake_handler)

        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)

        update.message.reply_text.assert_called()
        mock_voice.text_to_speech.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_voice_still_uses_tts(self, tmp_path):
        """Normal voice message without text keywords should still use TTS."""
        client = self._make_client()
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = "今天天氣怎樣"
        client.stt_client = mock_stt

        reply_mp3 = tmp_path / "reply.mp3"
        reply_mp3.write_bytes(b"\xff\xfb\x90\x00" * 50)

        mock_voice = AsyncMock()
        mock_voice.text_to_speech.return_value = str(reply_mp3)
        client.voice_worker = mock_voice

        async def fake_handler(text, chat_id, persona):
            return "今天晴天。"

        client.set_message_handler(fake_handler)

        mock_bot = MagicMock()
        mock_bot.send_voice = AsyncMock(return_value=MagicMock(message_id=100))
        client._jarvis_bot = mock_bot

        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)

        # Should use TTS voice
        mock_voice.text_to_speech.assert_called_once()
        mock_bot.send_voice.assert_called_once()

    @pytest.mark.asyncio
    async def test_reply_mode_text_from_ceo(self, tmp_path):
        """CEO can return reply_mode='text' to force text reply."""
        client = self._make_client()
        client._voice_cache_dir = tmp_path

        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = "查資料"
        client.stt_client = mock_stt

        mock_voice = AsyncMock()
        client.voice_worker = mock_voice

        async def fake_handler(text, chat_id, persona):
            return {"text": "查到了：xxx", "reply_mode": "text"}

        client.set_message_handler(fake_handler)

        update = self._make_update()
        context = self._make_context()

        await client._handle_voice_message(update, context)

        update.message.reply_text.assert_called()
        mock_voice.text_to_speech.assert_not_called()


class TestWantsTextReply:
    """Test the _wants_text_reply static method."""

    def test_chinese_keywords(self):
        from clients.telegram_client import TelegramClient
        assert TelegramClient._wants_text_reply("用文字回我") is True
        assert TelegramClient._wants_text_reply("你打字回我好了") is True
        assert TelegramClient._wants_text_reply("不要語音") is True
        assert TelegramClient._wants_text_reply("用打字的回覆") is True
        assert TelegramClient._wants_text_reply("傳文字給我") is True

    def test_normal_text(self):
        from clients.telegram_client import TelegramClient
        assert TelegramClient._wants_text_reply("今天天氣怎樣") is False
        assert TelegramClient._wants_text_reply("幫我查高鐵") is False
        assert TelegramClient._wants_text_reply("") is False


class TestTelegramSendVoice:
    @pytest.mark.asyncio
    async def test_send_voice_success(self, tmp_path):
        from clients.telegram_client import TelegramClient

        client = TelegramClient(jarvis_token="tok", chat_id=123)
        mock_bot = MagicMock()
        mock_bot.send_voice = AsyncMock(return_value=MagicMock(message_id=42))
        client._jarvis_bot = mock_bot

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff\xfb" * 50)

        result = await client.send_voice(str(audio), chat_id=123)
        assert result == 42

    @pytest.mark.asyncio
    async def test_send_voice_no_bot(self):
        from clients.telegram_client import TelegramClient

        client = TelegramClient()
        result = await client.send_voice("audio.mp3")
        assert result is None

    @pytest.mark.asyncio
    async def test_send_voice_error(self, tmp_path):
        from clients.telegram_client import TelegramClient

        client = TelegramClient(jarvis_token="tok", chat_id=123)
        mock_bot = MagicMock()
        mock_bot.send_voice = AsyncMock(side_effect=Exception("send failed"))
        client._jarvis_bot = mock_bot

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff\xfb" * 50)

        result = await client.send_voice(str(audio), chat_id=123)
        assert result is None


class TestTelegramBuildApplicationsVoice:
    def test_voice_handler_registered(self):
        from clients.telegram_client import TelegramClient

        client = TelegramClient(jarvis_token="fake-token", chat_id=123)
        apps = client.build_applications()

        if not apps:
            pytest.skip("python-telegram-bot not installed")

        app = apps[0]
        handlers = app.handlers.get(0, [])
        handler_types = [type(h).__name__ for h in handlers]
        assert handler_types.count("MessageHandler") >= 2  # TEXT + VOICE
        assert "CallbackQueryHandler" in handler_types


# ── Security Gate Browser Access ─────────────────────────────────


class TestSecurityGateBrowser:
    def test_browser_url_allowed(self):
        from core.security_gate import SecurityGate

        gate = SecurityGate()
        assert gate.check_browser_url("https://google.com") == "allow"

    def test_browser_dangerous_url_blocked(self):
        from core.security_gate import SecurityGate

        gate = SecurityGate()
        assert gate.check_browser_url("file:///etc/passwd") == "block"
        assert gate.check_browser_url("javascript:alert(1)") == "block"

    def test_groq_api_whitelisted(self):
        from core.security_gate import SecurityGate

        gate = SecurityGate()
        assert gate.check_api("api.groq.com") == "allow"

    def test_azure_tts_whitelisted(self):
        from core.security_gate import SecurityGate

        gate = SecurityGate()
        assert gate.check_api("eastasia.tts.speech.microsoft.com") == "allow"


# ── Browser Worker Fetch ─────────────────────────────────────────


class TestBrowserWorkerFetch:
    @pytest.mark.asyncio
    async def test_fetch_url_success(self):
        from workers.browser_worker import BrowserWorker

        worker = BrowserWorker()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = "<html><body>Hello</body></html>"
        mock_resp.url = "https://example.com"
        mock_resp.content = b"<html><body>Hello</body></html>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker.fetch_url("https://example.com")
        assert result["status"] == "ok"
        assert "Hello" in result["content"]
        assert result["worker"] == "browser"

    @pytest.mark.asyncio
    async def test_fetch_url_blocked(self):
        from workers.browser_worker import BrowserWorker
        from core.security_gate import SecurityGate

        gate = SecurityGate()
        worker = BrowserWorker(security_gate=gate)
        result = await worker.fetch_url("file:///etc/passwd")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fetch_url_timeout(self):
        import httpx
        from workers.browser_worker import BrowserWorker

        worker = BrowserWorker()
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker.fetch_url("https://slow.example.com")
        assert "error" in result
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_with_url(self):
        from workers.browser_worker import BrowserWorker

        worker = BrowserWorker()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"result": "ok"}'
        mock_resp.url = "https://api.example.com/data"
        mock_resp.content = b'{"result": "ok"}'

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker.execute("get data", url="https://api.example.com/data")
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_without_url(self):
        from workers.browser_worker import BrowserWorker

        worker = BrowserWorker()
        result = await worker.execute("do something")
        assert result["status"] == "ready"
        assert "URL" in result["note"]

    @pytest.mark.asyncio
    async def test_close(self):
        from workers.browser_worker import BrowserWorker

        worker = BrowserWorker()
        mock_client = AsyncMock()
        mock_client.is_closed = False
        worker._http_client = mock_client

        await worker.close()
        mock_client.aclose.assert_called_once()


# ── GLM-TTS (Zhipu) ─────────────────────────────────────────────


class TestZhipuTTS:
    @pytest.mark.asyncio
    async def test_zhipu_tts_success(self, tmp_path):
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            zhipu_key="test-key",
        )
        out_path = tmp_path / "test.mp3"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\xff\xfb\x90\x00" * 100

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker._zhipu_tts("測試語音", "jarvis", out_path)
        assert result is True
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        call_kwargs = mock_client.post.call_args
        assert "bigmodel.cn" in call_kwargs[0][0]
        body = call_kwargs[1]["json"]
        assert body["model"] == "glm-tts"
        assert body["input"] == "測試語音"
        assert body["voice"] == "chuichui"  # JARVIS uses male voice

    @pytest.mark.asyncio
    async def test_zhipu_tts_no_key_skips(self, tmp_path):
        worker = VoiceWorker(cache_dir=str(tmp_path))
        out_path = tmp_path / "test.mp3"
        result = await worker._zhipu_tts("test", "jarvis", out_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_zhipu_tts_api_error(self, tmp_path):
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            zhipu_key="bad-key",
        )
        out_path = tmp_path / "test.mp3"

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker._zhipu_tts("test", "jarvis", out_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_zhipu_tts_network_error(self, tmp_path):
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            zhipu_key="key",
        )
        out_path = tmp_path / "test.mp3"

        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("connection refused")
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker._zhipu_tts("test", "jarvis", out_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_zhipu_tts_empty_response(self, tmp_path):
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            zhipu_key="test-key",
        )
        out_path = tmp_path / "test.mp3"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b""

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker._zhipu_tts("test", "jarvis", out_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_zhipu_failure_azure_fallback(self, tmp_path):
        """GLM-TTS fails → falls back to Azure."""
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            azure_key="azure-key",
            azure_region="eastasia",
            zhipu_key="zhipu-key",
        )

        # GLM-TTS fails
        zhipu_resp = MagicMock()
        zhipu_resp.status_code = 500

        # Azure succeeds
        azure_resp = MagicMock()
        azure_resp.status_code = 200
        azure_resp.content = b"\xff\xfb\x90\x00" * 100

        mock_client = AsyncMock()
        mock_client.post.side_effect = [zhipu_resp, azure_resp]
        mock_client.is_closed = False
        worker._http_client = mock_client

        path = await worker.text_to_speech("Hello", persona="jarvis")
        assert Path(path).exists()
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_zhipu_azure_fail_edge_fallback(self, tmp_path):
        """GLM-TTS + Azure both fail → falls back to edge-tts."""
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            azure_key="azure-key",
            azure_region="eastasia",
            zhipu_key="zhipu-key",
        )

        # Both HTTP calls fail
        fail_resp = MagicMock()
        fail_resp.status_code = 500

        mock_client = AsyncMock()
        mock_client.post.return_value = fail_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        mock_communicate = AsyncMock()

        async def fake_save(path):
            Path(path).write_bytes(b"\xff\xfb\x90\x00" * 100)

        mock_communicate.save = fake_save

        with patch("workers.voice_worker.edge_tts") as mock_edge:
            mock_edge.Communicate.return_value = mock_communicate
            path = await worker.text_to_speech("test", persona="jarvis")

        assert Path(path).exists()

    @pytest.mark.asyncio
    async def test_zhipu_custom_voice_fallback(self, tmp_path):
        """Custom zhipu_voice is used for unknown personas not in ZHIPU_VOICE_MAP."""
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            zhipu_key="test-key",
            zhipu_voice="custom_voice",
        )
        out_path = tmp_path / "test.mp3"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\xff\xfb\x90\x00" * 100

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        await worker._zhipu_tts("test", "unknown_persona", out_path)

        body = mock_client.post.call_args[1]["json"]
        assert body["voice"] == "custom_voice"

    @pytest.mark.asyncio
    async def test_zhipu_per_persona_voice(self, tmp_path):
        """JARVIS uses chuichui (male), Clawra uses douji (female, Patch T+)."""
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            zhipu_key="test-key",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\xff\xfb\x90\x00" * 100

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        # JARVIS → chuichui (male)
        out_j = tmp_path / "j.mp3"
        await worker._zhipu_tts("test", "jarvis", out_j)
        body_j = mock_client.post.call_args[1]["json"]
        assert body_j["voice"] == ZHIPU_VOICE_MAP["jarvis"]
        assert body_j["voice"] == "chuichui"

        # Clawra → tongtong (female)
        out_c = tmp_path / "c.mp3"
        await worker._zhipu_tts("test", "clawra", out_c)
        body_c = mock_client.post.call_args[1]["json"]
        assert body_c["voice"] == ZHIPU_VOICE_MAP["clawra"]
        assert body_c["voice"] == "tongtong"


class TestTTSFallbackChain:
    @pytest.mark.asyncio
    async def test_cache_key_unaffected_by_emotion(self, tmp_path):
        """Cache key should only depend on text+persona, not emotion."""
        worker = VoiceWorker(cache_dir=str(tmp_path))
        p1 = worker._cache_path("same text", "jarvis")
        p2 = worker._cache_path("same text", "jarvis")
        assert p1 == p2

    @pytest.mark.asyncio
    async def test_execute_passes_emotion(self, tmp_path):
        """execute() should forward emotion kwarg to text_to_speech."""
        worker = VoiceWorker(
            cache_dir=str(tmp_path),
            zhipu_key="test-key",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\xff\xfb\x90\x00" * 100

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        worker._http_client = mock_client

        result = await worker.execute("Say hello", persona="jarvis", emotion="happy")
        assert "audio_path" in result


# ── VoiceTextCleaner ─────────────────────────────────────────


class TestVoiceTextCleaner:
    def test_removes_emoji(self):
        text = "早安呀☀️ 今天天氣不錯呢！❤️"
        cleaned = VoiceTextCleaner.clean(text)
        assert "☀" not in cleaned
        assert "❤" not in cleaned
        assert "早安呀" in cleaned

    def test_removes_action_words(self):
        text = "好啊（揮手）明天見*微笑*"
        cleaned = VoiceTextCleaner.clean(text)
        assert "揮手" not in cleaned
        assert "微笑" not in cleaned
        assert "好啊" in cleaned
        assert "明天見" in cleaned

    def test_removes_decorative(self):
        text = "早安～♪ 今天好開心♡"
        cleaned = VoiceTextCleaner.clean(text)
        assert "～" not in cleaned
        assert "♪" not in cleaned
        assert "♡" not in cleaned

    def test_preserves_normal_text(self):
        text = "今天天氣不錯，去公園走走吧"
        cleaned = VoiceTextCleaner.clean(text)
        assert cleaned == text

    def test_full_example(self):
        text = "早安呀～☀️ 今天天氣不錯呢！（揮手）記得帶傘喔 ❤️"
        cleaned = VoiceTextCleaner.clean(text)
        assert "早安呀" in cleaned
        assert "記得帶傘喔" in cleaned
        assert "☀" not in cleaned
        assert "揮手" not in cleaned
        assert "❤" not in cleaned

    def test_empty_after_cleaning(self):
        text = "❤️☀️✨"
        cleaned = VoiceTextCleaner.clean(text)
        assert cleaned == ""


# ── Selfie Worker Dual Mode ──────────────────────────────────


class TestClawraVoice:
    """Clawra voice should be tongtong (female)."""

    def test_clawra_voice_is_tongtong(self):
        assert ZHIPU_VOICE_MAP["clawra"] == "tongtong"

    def test_jarvis_voice_unchanged(self):
        assert ZHIPU_VOICE_MAP["jarvis"] == "chuichui"


class TestSelfieDualMode:
    """Backward-compat: old detect_mode / build_prompt still work."""

    def test_detect_direct_cafe(self):
        from workers.selfie_worker import detect_mode
        assert detect_mode("在咖啡廳拍張自拍") == "direct"

    def test_detect_direct_beach(self):
        from workers.selfie_worker import detect_mode
        assert detect_mode("beach sunset smile") == "direct"

    def test_detect_mirror_outfit(self):
        from workers.selfie_worker import detect_mode
        assert detect_mode("穿紅色洋裝") == "mirror"

    def test_detect_mirror_clothes(self):
        from workers.selfie_worker import detect_mode
        assert detect_mode("today's outfit wearing white") == "mirror"

    def test_detect_default_direct(self):
        from workers.selfie_worker import detect_mode
        assert detect_mode("some random scene") == "direct"

    def test_build_prompt_direct(self):
        from workers.selfie_worker import build_prompt
        prompt = build_prompt("cafe with latte", "direct")
        assert "half-body" in prompt  # Patch T+: direct now maps to medium
        assert "cafe with latte" in prompt

    def test_build_prompt_mirror(self):
        from workers.selfie_worker import build_prompt
        prompt = build_prompt("穿紅色洋裝", "mirror")
        assert "mirror selfie" in prompt
        assert "穿紅色洋裝" in prompt


# ── Soul Multi-file ──────────────────────────────────────────


class TestSoulMultiFile:
    def test_loads_split_files(self, tmp_path):
        from core.soul import Soul

        (tmp_path / "SOUL_JARVIS.md").write_text("# JARVIS\n專業的管家", encoding="utf-8")
        (tmp_path / "SOUL_CLAWRA.md").write_text("# Clawra\n可愛的女友", encoding="utf-8")
        (tmp_path / "USER.md").write_text("- 名字: Ted", encoding="utf-8")
        (tmp_path / "IDENTITY.md").write_text("# Identity", encoding="utf-8")

        soul = Soul(config_dir=str(tmp_path))
        soul.load()
        assert soul.is_loaded

        prompt_j = soul.build_system_prompt("jarvis")
        assert "JARVIS" in prompt_j
        assert "Ted" in prompt_j

        prompt_c = soul.build_system_prompt("clawra")
        assert "Clawra" in prompt_c
        assert "Ted" in prompt_c

    def test_fallback_to_legacy(self, tmp_path):
        from core.soul import Soul

        (tmp_path / "SOUL.md").write_text("# Legacy Soul", encoding="utf-8")

        soul = Soul(soul_path=str(tmp_path / "SOUL.md"))
        soul.load()
        assert soul.is_loaded

        # Should use legacy builders
        prompt = soul.build_system_prompt("jarvis")
        assert "J.A.R.V.I.S." in prompt

    def test_extra_context(self, tmp_path):
        from core.soul import Soul

        (tmp_path / "SOUL_JARVIS.md").write_text("# JARVIS", encoding="utf-8")
        soul = Soul(config_dir=str(tmp_path))
        soul.load()

        prompt = soul.build_system_prompt("jarvis", extra_context="用戶焦慮")
        assert "用戶焦慮" in prompt
        assert "當前上下文" in prompt
