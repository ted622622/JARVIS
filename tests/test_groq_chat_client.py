"""Tests for GroqChatClient.

Run: pytest tests/test_groq_chat_client.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.base_client import ChatMessage, ChatResponse
from clients.groq_chat_client import GroqAPIError, GroqChatClient


class TestGroqChatClient:
    def _make_client(self, model=None) -> GroqChatClient:
        return GroqChatClient(api_key="gsk_test-key", model=model)

    @pytest.mark.asyncio
    async def test_successful_chat(self):
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "pong"}}],
            "model": "llama-3.3-70b-versatile",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.chat([ChatMessage(role="user", content="ping")])
            assert result.content == "pong"
            assert result.model == "llama-3.3-70b-versatile"
            assert result.usage["total_tokens"] == 6

        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        """429 should trigger exponential backoff and retry."""
        client = self._make_client()
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Rate limited"

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {
            "choices": [{"message": {"content": "ok after retry"}}],
            "model": "llama-3.3-70b-versatile",
            "usage": {},
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=[mock_429, mock_200])
            mock_get.return_value = mock_http

            with patch("clients.groq_chat_client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.chat([ChatMessage(role="user", content="hi")])
                assert result.content == "ok after retry"

        await client.close()

    @pytest.mark.asyncio
    async def test_fail_on_4xx(self):
        """Non-429 4xx should raise immediately (no retry)."""
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            with pytest.raises(GroqAPIError, match="401"):
                await client.chat([ChatMessage(role="user", content="hi")])

        await client.close()

    @pytest.mark.asyncio
    async def test_fail_on_500_after_retries(self):
        """Repeated 500 should exhaust retries and raise."""
        client = self._make_client()
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.text = "Internal Server Error"

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_500)
            mock_get.return_value = mock_http

            with patch("clients.groq_chat_client.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(GroqAPIError, match="Failed after 4 attempts"):
                    await client.chat([ChatMessage(role="user", content="hi")])

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "p"}}],
            "model": "llama-3.3-70b-versatile",
            "usage": {},
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.health_check()
            assert result is True

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        client = self._make_client()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=Exception("connection refused"))
            mock_get.return_value = mock_http

            with patch("clients.groq_chat_client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.health_check()
                assert result is False

        await client.close()

    def test_headers_include_auth(self):
        client = self._make_client()
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer gsk_test-key"
        assert headers["Content-Type"] == "application/json"

    def test_default_model(self):
        client = self._make_client()
        assert client.model == "llama-3.3-70b-versatile"

    def test_custom_model_override(self):
        client = self._make_client(model="llama-3.1-8b-instant")
        assert client.model == "llama-3.1-8b-instant"

    @pytest.mark.asyncio
    async def test_close(self):
        client = self._make_client()
        # Close should work even with no active client
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_format_message(self):
        msg = ChatMessage(role="user", content="hello")
        result = GroqChatClient._format_msg(msg)
        assert result == {"role": "user", "content": "hello"}
