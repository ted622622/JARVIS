"""Tests for FalClient (fal.ai FLUX Kontext)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from clients.fal_client import FalClient, FalGenerationError, FalImageResponse


# ── FalImageResponse ──────────────────────────────────────────────


class TestFalImageResponse:
    def test_create(self):
        resp = FalImageResponse(url="https://fal.ai/img/abc.jpg", width=1024, height=1024)
        assert resp.url == "https://fal.ai/img/abc.jpg"
        assert resp.width == 1024

    def test_defaults(self):
        resp = FalImageResponse(url="test")
        assert resp.width == 0
        assert resp.content_type == "image/jpeg"


# ── FalClient ─────────────────────────────────────────────────────


class TestFalClient:
    def test_init(self):
        client = FalClient(api_key="key-test")
        assert client.api_key == "key-test"
        assert "kontext" in client.model

    def test_custom_model(self):
        client = FalClient(api_key="key", model="fal-ai/flux-schnell")
        assert client.model == "fal-ai/flux-schnell"

    @pytest.mark.asyncio
    async def test_generate_image_success(self):
        client = FalClient(api_key="key-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "images": [
                {"url": "https://fal.ai/result.jpg", "width": 1024, "height": 1024, "content_type": "image/jpeg"}
            ],
            "seed": 42,
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.generate_image("test prompt", image_url="https://ref.jpg")

        assert isinstance(result, FalImageResponse)
        assert result.url == "https://fal.ai/result.jpg"
        assert result.width == 1024
        assert result.seed == 42

        # Verify correct payload
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["prompt"] == "test prompt"
        assert payload["image_url"] == "https://ref.jpg"

    @pytest.mark.asyncio
    async def test_generate_image_no_images(self):
        client = FalClient(api_key="key-test")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"images": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False
        client._client = mock_client

        with pytest.raises(FalGenerationError, match="no images"):
            await client.generate_image("test")

    @pytest.mark.asyncio
    async def test_generate_without_reference_image(self):
        client = FalClient(api_key="key-test")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "images": [{"url": "https://fal.ai/r.jpg", "width": 512, "height": 512}],
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.generate_image("prompt only, no ref")
        assert result.url == "https://fal.ai/r.jpg"

        payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert "image_url" not in payload

    @pytest.mark.asyncio
    async def test_health_check_ok(self):
        client = FalClient(api_key="key-test")

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.is_closed = False
        client._client = mock_client

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_fail(self):
        client = FalClient(api_key="key-test")

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("timeout")
        mock_client.is_closed = False
        client._client = mock_client

        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_close(self):
        client = FalClient(api_key="key-test")
        mock_client = AsyncMock()
        mock_client.is_closed = False
        client._client = mock_client

        await client.close()
        mock_client.aclose.assert_called_once()
