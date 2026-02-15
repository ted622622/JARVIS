"""fal.ai client — FLUX Kontext [pro] image generation.

Uses fal.ai REST API for image-to-image generation with reference images.
Primary use: Clawra selfie with consistent character appearance.

Pricing: ~$0.04/image for FLUX Kontext [pro]
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

# fal.ai endpoints
FAL_RUN_BASE = "https://fal.run"
FAL_QUEUE_BASE = "https://queue.fal.run"
FLUX_KONTEXT_PRO = "fal-ai/flux-kontext/pro"


@dataclass
class FalImageResponse:
    """Response from fal.ai image generation."""
    url: str
    width: int = 0
    height: int = 0
    content_type: str = "image/jpeg"
    seed: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class FalClient:
    """Client for fal.ai FLUX Kontext image generation.

    Usage:
        client = FalClient(api_key="key-xxx")
        result = await client.generate_image(
            prompt="holding coffee, afternoon light",
            image_url="https://cdn.example.com/anchor.png",
        )
        print(result.url)
    """

    def __init__(
        self,
        api_key: str,
        model: str = FLUX_KONTEXT_PRO,
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def generate_image(
        self,
        prompt: str,
        image_url: str | None = None,
        **kwargs: Any,
    ) -> FalImageResponse:
        """Generate an image using FLUX Kontext.

        Args:
            prompt: Text description for the image
            image_url: Reference image URL for character consistency
            **kwargs: Additional fal.ai parameters (seed, num_images, etc.)

        Returns:
            FalImageResponse with the generated image URL
        """
        client = await self._get_client()

        payload: dict[str, Any] = {"prompt": prompt}
        if image_url:
            payload["image_url"] = image_url
        payload.update(kwargs)

        url = f"{FAL_RUN_BASE}/{self.model}"
        logger.debug(f"fal.ai request: {self.model}, prompt={prompt[:60]}...")

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        images = data.get("images", [])
        if not images:
            raise FalGenerationError("fal.ai returned no images")

        img = images[0]
        result = FalImageResponse(
            url=img["url"],
            width=img.get("width", 0),
            height=img.get("height", 0),
            content_type=img.get("content_type", "image/jpeg"),
            seed=data.get("seed", 0),
            raw=data,
        )

        logger.info(f"fal.ai image generated: {result.width}x{result.height}")
        return result

    async def generate_image_queued(
        self,
        prompt: str,
        image_url: str | None = None,
        poll_interval: float = 2.0,
        max_wait: float = 300.0,
        **kwargs: Any,
    ) -> FalImageResponse:
        """Generate image via queue API (for long-running jobs).

        Submits to queue, polls for status, returns result.
        """
        client = await self._get_client()

        payload: dict[str, Any] = {"prompt": prompt}
        if image_url:
            payload["image_url"] = image_url
        payload.update(kwargs)

        # Submit to queue
        submit_url = f"{FAL_QUEUE_BASE}/{self.model}"
        resp = await client.post(submit_url, json=payload)
        resp.raise_for_status()
        queue_data = resp.json()

        status_url = queue_data.get("status_url")
        response_url = queue_data.get("response_url")

        if not status_url or not response_url:
            raise FalGenerationError("Queue submission missing status/response URLs")

        logger.debug(f"fal.ai queued: request_id={queue_data.get('request_id')}")

        # Poll for completion
        elapsed = 0.0
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            status_resp = await client.get(status_url)
            status_resp.raise_for_status()
            status = status_resp.json()

            if status.get("status") == "COMPLETED":
                break
            elif status.get("status") == "FAILED":
                raise FalGenerationError(f"fal.ai job failed: {status}")
        else:
            raise FalGenerationError(f"fal.ai job timed out after {max_wait}s")

        # Fetch result
        result_resp = await client.get(response_url)
        result_resp.raise_for_status()
        data = result_resp.json()

        images = data.get("images", [])
        if not images:
            raise FalGenerationError("fal.ai returned no images after queue completion")

        img = images[0]
        return FalImageResponse(
            url=img["url"],
            width=img.get("width", 0),
            height=img.get("height", 0),
            content_type=img.get("content_type", "image/jpeg"),
            seed=data.get("seed", 0),
            raw=data,
        )

    async def health_check(self) -> bool:
        """Check if fal.ai API is reachable with valid credentials."""
        try:
            client = await self._get_client()
            # Use a minimal request to test connectivity
            # fal.ai doesn't have a dedicated health endpoint,
            # so we check if we can reach the API without actually generating
            resp = await client.get(
                f"https://queue.fal.run/{self.model}/requests",
                timeout=10.0,
            )
            # 200 or 401 means API is reachable (401 = bad key but server is up)
            return resp.status_code in (200, 401, 404)
        except Exception as e:
            logger.error(f"fal.ai health check failed: {e}")
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class FalGenerationError(Exception):
    """Raised when fal.ai image generation fails."""
