"""Zhipu AI client for GLM-4V (vision) and CogView (image generation)."""

from __future__ import annotations

from typing import Any

from loguru import logger
from zhipuai import ZhipuAI

from .base_client import BaseClient, ChatMessage, ChatResponse


class ZhipuClient(BaseClient):
    """Wraps Zhipu AI APIs: GLM-4V-Flash (vision) and CogView-4 (image).

    Uses the official zhipuai SDK for reliability.
    """

    DEFAULT_VISION_MODEL = "glm-4v-flash"
    DEFAULT_IMAGE_MODEL = "cogview-4-250304"

    def __init__(
        self,
        api_key: str,
        vision_model: str | None = None,
        image_model: str | None = None,
        timeout: float = 60.0,
    ):
        # BaseClient for potential direct HTTP calls
        super().__init__("https://open.bigmodel.cn/api/paas/v4", api_key, timeout)
        self.vision_model = vision_model or self.DEFAULT_VISION_MODEL
        self.image_model = image_model or self.DEFAULT_IMAGE_MODEL
        self._sdk = ZhipuAI(api_key=api_key)

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatResponse:
        """Vision chat using GLM-4V."""
        import asyncio

        target_model = model or self.vision_model
        formatted = [self._format_msg(m) for m in messages]

        # SDK is synchronous, run in executor
        def _call():
            return self._sdk.chat.completions.create(
                model=target_model,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        resp = await asyncio.to_thread(_call)

        return ChatResponse(
            content=resp.choices[0].message.content,
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str = "1024x1024",
        **kwargs: Any,
    ) -> ImageResponse:
        """Generate image using CogView."""
        import asyncio

        target_model = model or self.image_model

        def _call():
            return self._sdk.images.generations(
                model=target_model,
                prompt=prompt,
                size=size,
                **kwargs,
            )

        resp = await asyncio.to_thread(_call)

        return ImageResponse(
            url=resp.data[0].url,
            model=target_model,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    async def vision_analyze(
        self,
        image_url: str,
        prompt: str,
        *,
        model: str | None = None,
    ) -> ChatResponse:
        """Analyze an image with a text prompt using GLM-4V."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            )
        ]
        return await self.chat(messages, model=model)

    @staticmethod
    def _format_msg(msg: ChatMessage) -> dict[str, Any]:
        return {"role": msg.role, "content": msg.content}

    async def health_check(self) -> bool:
        try:
            await self.chat(
                [ChatMessage(role="user", content="ping")],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error(f"ZhipuClient health check failed: {e}")
            return False

    async def close(self) -> None:
        await super().close()


class ImageResponse:
    """Response from image generation."""

    def __init__(self, url: str, model: str, raw: dict[str, Any] | None = None):
        self.url = url
        self.model = model
        self.raw = raw or {}
