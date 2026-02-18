"""GLM-TTS Client — wraps official zhipuai SDK for text-to-speech.

Uses the official zai-sdk (zhipuai package) instead of raw httpx.
The SDK handles authentication, retries, and request formatting.

Known issue: GLM-TTS model embeds ~100Hz tone bursts (max ~1794 amp)
in the first ~700ms of WAV output. Callers must trim before playback.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

try:
    from zhipuai import ZhipuAI

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False
    logger.debug("zhipuai SDK not installed, GlmTtsClient unavailable")


class GlmTtsClient:
    """Sync-wrapped GLM-TTS client using official zhipuai SDK.

    Usage:
        client = GlmTtsClient(api_key="xxx")
        wav_bytes = await client.synthesize("你好", voice="tongtong")
    """

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._client: Any = None

    @property
    def is_available(self) -> bool:
        return _HAS_SDK and bool(self._api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            if not _HAS_SDK:
                raise RuntimeError("zhipuai SDK not installed")
            self._client = ZhipuAI(api_key=self._api_key)
        return self._client

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "tongtong",
        response_format: str = "wav",
    ) -> bytes:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize
            voice: Voice name (tongtong, chuichui, xiaochen)
            response_format: Output format (wav recommended for trimming)

        Returns:
            Raw audio bytes (WAV format)

        Raises:
            RuntimeError: If SDK not installed or API key missing
            Exception: On API errors
        """
        if not self.is_available:
            raise RuntimeError("GlmTtsClient not available (missing SDK or API key)")

        client = self._get_client()

        def _call() -> bytes:
            response = client.audio.speech(
                model="glm-tts",
                input=text,
                voice=voice,
                response_format=response_format,
                encode_format=None,  # SDK requires param, None skips it in body
            )
            return response.content

        # SDK is sync — run in thread to avoid blocking the event loop
        return await asyncio.to_thread(_call)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
