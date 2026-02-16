"""Groq Whisper STT client — speech-to-text via Groq API.

Uses httpx directly (not the groq SDK) to avoid Python 3.14 compatibility
issues. Sends audio files to the Groq Whisper endpoint for transcription.

Model: whisper-large-v3-turbo (fast, multilingual)
"""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"


class GroqSTTClient:
    """Speech-to-text client using Groq's Whisper API.

    Usage:
        client = GroqSTTClient(api_key="gsk_xxx")
        text = await client.transcribe("audio.oga", language="zh")
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._client

    async def transcribe(
        self,
        audio_path: str | Path,
        language: str = "zh",
    ) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (.oga, .mp3, .wav, etc.)
            language: Language hint for Whisper (default: zh for Chinese)

        Returns:
            Transcribed text string

        Raises:
            GroqSTTError: If the API call fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise GroqSTTError(f"Audio file not found: {audio_path}")

        client = await self._get_client()

        # Determine MIME type from extension
        mime_map = {
            ".oga": "audio/ogg",
            ".ogg": "audio/ogg",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".webm": "audio/webm",
        }
        mime_type = mime_map.get(audio_path.suffix.lower(), "audio/ogg")

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, mime_type)}
                data = {
                    "model": self.model,
                    "language": language,
                    "response_format": "text",
                }
                response = await client.post(
                    GROQ_TRANSCRIPTION_URL,
                    files=files,
                    data=data,
                )

            if response.status_code != 200:
                error_detail = response.text[:200]
                raise GroqSTTError(
                    f"Groq API error {response.status_code}: {error_detail}"
                )

            text = response.text.strip()
            logger.info(f"Groq STT transcribed ({len(text)} chars): {text[:60]}...")
            return text

        except httpx.HTTPError as e:
            raise GroqSTTError(f"HTTP error during transcription: {e}") from e

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class GroqSTTError(Exception):
    """Raised when Groq STT transcription fails."""
