"""Voice Worker — TTS for J.A.R.V.I.S. and Clawra.

Primary: Azure Speech (SSML, natural prosody)
Fallback: edge-tts (free, if Azure key unavailable)

Each persona has a distinct voice, speaking rate, and style.
"""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

try:
    import edge_tts

    _HAS_EDGE_TTS = True
except ImportError:
    _HAS_EDGE_TTS = False

# Voice configuration per persona
VOICE_MAP: dict[str, str] = {
    "clawra": "zh-TW-HsiaoChenNeural",
    "jarvis": "zh-TW-YunJheNeural",
}

VOICE_RATE: dict[str, str] = {
    "clawra": "+0%",
    "jarvis": "+5%",
}

# Azure Speech SSML style
VOICE_STYLE: dict[str, str] = {
    "clawra": "chat",
    "jarvis": "chat",
}

DEFAULT_CACHE_DIR = "./data/voice_cache"
AZURE_OUTPUT_FORMAT = "audio-16khz-128kbitrate-mono-mp3"


class VoiceWorker:
    """Worker for text-to-speech generation.

    Usage:
        worker = VoiceWorker(azure_key="xxx", azure_region="eastasia")
        path = await worker.text_to_speech("Hello!", persona="jarvis")
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        azure_key: str = "",
        azure_region: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = "voice"
        self.azure_key = azure_key
        self.azure_region = azure_region
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._http_client

    def _cache_path(self, text: str, persona: str) -> Path:
        """Generate a deterministic cache path for a text+persona combo."""
        key = f"{persona}:{text}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{h}.mp3"

    @staticmethod
    def _insert_breaks(escaped_text: str) -> str:
        """Insert SSML <break> tags at Chinese punctuation for natural pauses."""
        # Long pause after sentence-ending punctuation
        text = re.sub(r'([。！？])', r'\1<break time="350ms"/>', escaped_text)
        # Short pause after comma / semicolon / colon
        text = re.sub(r'([，；：、])', r'\1<break time="180ms"/>', text)
        # Medium pause after ellipsis
        text = re.sub(r'(……|⋯⋯|\.\.\.)', r'\1<break time="400ms"/>', text)
        return text

    def _build_ssml(self, text: str, persona: str) -> str:
        """Build Azure Speech SSML with natural prosody and breaks."""
        voice = VOICE_MAP.get(persona, VOICE_MAP["jarvis"])
        rate = VOICE_RATE.get(persona, VOICE_RATE["jarvis"])
        style = VOICE_STYLE.get(persona, "general")
        safe_text = html.escape(text)
        # Insert natural pauses at punctuation
        safe_text = self._insert_breaks(safe_text)

        # Use mstts:express-as for conversational style
        return (
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='zh-TW'>"
            f"<voice name='{voice}'>"
            f"<mstts:express-as style='{style}'>"
            f"<prosody rate='{rate}' pitch='+0%'>{safe_text}</prosody>"
            "</mstts:express-as>"
            "</voice></speak>"
        )

    async def _azure_tts(self, text: str, persona: str, out_path: Path) -> bool:
        """Generate speech via Azure Speech REST API. Returns True on success."""
        if not self.azure_key or not self.azure_region:
            return False

        url = (
            f"https://{self.azure_region}.tts.speech.microsoft.com"
            "/cognitiveservices/v1"
        )
        ssml = self._build_ssml(text, persona)

        client = await self._get_http_client()
        try:
            resp = await client.post(
                url,
                content=ssml.encode("utf-8"),
                headers={
                    "Ocp-Apim-Subscription-Key": self.azure_key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": AZURE_OUTPUT_FORMAT,
                },
            )

            if resp.status_code != 200:
                logger.warning(
                    f"Azure TTS failed ({resp.status_code}): {resp.text[:100]}"
                )
                return False

            out_path.write_bytes(resp.content)
            size = out_path.stat().st_size
            if size == 0:
                out_path.unlink(missing_ok=True)
                return False

            voice = VOICE_MAP.get(persona, "?")
            logger.info(
                f"Azure TTS generated: {out_path.name} ({size} bytes, "
                f"voice={voice})"
            )
            return True

        except Exception as e:
            logger.warning(f"Azure TTS error: {e}")
            out_path.unlink(missing_ok=True)
            return False

    async def _edge_tts(self, text: str, persona: str, out_path: Path) -> bool:
        """Generate speech via edge-tts (fallback). Returns True on success."""
        if not _HAS_EDGE_TTS:
            return False

        voice = VOICE_MAP.get(persona, VOICE_MAP["jarvis"])
        rate = VOICE_RATE.get(persona, VOICE_RATE["jarvis"])

        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
            await communicate.save(str(out_path))

            size = out_path.stat().st_size
            if size == 0:
                out_path.unlink(missing_ok=True)
                return False

            logger.info(
                f"edge-tts generated: {out_path.name} ({size} bytes, "
                f"voice={voice}, rate={rate})"
            )
            return True

        except Exception as e:
            logger.warning(f"edge-tts error: {e}")
            out_path.unlink(missing_ok=True)
            return False

    async def text_to_speech(
        self,
        text: str,
        persona: str = "jarvis",
    ) -> str:
        """Generate MP3 audio from text.

        Tries Azure Speech first, falls back to edge-tts.

        Args:
            text: Text to synthesize
            persona: "jarvis" or "clawra" (determines voice)

        Returns:
            Path to the generated MP3 file

        Raises:
            VoiceError: If all TTS engines fail
        """
        # Clean text for TTS (remove emoji, action words)
        text = VoiceTextCleaner.clean(text)

        if not text.strip():
            raise VoiceError("Empty text provided for TTS")

        out_path = self._cache_path(text, persona)

        # Return cached version if exists and non-empty
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.debug(f"TTS cache hit: {out_path.name}")
            return str(out_path)

        # Try Azure Speech first
        if await self._azure_tts(text, persona, out_path):
            return str(out_path)

        # Fallback to edge-tts
        if await self._edge_tts(text, persona, out_path):
            return str(out_path)

        raise VoiceError("All TTS engines failed")

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """CEO-compatible execute interface."""
        persona = kwargs.get("persona", "jarvis")
        try:
            path = await self.text_to_speech(task, persona=persona)
            return {"worker": self.name, "audio_path": path}
        except Exception as e:
            logger.error(f"VoiceWorker failed: {e}")
            return {"error": str(e), "worker": self.name}

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None


class VoiceTextCleaner:
    """Clean text for TTS — strip emoji, action words, decorative symbols."""

    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U0001F900-\U0001F9FF"  # supplemental
        "\U0001FA00-\U0001FA6F"  # extended symbols
        "\U0001FA70-\U0001FAFF"  # extended-A
        "\U00002702-\U000027B0"  # dingbats
        "\U00002600-\U000026FF"  # misc symbols
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0000200D"             # ZWJ
        "]+",
        flags=re.UNICODE,
    )

    ACTION_PATTERN = re.compile(r"[\(（\*].*?[\)）\*]")
    DECORATIVE_PATTERN = re.compile(r"[~～♪♫★☆♡♥💤✨]+")

    @classmethod
    def clean(cls, text: str) -> str:
        """Remove emoji, action words, and decorative symbols for TTS."""
        text = cls.EMOJI_PATTERN.sub("", text)
        text = cls.ACTION_PATTERN.sub("", text)
        text = cls.DECORATIVE_PATTERN.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class VoiceError(Exception):
    """Raised when TTS generation fails."""
