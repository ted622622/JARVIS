"""Voice Worker — TTS for J.A.R.V.I.S. and Clawra.

Primary: 智譜 GLM-TTS (context-aware emotion, GRPO-optimized)
Fallback 1: Azure Speech (SSML, natural prosody)
Fallback 2: edge-tts (free, unlimited)

Each persona has a distinct voice, speaking rate, and style.
"""

from __future__ import annotations

import hashlib
import html
import io
import re
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

try:
    import edge_tts

    _HAS_EDGE_TTS = True
except ImportError:
    _HAS_EDGE_TTS = False

# Ensure static-ffmpeg paths are available (bundles ffmpeg binary)
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    _HAS_FFMPEG = True
except ImportError:
    _HAS_FFMPEG = False

# Voice configuration per persona
VOICE_MAP: dict[str, str] = {
    "clawra": "zh-TW-HsiaoChenNeural",
    "jarvis": "zh-TW-YunJheNeural",
}

# GLM-TTS voice per persona
ZHIPU_VOICE_MAP: dict[str, str] = {
    "clawra": "tongtong",    # 彤彤 — 女聲
    "jarvis": "chuichui",    # 錘錘 — 男聲
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

    Three-tier fallback: GLM-TTS → Azure Speech → edge-tts.

    Usage:
        worker = VoiceWorker(azure_key="xxx", azure_region="eastasia")
        path = await worker.text_to_speech("Hello!", persona="jarvis")
    """

    ZHIPU_TTS_URL = "https://open.bigmodel.cn/api/paas/v4/audio/speech"

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        azure_key: str = "",
        azure_region: str = "",
        zhipu_key: str = "",
        zhipu_voice: str = "tongtong",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = "voice"
        self.azure_key = azure_key
        self.azure_region = azure_region
        self.zhipu_key = zhipu_key
        self.zhipu_voice = zhipu_voice
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._http_client

    def _cache_path(self, text: str, persona: str, ext: str = ".mp3") -> Path:
        """Generate a deterministic cache path for a text+persona combo."""
        key = f"{persona}:{text}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{h}{ext}"

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

    @staticmethod
    def _trim_leading_tone(
        raw_wav: bytes,
        tone_threshold: int = 500,
        scan_limit_ms: int = 800,
    ) -> bytes:
        """Remove GLM-TTS leading tones from WAV audio.

        GLM-TTS embeds multiple ~120 ms constant-amplitude tone bursts
        (max ~1794, avg ~1137) before the actual speech begins.  Pattern:
        tone → silence → tone → silence → speech.

        Strategy: scan the first *scan_limit_ms* in 10 ms windows.
        Find the last window whose average absolute amplitude exceeds
        *tone_threshold*, then trim everything up to and including that
        window (plus a small 20 ms margin).

        Returns cleaned WAV bytes.  On any error, returns the original.
        """
        try:
            src = wave.open(io.BytesIO(raw_wav), "rb")
            framerate = src.getframerate()
            n_channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            n_frames = src.getnframes()

            if sampwidth != 2:
                src.close()
                return raw_wav  # only handle 16-bit PCM

            window_frames = int(framerate * 0.01)  # 10 ms
            max_windows = int(scan_limit_ms / 10)
            scan_windows = min(max_windows, n_frames // window_frames)

            last_tone_end = 0  # frame index past the last tone window
            for i in range(scan_windows):
                chunk = src.readframes(window_frames)
                samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
                avg_abs = sum(abs(s) for s in samples) // len(samples)
                if avg_abs > tone_threshold:
                    last_tone_end = (i + 1) * window_frames

            if last_tone_end == 0:
                src.close()
                return raw_wav  # no tone detected — keep as-is

            # Add 20 ms margin after last tone
            margin_frames = int(framerate * 0.02)
            skip_frames = min(last_tone_end + margin_frames, n_frames)

            # Re-open and skip to the cut point
            src.close()
            src = wave.open(io.BytesIO(raw_wav), "rb")
            src.readframes(skip_frames)
            remaining = src.readframes(n_frames - skip_frames)
            src.close()

            if not remaining:
                return raw_wav  # nothing left after trim

            buf = io.BytesIO()
            with wave.open(buf, "wb") as dst:
                dst.setnchannels(n_channels)
                dst.setsampwidth(sampwidth)
                dst.setframerate(framerate)
                dst.writeframes(remaining)

            logger.debug(
                f"WAV tone trim: removed first {skip_frames/framerate*1000:.0f}ms "
                f"({skip_frames} frames)"
            )
            return buf.getvalue()
        except Exception as e:
            logger.debug(f"WAV trim skipped ({e}), using raw")
            return raw_wav

    async def _zhipu_tts(self, text: str, persona: str, out_path: Path) -> bool:
        """Generate speech via Zhipu GLM-TTS REST API. Returns True on success."""
        if not self.zhipu_key:
            return False

        client = await self._get_http_client()
        voice = ZHIPU_VOICE_MAP.get(persona, self.zhipu_voice)
        body: dict[str, Any] = {
            "model": "glm-tts",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": 1.0,
        }

        try:
            resp = await client.post(
                self.ZHIPU_TTS_URL,
                json=body,
                headers={"Authorization": f"Bearer {self.zhipu_key}"},
            )
            if resp.status_code != 200:
                logger.warning(
                    f"GLM-TTS failed ({resp.status_code}): "
                    f"{resp.text[:200] if resp.text else 'no body'}"
                )
                return False

            # GLM-TTS returns WAV with a ~120ms tone/beep at the start
            # (baked into the PCM data by the model).  We trim it, then
            # convert WAV → MP3 via ffmpeg for Telegram compatibility.
            raw_wav = resp.content
            trimmed_wav = self._trim_leading_tone(raw_wav)

            if _HAS_FFMPEG:
                tmp_wav = out_path.with_suffix(".tmp.wav")
                try:
                    tmp_wav.write_bytes(trimmed_wav)
                    # Convert to OGG/OPUS (Telegram native voice format)
                    # + 50ms fade-in as safety net against any residual click
                    result = subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", str(tmp_wav),
                            "-af", "afade=in:st=0:d=0.05",
                            "-codec:a", "libopus", "-b:a", "64k",
                            "-ar", "48000", str(out_path),
                        ],
                        capture_output=True, timeout=15,
                    )
                    if result.returncode != 0:
                        logger.warning(f"ffmpeg WAV→OGG failed: {result.stderr[:200]}")
                        out_path.write_bytes(trimmed_wav)
                finally:
                    tmp_wav.unlink(missing_ok=True)
            else:
                # No ffmpeg — write trimmed WAV directly
                out_path.write_bytes(trimmed_wav)

            size = out_path.stat().st_size
            if size == 0:
                out_path.unlink(missing_ok=True)
                return False

            logger.info(f"GLM-TTS: {out_path.name} ({size} bytes)")
            return True

        except Exception as e:
            logger.warning(f"GLM-TTS error: {e}")
            out_path.unlink(missing_ok=True)
            return False

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
        emotion: str | None = None,
    ) -> str:
        """Generate audio from text.

        Three-tier fallback: GLM-TTS (OGG/OPUS) → Azure Speech (MP3) → edge-tts (MP3).

        Returns:
            Path to the generated audio file

        Raises:
            VoiceError: If all TTS engines fail
        """
        # Clean text for TTS (remove emoji, action words)
        text = VoiceTextCleaner.clean(text)

        if not text.strip():
            raise VoiceError("Empty text provided for TTS")

        # GLM-TTS outputs OGG/OPUS (Telegram native), fallbacks output MP3
        ogg_path = self._cache_path(text, persona, ext=".ogg")
        mp3_path = self._cache_path(text, persona, ext=".mp3")

        # Return cached version if exists and non-empty (either format)
        for cached in (ogg_path, mp3_path):
            if cached.exists() and cached.stat().st_size > 0:
                logger.debug(f"TTS cache hit: {cached.name}")
                return str(cached)

        # Three-tier fallback: GLM-TTS (WAV→OGG) → Azure (MP3) → edge-tts (MP3)
        if await self._zhipu_tts(text, persona, ogg_path):
            return str(ogg_path)

        if await self._azure_tts(text, persona, mp3_path):
            return str(mp3_path)

        if await self._edge_tts(text, persona, mp3_path):
            return str(mp3_path)

        raise VoiceError("All TTS engines failed")

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """CEO-compatible execute interface."""
        persona = kwargs.get("persona", "jarvis")
        emotion = kwargs.get("emotion")
        try:
            path = await self.text_to_speech(task, persona=persona, emotion=emotion)
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
        "\U00002300-\U000023FF"  # misc technical (⏰⌚ etc.)
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002B05-\U00002B55"  # arrows & geometric emoji
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U000020E3"             # combining keycap
        "\U0000200D"             # ZWJ
        "\U000000A9"             # ©
        "\U000000AE"             # ®
        "\U0000203C"             # ‼
        "\U00002049"             # ⁉
        "\U00002934-\U00002935"  # ⤴⤵
        "\U000025AA-\U000025FE"  # geometric shapes
        "\U00003030"             # 〰
        "\U0000303D"             # 〽
        "\U00003297"             # ㊗
        "\U00003299"             # ㊙
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
