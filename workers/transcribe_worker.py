"""Transcribe Worker — long audio ASR + two-stage LLM summary.

For meeting recordings and long audio files (not short voice clips).
Uses 智譜 GLM-ASR for transcription, then ModelRouter CEO chain for
a structured two-stage summary.

Usage:
    worker = TranscribeWorker(model_router=router, zhipu_key="...")
    result = await worker.process_audio("meeting.m4a", context="週會")
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger


SUPPORTED_FORMATS = {".ogg", ".mp3", ".m4a", ".wav", ".flac", ".opus"}
CHUNK_SIZE = 3000  # chars per summary segment


class TranscribeWorker:
    """Long-form audio transcription + structured meeting summary."""

    def __init__(
        self,
        model_router: Any = None,
        zhipu_key: str = "",
    ) -> None:
        self.name = "transcribe"
        self.router = model_router
        self.zhipu_key = zhipu_key

    # ── Worker interface ──────────────────────────────────────────

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Worker interface.

        Args:
            task: Path to audio file.
            **kwargs: context (str) — topic hint for better summary.

        Returns:
            dict with result (summary), transcript, source, worker.
        """
        context = kwargs.get("context", "")
        return await self.process_audio(task, context=context)

    # ── Public API ────────────────────────────────────────────────

    async def process_audio(
        self, audio_path: str, context: str = "",
    ) -> dict[str, Any]:
        """Transcribe audio and produce a structured summary.

        Args:
            audio_path: Path to the audio file.
            context: Optional topic/meeting context for better summary.

        Returns:
            dict with keys: result (summary str), transcript (raw text),
            source, worker.  On error: error key present.
        """
        path = Path(audio_path)
        if not path.exists():
            return {"error": f"檔案不存在: {audio_path}", "worker": self.name}

        if path.suffix.lower() not in SUPPORTED_FORMATS:
            return {
                "error": f"不支援的格式: {path.suffix} (支援: {', '.join(sorted(SUPPORTED_FORMATS))})",
                "worker": self.name,
            }

        if not self.zhipu_key:
            return {"error": "未設定 ZHIPU_API_KEY，無法轉錄", "worker": self.name}

        # Step 1: ASR transcription
        transcript = await self._transcribe(path)
        if not transcript:
            return {"error": "轉錄失敗，未取得文字", "worker": self.name}

        # Step 2: Two-stage summary
        summary = await self._two_stage_summary(transcript, context)

        return {
            "result": summary,
            "transcript": transcript,
            "source": "transcribe",
            "worker": self.name,
        }

    # ── ASR via zhipuai SDK ───────────────────────────────────────

    async def _transcribe(self, audio_path: Path) -> str:
        """Transcribe audio using 智譜 GLM-ASR.

        Uses the zhipuai SDK synchronously in a thread pool to avoid
        blocking the event loop.
        """
        try:
            transcript = await asyncio.to_thread(
                self._transcribe_sync, audio_path,
            )
            logger.info(
                f"TranscribeWorker: transcribed {audio_path.name} "
                f"({len(transcript)} chars)"
            )
            return transcript
        except Exception as e:
            logger.error(f"TranscribeWorker ASR failed: {e}")
            return ""

    def _transcribe_sync(self, audio_path: Path) -> str:
        """Synchronous transcription call (run in thread)."""
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=self.zhipu_key)
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="glm-asr-2512",
                file=f,
            )
        # result.text contains the transcribed text
        return result.text if hasattr(result, "text") else str(result)

    # ── Two-stage LLM summary ────────────────────────────────────

    async def _two_stage_summary(self, transcript: str, context: str) -> str:
        """Produce a structured summary via two LLM passes.

        Stage 1: Summarize each chunk individually.
        Stage 2: Merge chunk summaries into a final structured output.
        """
        if not self.router:
            return transcript[:2000]  # fallback: raw truncated text

        chunks = self._split_transcript(transcript)

        # Stage 1: per-chunk summaries
        chunk_summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            prompt = (
                f"這是一段錄音的第 {i + 1}/{len(chunks)} 部分逐字稿。"
                f"{f'主題: {context}。' if context else ''}\n\n"
                f"{chunk}\n\n"
                f"請用繁體中文摘要這段內容的重點（3-5 句），"
                f"保留關鍵數字、人名、決議。"
            )
            summary = await self._llm_generate(prompt, max_tokens=400)
            chunk_summaries.append(summary)

        # Stage 2: merge into structured summary
        merged = "\n\n".join(
            f"【第 {i + 1} 段】\n{s}" for i, s in enumerate(chunk_summaries)
        )

        final_prompt = (
            f"以下是一段{'會議' if context else ''}錄音的分段摘要。"
            f"{f'主題: {context}。' if context else ''}\n\n"
            f"{merged}\n\n"
            f"請整合成一份結構化摘要，包含：\n"
            f"📋 會議摘要（3-5 句總結）\n"
            f"✅ 關鍵決議（條列）\n"
            f"📌 行動項目（誰、做什麼、何時）\n"
            f"❓ 未解決問題（如有）\n"
            f"🔢 重要數字（如有）\n\n"
            f"用繁體中文，結論先行。"
        )

        return await self._llm_generate(final_prompt, max_tokens=800)

    def _split_transcript(self, text: str) -> list[str]:
        """Split transcript into chunks of ~CHUNK_SIZE chars."""
        if len(text) <= CHUNK_SIZE:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            # Try to break at a sentence boundary
            if end < len(text):
                for sep in ("。", ".", "\n", "，", ","):
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + 1
                        break
            # Safety: ensure forward progress
            if end <= start:
                end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start = end
        return chunks

    # ── LLM helper ────────────────────────────────────────────────

    async def _llm_generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text via ModelRouter CEO chain."""
        if not self.router:
            return ""
        try:
            from clients.base_client import ChatMessage
            from core.model_router import ModelRole

            response = await self.router.chat(
                [ChatMessage(role="user", content=prompt)],
                role=ModelRole.CEO,
                max_tokens=max_tokens,
            )
            return response.content
        except Exception as e:
            logger.warning(f"TranscribeWorker LLM failed: {e}")
            return ""

    async def close(self) -> None:
        """Cleanup (no persistent resources)."""
