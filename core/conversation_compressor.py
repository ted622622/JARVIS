"""Conversation Compressor — prevent context window overflow.

Keeps the last N turns in full detail, compresses older turns into
one-line summaries.  Pairs with G2 Memory Flush so that important
information is saved to Markdown before compression discards it.
"""

from __future__ import annotations

from datetime import datetime


class ConversationCompressor:
    """Compress conversation history to keep context window manageable.

    Recent turns are kept verbatim; older turns are collapsed into
    one-line summaries.  The CEO Agent always sees a bounded context.

    Args:
        recent_turns_keep: Number of recent *user+assistant pairs* to keep.
        max_summary_lines: Maximum number of summary lines retained.
    """

    def __init__(
        self,
        recent_turns_keep: int = 10,
        max_summary_lines: int = 30,
    ):
        self.recent_turns_keep = recent_turns_keep
        self.max_summary_lines = max_summary_lines
        self.full_history: list[dict] = []
        self.compressed_summary: list[str] = []

    # ── Public API ──────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """Append a single turn (user or assistant) and compress if needed."""
        self.full_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._compress_if_needed()

    def get_context_for_ceo(self) -> list[dict]:
        """Return a compressed message list suitable for the LLM.

        Structure:
          1. A system message with the compressed summary (if any).
          2. The most recent *recent_turns_keep* pairs in full.
        """
        result: list[dict] = []

        if self.compressed_summary:
            result.append({
                "role": "system",
                "content": "[先前對話摘要]\n" + "\n".join(self.compressed_summary),
            })

        # Keep the last N pairs (each pair = 2 messages: user + assistant)
        keep_count = self.recent_turns_keep * 2
        recent = self.full_history[-keep_count:]
        for turn in recent:
            result.append({
                "role": turn["role"],
                "content": turn["content"],
            })

        return result

    @property
    def turn_count(self) -> int:
        """Total number of messages (including compressed ones conceptually)."""
        return len(self.full_history) + len(self.compressed_summary)

    def reset(self) -> None:
        """Clear all history and summaries."""
        self.full_history.clear()
        self.compressed_summary.clear()

    # ── Internal ────────────────────────────────────────────────

    def _compress_if_needed(self) -> None:
        """When history exceeds the keep window, compress old turns."""
        total = len(self.full_history)
        keep_from = total - self.recent_turns_keep * 2

        if keep_from <= 0:
            return

        to_compress = self.full_history[:keep_from]

        # Walk through pairs (user, assistant) and build summaries
        i = 0
        while i < len(to_compress):
            user_msg = to_compress[i]["content"][:80]
            ts = to_compress[i].get("timestamp", "")[:10]

            assistant_msg = ""
            if i + 1 < len(to_compress):
                assistant_msg = to_compress[i + 1]["content"][:80]
                i += 2
            else:
                i += 1

            summary = f"[{ts}] 用戶: {user_msg}... → 回覆: {assistant_msg}..."

            if summary not in self.compressed_summary:
                self.compressed_summary.append(summary)

        # Trim history to only the recent window
        self.full_history = self.full_history[keep_from:]

        # Cap summary length
        if len(self.compressed_summary) > self.max_summary_lines:
            self.compressed_summary = self.compressed_summary[-self.max_summary_lines:]
