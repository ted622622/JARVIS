"""Help Decision Engine — decide retry vs ask-human vs give-up.

Integrates with Patch H's ReactExecutor to prevent "ghost-looping"
(鬼打牆): things that need a human should ask immediately, things
that are impossible should be abandoned with an honest explanation.
"""

from __future__ import annotations

from typing import Literal

Decision = Literal["retry", "ask_human", "give_up"]


# ── Error categories ────────────────────────────────────────────

# Recoverable by retrying (temporary / technical)
SELF_RECOVERABLE: frozenset[str] = frozenset({
    "network_unreachable",
    "network_timeout",
    "timeout",
    "rate_limited",
    "server_error_5xx",
    "unknown",
})

# Needs human intervention immediately
MUST_ASK_HUMAN: frozenset[str] = frozenset({
    "login_required",
    "captcha_detected",
    "payment_required",
    "two_factor_auth",
    "identity_verification",
    "permission_denied",
    "unknown_ui",
    "session_expired",
})

# Not possible — give up and be honest
GIVE_UP: frozenset[str] = frozenset({
    "site_not_accessible",
    "service_discontinued",
    "region_blocked",
    "provider_down",
    "security_blocked",
})


# ── JARVIS-style help messages (zh-TW) ──────────────────────────

JARVIS_HELP_MESSAGES: dict[str, str] = {
    "login_required":   "Sir，{site_name}需要您登入一次。已開啟頁面並截圖，請協助完成登入。",
    "captcha_detected": "Sir，驗證碼已截圖傳送，請協助輸入。",
    "payment_required": "Sir，此操作需要付款。金額截圖如附，是否繼續？",
    "two_factor_auth":  "Sir，系統要求二次驗證，驗證碼應已發送至您的手機。",
    "unknown_ui":       "Sir，我無法辨識目前頁面，已截圖傳送，請指示下一步。",
    "session_expired":  "Sir，{site_name}的登入已過期，需要您重新登入一次。",
    "give_up_network":  "Sir，{site_name}連線失敗，已重試 {attempts} 次。建議直接使用官方 App 或稍後再試。",
    "give_up_general":  "Sir，此任務已嘗試 {attempts} 次均未成功，可能需要您手動處理。",
    "retry_failed":     "Sir，已嘗試 {attempts} 次均失敗，是否需要我繼續嘗試？",
}


class HelpDecisionEngine:
    """Decide what to do when a task fails.

    Returns:
        ``"retry"``     — try again (technical / temporary error)
        ``"ask_human"`` — need human help (login, captcha, payment …)
        ``"give_up"``   — impossible, be honest
    """

    @staticmethod
    def decide(error_type: str, attempt_count: int = 0) -> Decision:
        """Classify failure into an action.

        Args:
            error_type: Normalised error label (e.g. "login_required").
            attempt_count: How many times we have already retried.
        """
        if error_type in MUST_ASK_HUMAN:
            return "ask_human"

        if error_type in GIVE_UP:
            return "give_up"

        if error_type in SELF_RECOVERABLE:
            if attempt_count >= 3:
                return "ask_human"  # exhausted retries → ask human
            return "retry"

        # Unknown error type — retry once, then ask
        if attempt_count >= 1:
            return "ask_human"
        return "retry"

    @staticmethod
    def get_message(
        error_type: str,
        *,
        site_name: str = "此網站",
        attempts: int = 0,
    ) -> str:
        """Get a JARVIS-style message for the given error type."""
        template = JARVIS_HELP_MESSAGES.get(error_type)
        if template:
            return template.format(site_name=site_name, attempts=attempts)

        # Fallback messages
        decision = HelpDecisionEngine.decide(error_type, attempts)
        if decision == "give_up":
            return JARVIS_HELP_MESSAGES["give_up_general"].format(
                site_name=site_name, attempts=attempts,
            )
        if decision == "ask_human":
            return JARVIS_HELP_MESSAGES["retry_failed"].format(
                site_name=site_name, attempts=attempts,
            )
        return ""
