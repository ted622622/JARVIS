"""Error Classifier — categorize worker errors and determine retry strategy.

Maps exception strings to ErrorType + ErrorStrategy, enabling the ReactExecutor
to decide whether to retry, fallback, or give up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorType(str, Enum):
    NETWORK_UNREACHABLE = "network_unreachable"
    TIMEOUT = "timeout"
    CAPTCHA_DETECTED = "captcha_detected"
    LOGIN_REQUIRED = "login_required"
    RATE_LIMITED = "rate_limited"
    ELEMENT_NOT_FOUND = "element_not_found"
    PROVIDER_DOWN = "provider_down"
    SECURITY_BLOCKED = "security_blocked"
    DEPENDENCY_MISSING = "dependency_missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorStrategy:
    """What to do about a classified error."""

    error_type: ErrorType
    retry: bool
    max_retries: int
    delay_seconds: float
    fallback_worker: str | None
    user_message_zh: str


# ── Classification patterns ──────────────────────────────────────

_PATTERNS: list[tuple[re.Pattern[str], ErrorStrategy]] = [
    (
        re.compile(r"connect|dns|refused|unreachable|no route", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.NETWORK_UNREACHABLE,
            retry=True, max_retries=2, delay_seconds=3.0,
            fallback_worker="knowledge",
            user_message_zh="網路連線失敗，正在嘗試其他方式",
        ),
    ),
    (
        re.compile(r"timeout|timed?\s*out", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.TIMEOUT,
            retry=True, max_retries=1, delay_seconds=5.0,
            fallback_worker="knowledge",
            user_message_zh="連線逾時，正在重試",
        ),
    ),
    (
        re.compile(r"captcha|recaptcha|hcaptcha", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.CAPTCHA_DETECTED,
            retry=False, max_retries=0, delay_seconds=0.0,
            fallback_worker="knowledge",
            user_message_zh="網站需要人機驗證，改用其他方式回答",
        ),
    ),
    (
        re.compile(r"401|403|login|unauthorized|forbidden", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.LOGIN_REQUIRED,
            retry=False, max_retries=0, delay_seconds=0.0,
            fallback_worker="knowledge",
            user_message_zh="需要登入才能存取，改用其他方式回答",
        ),
    ),
    (
        re.compile(r"429|rate.?limit|too many requests", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.RATE_LIMITED,
            retry=True, max_retries=1, delay_seconds=10.0,
            fallback_worker="knowledge",
            user_message_zh="請求頻率過高，稍後重試",
        ),
    ),
    (
        re.compile(r"404|not found|element not found", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.ELEMENT_NOT_FOUND,
            retry=False, max_retries=0, delay_seconds=0.0,
            fallback_worker="knowledge",
            user_message_zh="找不到頁面，改用其他方式回答",
        ),
    ),
    (
        re.compile(r"RouterError|all providers down", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.PROVIDER_DOWN,
            retry=False, max_retries=0, delay_seconds=0.0,
            fallback_worker=None,
            user_message_zh="所有模型服務暫時無法使用",
        ),
    ),
    (
        re.compile(r"blocked by security|security gate", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.SECURITY_BLOCKED,
            retry=False, max_retries=0, delay_seconds=0.0,
            fallback_worker=None,
            user_message_zh="操作被安全閘門攔截",
        ),
    ),
    (
        re.compile(r"playwright.*timeout|page\.goto.*timeout", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.TIMEOUT,
            retry=True, max_retries=1, delay_seconds=5.0,
            fallback_worker="knowledge",
            user_message_zh="瀏覽器操作逾時，正在重試",
        ),
    ),
    (
        re.compile(r"playwright.*not installed|playwright.*missing", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.DEPENDENCY_MISSING,
            retry=False, max_retries=0, delay_seconds=0.0,
            fallback_worker="knowledge",
            user_message_zh="瀏覽器自動化元件未安裝，改用其他方式",
        ),
    ),
    (
        re.compile(r"selector.*not found|waiting for selector", re.IGNORECASE),
        ErrorStrategy(
            error_type=ErrorType.ELEMENT_NOT_FOUND,
            retry=False, max_retries=0, delay_seconds=0.0,
            fallback_worker="knowledge",
            user_message_zh="找不到頁面元素，改用其他方式",
        ),
    ),
]

_UNKNOWN_STRATEGY = ErrorStrategy(
    error_type=ErrorType.UNKNOWN,
    retry=True, max_retries=1, delay_seconds=2.0,
    fallback_worker="knowledge",
    user_message_zh="發生未知錯誤，嘗試其他方式",
)


class ErrorClassifier:
    """Classify exceptions or error strings into an ErrorStrategy."""

    @staticmethod
    def classify(error: Exception | str, worker_name: str = "") -> ErrorStrategy:
        """Classify an error and return the appropriate strategy.

        Args:
            error: The exception or error string to classify.
            worker_name: The worker that produced the error (for logging).

        Returns:
            ErrorStrategy with retry/fallback instructions.
        """
        error_str = str(error)
        for pattern, strategy in _PATTERNS:
            if pattern.search(error_str):
                return strategy
        return _UNKNOWN_STRATEGY

    @staticmethod
    def classify_worker_result(
        result: dict, worker_name: str = "",
    ) -> ErrorStrategy | None:
        """Classify a worker result dict.

        Args:
            result: Worker result dict. If it has an "error" key, classify it.
            worker_name: The worker that produced the result.

        Returns:
            ErrorStrategy if the result contains an error, None if success.
        """
        error = result.get("error")
        if error is None:
            return None
        return ErrorClassifier.classify(error, worker_name)
