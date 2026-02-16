"""Tests for ErrorClassifier — error categorization and strategy mapping."""

from __future__ import annotations

import pytest

from core.error_classifier import ErrorClassifier, ErrorStrategy, ErrorType


class TestClassifyExceptions:
    """Test classify() with various exception/error strings."""

    def test_network_connect_error(self):
        strategy = ErrorClassifier.classify(ConnectionError("Connection refused"))
        assert strategy.error_type == ErrorType.NETWORK_UNREACHABLE
        assert strategy.retry is True
        assert strategy.max_retries == 2
        assert strategy.delay_seconds == 3.0

    def test_dns_error(self):
        strategy = ErrorClassifier.classify("DNS resolution failed")
        assert strategy.error_type == ErrorType.NETWORK_UNREACHABLE

    def test_timeout_error(self):
        strategy = ErrorClassifier.classify(TimeoutError("Request timed out"))
        assert strategy.error_type == ErrorType.TIMEOUT
        assert strategy.retry is True
        assert strategy.max_retries == 1
        assert strategy.delay_seconds == 5.0

    def test_captcha_detected(self):
        strategy = ErrorClassifier.classify("Page requires CAPTCHA verification")
        assert strategy.error_type == ErrorType.CAPTCHA_DETECTED
        assert strategy.retry is False

    def test_login_required_401(self):
        strategy = ErrorClassifier.classify("HTTP 401 Unauthorized")
        assert strategy.error_type == ErrorType.LOGIN_REQUIRED
        assert strategy.retry is False

    def test_login_required_403(self):
        strategy = ErrorClassifier.classify("403 Forbidden access")
        assert strategy.error_type == ErrorType.LOGIN_REQUIRED

    def test_rate_limited_429(self):
        strategy = ErrorClassifier.classify("HTTP 429 Too Many Requests")
        assert strategy.error_type == ErrorType.RATE_LIMITED
        assert strategy.retry is True
        assert strategy.delay_seconds == 10.0

    def test_rate_limit_text(self):
        strategy = ErrorClassifier.classify("rate limit exceeded")
        assert strategy.error_type == ErrorType.RATE_LIMITED

    def test_element_not_found_404(self):
        strategy = ErrorClassifier.classify("HTTP 404 Not Found")
        assert strategy.error_type == ErrorType.ELEMENT_NOT_FOUND
        assert strategy.retry is False

    def test_provider_down(self):
        strategy = ErrorClassifier.classify("RouterError: all providers down")
        assert strategy.error_type == ErrorType.PROVIDER_DOWN
        assert strategy.retry is False
        assert strategy.fallback_worker is None

    def test_security_blocked(self):
        strategy = ErrorClassifier.classify("Blocked by security gate")
        assert strategy.error_type == ErrorType.SECURITY_BLOCKED
        assert strategy.retry is False
        assert strategy.fallback_worker is None

    def test_unknown_error(self):
        strategy = ErrorClassifier.classify("Something completely unexpected")
        assert strategy.error_type == ErrorType.UNKNOWN
        assert strategy.retry is True
        assert strategy.max_retries == 1


class TestClassifyWorkerResult:
    """Test classify_worker_result() with worker result dicts."""

    def test_success_result_returns_none(self):
        result = {"status": "ok", "content": "data"}
        assert ErrorClassifier.classify_worker_result(result) is None

    def test_error_result_classified(self):
        result = {"error": "Connection refused", "worker": "browser"}
        strategy = ErrorClassifier.classify_worker_result(result, "browser")
        assert strategy is not None
        assert strategy.error_type == ErrorType.NETWORK_UNREACHABLE

    def test_timeout_result(self):
        result = {"error": "Request timed out: https://example.com"}
        strategy = ErrorClassifier.classify_worker_result(result)
        assert strategy is not None
        assert strategy.error_type == ErrorType.TIMEOUT

    def test_worker_name_passed(self):
        result = {"error": "unknown failure"}
        strategy = ErrorClassifier.classify_worker_result(result, "browser")
        assert strategy is not None
        assert strategy.error_type == ErrorType.UNKNOWN


class TestStrategyConsistency:
    """Verify strategy field consistency."""

    def test_retry_true_implies_max_retries_positive(self):
        """If retry is True, max_retries must be > 0."""
        test_errors = [
            "Connection refused",
            "timed out",
            "429 rate limit",
            "random unknown error",
        ]
        for err in test_errors:
            strategy = ErrorClassifier.classify(err)
            if strategy.retry:
                assert strategy.max_retries > 0, f"retry=True but max_retries=0 for: {err}"

    def test_retry_false_implies_zero_retries(self):
        """If retry is False, max_retries should be 0."""
        test_errors = [
            "CAPTCHA detected",
            "401 Unauthorized",
            "RouterError: all providers down",
            "Blocked by security gate",
        ]
        for err in test_errors:
            strategy = ErrorClassifier.classify(err)
            if not strategy.retry:
                assert strategy.max_retries == 0, f"retry=False but max_retries>0 for: {err}"

    def test_all_strategies_have_user_message(self):
        """Every strategy must have a non-empty user message."""
        test_errors = [
            "Connection refused", "timed out", "CAPTCHA", "401",
            "429", "404", "RouterError", "security gate", "wtf",
        ]
        for err in test_errors:
            strategy = ErrorClassifier.classify(err)
            assert strategy.user_message_zh, f"Empty user message for: {err}"
