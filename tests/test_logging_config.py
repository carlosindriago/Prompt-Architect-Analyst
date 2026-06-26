"""
Tests for src/logging_config.py

Covers:
- SensitiveDataFilter redacts Anthropic keys (sk-ant-...)
- SensitiveDataFilter redacts OpenAI keys (sk-proj-..., sk-...)
- SensitiveDataFilter redacts AWS-style keys (AKIA...)
- SensitiveDataFilter redacts Bearer tokens
- SensitiveDataFilter redacts absolute paths with usernames
- SensitiveDataFilter passes non-sensitive messages unchanged
- SensitiveDataFilter handles tuple and dict args
- SensitiveDataFilter scrubs API keys from exc_info tracebacks
- setup_logging() returns a Logger and attaches the filter
- setup_logging() is idempotent (no duplicate handlers, no duplicate filters)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.logging_config import SensitiveDataFilter, setup_logging

if TYPE_CHECKING:
    from types import TracebackType

# Synthetic test keys (NOT real credentials) shaped to match each filter
# pattern. Each key body is long enough and uses only characters accepted
# by the corresponding regex.
_ANTHROPIC_KEY = "sk-ant-api03x9d2jf8aaaa"
_OPENAI_PROJ_KEY = "sk-proj-api03x9d2jf8aaaa"
_OPENAI_LEGACY_KEY = "sk-api03x9d2jf8aaaa1q2w3e"
_AWS_KEY = "AKIA" + "A" * 16
_BEARER_TOKEN = "Bearer " + "a" * 20


class TestSensitiveDataFilter:
    def _apply(
        self,
        message: str,
        args: tuple[object, ...] | dict[str, object] | None = None,
    ) -> str:
        """Run the filter on a record and return the sanitised message."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=args,
            exc_info=None,
        )
        f = SensitiveDataFilter()
        f.filter(record)
        # Reconstruct the final message as the handler would
        if record.args:
            return record.msg % record.args
        return record.msg

    # --- API key redaction ---

    def test_redacts_anthropic_key(self):
        result = self._apply(f"key={_ANTHROPIC_KEY}")
        assert _ANTHROPIC_KEY not in result
        assert "[REDACTED-KEY]" in result

    def test_redacts_openai_project_key(self):
        result = self._apply(f"Authorization: {_OPENAI_PROJ_KEY}")
        assert _OPENAI_PROJ_KEY not in result
        assert "[REDACTED-KEY]" in result

    def test_redacts_openai_legacy_key(self):
        result = self._apply(f"using key {_OPENAI_LEGACY_KEY}")
        assert _OPENAI_LEGACY_KEY not in result
        assert "[REDACTED-KEY]" in result

    def test_redacts_aws_style_key(self):
        result = self._apply(f"{_AWS_KEY} was found in config")
        assert _AWS_KEY not in result
        assert "[REDACTED-KEY]" in result

    def test_redacts_bearer_token(self):
        result = self._apply(f"header: {_BEARER_TOKEN}")
        assert _BEARER_TOKEN not in result
        assert "[REDACTED-KEY]" in result

    # --- Path redaction ---

    def test_redacts_linux_home_path(self):
        result = self._apply("reading /home/carlos/projects/app")
        assert "carlos" not in result
        assert "[REDACTED-PATH]" in result

    def test_redacts_macos_users_path(self):
        result = self._apply("found file at /Users/alice/Documents/db")
        assert "alice" not in result
        assert "[REDACTED-PATH]" in result

    # --- Safe messages pass through ---

    def test_safe_message_passes_unchanged(self):
        msg = "Processing session 01ARZ3NDEKTSV4RRFFQ69G5FAV"
        result = self._apply(msg)
        assert result == msg

    def test_tilde_paths_pass_unchanged(self):
        msg = "DB found at ~/.local/share/opencode/opencode.db"
        result = self._apply(msg)
        assert result == msg

    # --- Args handling ---

    def test_redacts_key_in_tuple_args(self):
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="key=%s",
            args=(_ANTHROPIC_KEY,),
            exc_info=None,
        )
        f = SensitiveDataFilter()
        f.filter(record)
        final = record.msg % record.args
        assert _ANTHROPIC_KEY not in final
        assert "[REDACTED-KEY]" in final

    def test_redacts_key_in_dict_args(self):
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="%(key)s",
            args=({"key": _ANTHROPIC_KEY},),
            exc_info=None,
        )
        f = SensitiveDataFilter()
        f.filter(record)
        final = record.msg % record.args
        assert _ANTHROPIC_KEY not in final
        assert "[REDACTED-KEY]" in final

    def test_filter_always_returns_true(self):
        """Filter must never suppress records - only sanitise them."""
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg=_ANTHROPIC_KEY,
            args=None,
            exc_info=None,
        )
        f = SensitiveDataFilter()
        assert f.filter(record) is True


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_verbose_sets_debug_level(self):
        logger = setup_logging(verbose=True)
        assert logger.level == logging.DEBUG

    def test_default_sets_info_level(self):
        logger = setup_logging(verbose=False)
        assert logger.level == logging.INFO

    def test_sensitive_filter_is_attached(self):
        logger = setup_logging()
        filters = logger.filters
        assert any(isinstance(f, SensitiveDataFilter) for f in filters)

    def test_idempotent_no_duplicate_handlers(self):
        """Calling setup_logging twice must not add duplicate StreamHandlers."""
        setup_logging()
        before = sum(
            1 for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)
        )
        setup_logging()
        after = sum(1 for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler))
        assert after == before

    def test_idempotent_no_duplicate_filters(self):
        """Calling setup_logging twice must not add duplicate SensitiveDataFilters."""
        setup_logging()
        setup_logging()
        root = logging.getLogger()
        sdf_count = sum(1 for f in root.filters if isinstance(f, SensitiveDataFilter))
        assert sdf_count == 1


# ---------------------------------------------------------------------------
# Security audit S4: SensitiveDataFilter must scrub tracebacks that contain
# API keys. The key can be in the exception message or in local variables
# captured by the traceback.
# ---------------------------------------------------------------------------


class TestSensitiveDataFilterScrubbsTracebacks:
    def _apply(
        self,
        message: str,
        exc_info: tuple[type[BaseException], BaseException, TracebackType | None],
    ) -> logging.LogRecord:
        """Run the filter on a record with exc_info and return the record."""
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=message,
            args=None,
            exc_info=exc_info,
        )
        SensitiveDataFilter().filter(record)
        return record

    def test_filter_scrubs_api_key_from_exception_traceback(self) -> None:
        """An exc_info containing a key must be redacted in exc_text."""
        secret = "sk-ant-api03x9d2jf8cdef"

        try:
            raise RuntimeError(secret)
        except Exception:
            import sys

            exc_info = sys.exc_info()

        assert exc_info[0] is not None
        assert exc_info[1] is not None
        record = self._apply("LLM call failed", exc_info)

        # exc_text should have been generated and scrubbed.
        assert record.exc_text is not None
        assert secret not in record.exc_text
        assert "[REDACTED-KEY]" in record.exc_text
        # exc_info is cleared so the handler does not re-format raw.
        assert record.exc_info is None
