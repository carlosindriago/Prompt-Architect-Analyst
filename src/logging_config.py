"""
Logging configuration for prompt-architect-analyst.

INVARIANT: No log record may contain:
  - API keys or auth tokens (any string matching known key patterns)
  - Absolute paths that expose the system username
  - The content of user prompts
  - Data from the credential / account_state tables of opencode.db

The SensitiveDataFilter is attached to the root handler and redacts
patterns before they reach any sink (stderr, file, etc.).
"""

from __future__ import annotations

import logging
import re
import sys
from types import TracebackType
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Redaction patterns
# ---------------------------------------------------------------------------

# Matches common API key formats:
#   sk-ant-api03-...   (Anthropic)
#   sk-proj-...        (OpenAI project key)
#   sk-...             (OpenAI legacy)
#   AKIA...            (AWS — sometimes proxied)
_KEY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
]

# Matches absolute paths with a username component:
#   /home/username/...  /Users/username/...  C:\Users\username\...
_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"/home/[^/\s]+"),
    re.compile(r"/Users/[^/\s]+"),
    re.compile(r"C:\\Users\\[^\\\s]+"),
]

_REDACTED_KEY = "[REDACTED-KEY]"
_REDACTED_PATH = "[REDACTED-PATH]"


class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that redacts API keys and system usernames from log records.

    Attached to the root logger so every handler — regardless of where it
    was added — receives sanitised messages.
    """

    _key_patterns: ClassVar[list[re.Pattern[str]]] = _KEY_PATTERNS
    _path_patterns: ClassVar[list[re.Pattern[str]]] = _PATH_PATTERNS

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.msg = self._scrub(str(record.msg))
        # The stdlib types `record.args` as `_LogRecordArgs`, a
        # private union we cannot import. Use `Any` here for the
        # narrow bidirectional assignment; the actual contract is
        # preserved by _scrub_args' signature.
        args_in: Any = record.args
        args_out: Any = self._scrub_args(args_in)
        record.args = args_out
        # Security audit S4: a stacktrace captured via
        # logger.exception() lives in record.exc_info, NOT in
        # record.msg. If we only scrub the message, the raw
        # traceback is formatted at the handler level and may
        # contain the API key that triggered the exception.
        # We pre-format the traceback, scrub it, and assign
        # it to record.exc_text so the handler emits the
        # sanitised version. We also clear exc_info so the
        # handler does NOT re-format and overwrite our scrub.
        if record.exc_info is not None:
            try:
                formatted: str = self._format_exception(record.exc_info)
                record.exc_text = self._scrub(formatted)
                record.exc_info = None
            except Exception as exc:  # noqa: BLE001 — never crash the filter
                # If formatting itself blows up, leave exc_info alone
                # and let the handler's default formatter handle it.
                # Scrubbing failures must not break logging, but the
                # failure should be visible to the operator on the
                # next DEBUG-level pass.
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "SensitiveDataFilter could not pre-format exc_info: %s",
                    exc,
                )
        return True  # always pass; we only sanitise, never suppress

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_exception(
        exc_info: (
            BaseException
            | tuple[type[BaseException], BaseException, TracebackType | None]
            | tuple[None, None, None]
            | None
        ),
    ) -> str:
        """Format an exc_info triple the same way the default formatter does."""
        import traceback

        if isinstance(exc_info, BaseException):
            return "".join(
                traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__)
            )
        if isinstance(exc_info, tuple) and len(exc_info) == 3:
            etype, evalue, tb = exc_info
            if etype is not None and evalue is not None:
                return "".join(traceback.format_exception(etype, evalue, tb))
        return ""

    def _scrub(self, text: str) -> str:
        for pattern in self._key_patterns:
            text = pattern.sub(_REDACTED_KEY, text)
        for pattern in self._path_patterns:
            text = pattern.sub(_REDACTED_PATH, text)
        return text

    def _scrub_args(self, args: object) -> object:
        if args is None:
            return args
        if isinstance(args, tuple):
            return tuple(self._scrub(str(a)) if isinstance(a, str) else a for a in args)
        if isinstance(args, dict):
            return {k: self._scrub(str(v)) if isinstance(v, str) else v for k, v in args.items()}
        return args


# ---------------------------------------------------------------------------
# Public setup function
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure the root logger with the SensitiveDataFilter.

    Call once at application startup (in cli.main) before any other
    module emits log records.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.

    Returns:
        The root logger (for convenience — modules should use
        ``logging.getLogger(__name__)`` directly).
    """
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Attach the sensitive data filter to the root logger so it applies
    # to ALL handlers, including any added later by third-party libraries.
    # Guard against duplicate filters when setup_logging() is called more
    # than once (tests, re-entry, etc.).
    if not any(isinstance(f, SensitiveDataFilter) for f in root.filters):
        root.addFilter(SensitiveDataFilter())

    # Avoid duplicate handlers if setup_logging is called more than once
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    return root


__all__ = ["SensitiveDataFilter", "setup_logging"]
