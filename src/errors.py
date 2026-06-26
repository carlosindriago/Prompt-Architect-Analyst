"""
Exception hierarchy for prompt-architect-analyst.

DESIGN RULE: Exception messages must never include:
  - API keys or auth tokens
  - The content of user prompts
  - Absolute paths that expose the system username
  - Data from the `credential` / `account_state` tables

Use scrub_paths() from src.utils before interpolating any path into a message.
"""

from __future__ import annotations


class InsightError(Exception):
    """Base class for all prompt-architect-analyst exceptions."""


# ---------------------------------------------------------------------------
# Database errors
# ---------------------------------------------------------------------------


class DatabaseError(InsightError):
    """Error accessing or reading opencode.db."""


class DatabaseCorruptError(DatabaseError):
    """
    The database file is corrupt, incomplete, or not a valid SQLite3 file.
    Check the magic bytes and try opening the file with the sqlite3 CLI.
    """


class SchemaVersionError(DatabaseError):
    """
    The opencode.db schema does not match the version supported by this tool.
    This usually means OpenCode was updated with a breaking schema change.
    Check https://github.com/carlosindriago/prompt-architect-analyst for updates.
    """


class DatabasePermissionError(DatabaseError):
    """The database file exists but cannot be opened for reading."""


# ---------------------------------------------------------------------------
# Parsing errors
# ---------------------------------------------------------------------------


class ParsingError(InsightError):
    """Error parsing a message, part, or JSON field from the database."""


class JSONParseError(ParsingError):
    """
    A `data` column contained invalid JSON.
    The offending content is NOT included in this message to avoid
    accidentally logging user prompt text.
    """


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigurationError(InsightError):
    """Misconfigured flag, missing required setting, or invalid path."""


# ---------------------------------------------------------------------------
# LLM provider errors
# ---------------------------------------------------------------------------


class LLMProviderError(InsightError):
    """Error communicating with an LLM provider."""


class LLMRateLimitError(LLMProviderError):
    """
    The provider returned a rate-limit response (HTTP 429).
    prompt-architect-analyst will retry with exponential backoff up to LLM_MAX_RETRIES.
    """


class LLMAuthError(LLMProviderError):
    """
    The provider rejected the API key (HTTP 401 / 403).
    NEVER include the key value in this message.
    """


class LLMTimeoutError(LLMProviderError):
    """The provider did not respond within LLM_READ_TIMEOUT seconds."""


class LLMResponseError(LLMProviderError):
    """The provider returned an unexpected or unparseable response."""


class LLMFingerprintMismatchError(LLMProviderError):
    """
    The cached LLM analysis was produced for a different corpus.
    The report will fall back to the deterministic output.
    """


# ---------------------------------------------------------------------------
# Report errors
# ---------------------------------------------------------------------------


class ReportError(InsightError):
    """Error generating or writing the HTML report."""


class ReportPermissionError(ReportError):
    """Cannot write the report to the specified output path."""


__all__ = [
    "InsightError",
    "DatabaseError",
    "DatabaseCorruptError",
    "SchemaVersionError",
    "DatabasePermissionError",
    "ParsingError",
    "JSONParseError",
    "ConfigurationError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMAuthError",
    "LLMTimeoutError",
    "LLMResponseError",
    "LLMFingerprintMismatchError",
    "ReportError",
    "ReportPermissionError",
]
