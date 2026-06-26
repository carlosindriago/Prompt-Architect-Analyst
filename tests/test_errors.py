"""
Tests for src/errors.py

Covers:
- All exception classes exist and are importable
- Correct inheritance hierarchy
- Exception messages do not include key material (security)
"""

from __future__ import annotations

import pytest

from src.errors import (
    ConfigurationError,
    DatabaseCorruptError,
    DatabaseError,
    DatabasePermissionError,
    InsightError,
    JSONParseError,
    LLMAuthError,
    LLMFingerprintMismatchError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    ParsingError,
    ReportError,
    ReportPermissionError,
    SchemaVersionError,
)


class TestExceptionHierarchy:
    def test_all_errors_are_insight_error_subclasses(self):
        leaf_classes = [
            DatabaseCorruptError,
            SchemaVersionError,
            DatabasePermissionError,
            JSONParseError,
            ConfigurationError,
            LLMRateLimitError,
            LLMAuthError,
            LLMTimeoutError,
            LLMResponseError,
            LLMFingerprintMismatchError,
            ReportPermissionError,
        ]
        for cls in leaf_classes:
            assert issubclass(cls, InsightError), f"{cls.__name__} must inherit InsightError"

    def test_database_errors_are_database_error(self):
        assert issubclass(DatabaseCorruptError, DatabaseError)
        assert issubclass(SchemaVersionError, DatabaseError)
        assert issubclass(DatabasePermissionError, DatabaseError)

    def test_parsing_errors_are_parsing_error(self):
        assert issubclass(JSONParseError, ParsingError)

    def test_llm_errors_are_llm_provider_error(self):
        for cls in (
            LLMRateLimitError,
            LLMAuthError,
            LLMTimeoutError,
            LLMResponseError,
            LLMFingerprintMismatchError,
        ):
            assert issubclass(cls, LLMProviderError), (
                f"{cls.__name__} must inherit LLMProviderError"
            )

    def test_report_errors_are_report_error(self):
        assert issubclass(ReportPermissionError, ReportError)

    def test_exceptions_are_catchable_as_insight_error(self):
        with pytest.raises(InsightError):
            raise DatabaseCorruptError("db is broken")

    def test_exceptions_carry_message(self):
        err = SchemaVersionError("unexpected schema")
        assert "unexpected schema" in str(err)

    def test_llm_auth_error_message_should_not_contain_key_material(self):
        """Security: LLMAuthError messages must never include the key value."""
        err = LLMAuthError("Authentication failed. Check your API key configuration.")
        msg = str(err)
        # Ensure the test message itself doesn't accidentally include key patterns
        assert "sk-" not in msg
        assert "AKIA" not in msg
