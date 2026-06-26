"""Tests for the local LLM Cache."""

import sqlite3
from pathlib import Path

import pytest

from src.llm_cache import LLMCache


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Provide a temporary database path for isolation."""
    return tmp_path / "test_cache.db"


def test_llm_cache_init_creates_table(temp_db: Path) -> None:
    """Test that initializing LLMCache creates the cache table."""
    LLMCache(temp_db)

    # Verify table exists
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_cache'"
        )
        assert cursor.fetchone() is not None


def test_llm_cache_set_and_get(temp_db: Path) -> None:
    """Test that a stored response can be retrieved if turn_count matches."""
    cache = LLMCache(temp_db)
    response = {"architecture_score": 0.9, "workflow_level": "Senior"}

    # Store
    cache.set("session_123", 5, response)

    # Retrieve with matching turn_count
    retrieved = cache.get("session_123", 5)
    assert retrieved == response


def test_llm_cache_get_miss_wrong_turn_count(temp_db: Path) -> None:
    """Test that get returns None if the turn_count has changed (cache invalidation)."""
    cache = LLMCache(temp_db)
    response = {"architecture_score": 0.9}
    cache.set("session_123", 5, response)

    # Retrieve with different turn_count (e.g. session had new messages)
    assert cache.get("session_123", 6) is None


def test_llm_cache_get_miss_not_found(temp_db: Path) -> None:
    """Test that get returns None for unknown session IDs."""
    cache = LLMCache(temp_db)
    assert cache.get("unknown", 1) is None


def test_llm_cache_set_upsert(temp_db: Path) -> None:
    """Test that setting a response for an existing session overwrites the old one."""
    cache = LLMCache(temp_db)

    # Initial store
    cache.set("session_123", 5, {"score": 0.5})

    # Upsert (new turn_count and new response)
    cache.set("session_123", 6, {"score": 0.9})

    # Old cache should be gone
    assert cache.get("session_123", 5) is None

    # New cache should be returned
    assert cache.get("session_123", 6) == {"score": 0.9}


def test_llm_cache_graceful_error_handling(temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that DB errors are caught and logged gracefully without crashing."""
    cache = LLMCache(temp_db)

    # Force an error by putting invalid JSON into the database directly
    with sqlite3.connect(temp_db) as conn:
        conn.execute(
            "INSERT INTO llm_cache (session_id, turn_count, response_json) VALUES (?, ?, ?)",
            ("bad_json", 1, "{invalid_json"),
        )
        conn.commit()

    # The JSONDecodeError should be caught and get() should return None
    assert cache.get("bad_json", 1) is None
