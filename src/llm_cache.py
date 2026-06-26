"""
SQLite-backed local cache for LLM responses.

This prevents the N+1 problem by caching qualitative evaluations
so we only call the LLM when a session is new or has been updated
(i.e., its turn_count has increased).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

_logger: logging.Logger = logging.getLogger(__name__)


class LLMCache:
    """A persistent local cache for LLM analysis results.

    Keys are a combination of session_id and turn_count. If a session
    receives new messages (turn_count increases), the cache is naturally
    invalidated/overwritten.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the cache. If db_path is None, defaults to the user config dir."""
        if db_path is None:
            config_dir = Path.home() / ".config" / "prompt-architect-analyst"
            config_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = config_dir / "cache.db"
        else:
            self.db_path = db_path

        # Ensure the parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the cache table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS llm_cache (
            session_id TEXT PRIMARY KEY,
            turn_count INTEGER NOT NULL,
            fingerprint TEXT NOT NULL DEFAULT '',
            response_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(query)
                # Ensure existing installations get the new column
                try:
                    conn.execute(
                        "ALTER TABLE llm_cache ADD COLUMN fingerprint TEXT NOT NULL DEFAULT ''"
                    )
                except sqlite3.OperationalError:
                    pass
                conn.commit()
        except sqlite3.Error as exc:
            _logger.warning("Failed to initialize LLMCache DB at %s: %s", self.db_path, exc)

    def get(self, session_id: str, turn_count: int, fingerprint: str = "") -> dict[str, Any] | None:
        """Retrieve a cached LLM response for a session if the turn count matches."""
        query = "SELECT turn_count, fingerprint, response_json FROM llm_cache WHERE session_id = ?"
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, (session_id,))
                row = cursor.fetchone()

                if row is None:
                    return None

                cached_turn_count, cached_fingerprint, response_json = row
                if cached_turn_count != turn_count:
                    return None  # Session has new messages, cache is stale
                if fingerprint and cached_fingerprint and cached_fingerprint != fingerprint:
                    return None  # Different prompt / model context, cache is stale

                return json.loads(response_json)
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            _logger.warning("Failed to read from LLMCache for session %s: %s", session_id, exc)
            return None

    def set(
        self, session_id: str, turn_count: int, response: dict[str, Any], fingerprint: str = ""
    ) -> None:
        """Store or update an LLM response in the cache."""
        query = """
        INSERT INTO llm_cache (session_id, turn_count, fingerprint, response_json, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(session_id) DO UPDATE SET
            turn_count = excluded.turn_count,
            fingerprint = excluded.fingerprint,
            response_json = excluded.response_json,
            updated_at = CURRENT_TIMESTAMP
        """
        try:
            response_str = json.dumps(response)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(query, (session_id, turn_count, fingerprint, response_str))
                conn.commit()
        except sqlite3.Error as exc:
            _logger.warning("Failed to write to LLMCache for session %s: %s", session_id, exc)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM llm_cache")
                conn.commit()
        except sqlite3.Error as exc:
            _logger.warning("Failed to clear LLMCache: %s", exc)
