"""
Tests for the Data Readers layer.

These tests assume OpenCodeReader will be implemented in src/reader/opencode.py.
This file is the RED phase of TDD: the import of OpenCodeReader is expected
to fail until the Green phase lands.

Coverage:
- OpenCodeReader rejects symlink paths via resolve_db_path (defence in depth).
- interactions() returns normalized RawInteraction records.
- Long text is truncated to MAX_HUMAN_PROMPT_CHARS.
- session_count() returns the correct count.
- OpenCodeReader works as a context manager (with-statement).
- Boundary cases: empty database, multiple sessions, tool calls.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config import MAX_HUMAN_PROMPT_CHARS
from src.reader.base import AbstractReader, RawInteraction, ToolCall

# ---------------------------------------------------------------------------
# Fixtures: synthetic OpenCode SQLite schema
# ---------------------------------------------------------------------------

# Minimal schema that mirrors the (assumed) OpenCode database shape.
# Each table mirrors the real OpenCode columns the reader will need.
_SCHEMA_SQL = """
CREATE TABLE session (
    id        TEXT PRIMARY KEY,
    directory TEXT NOT NULL
);

CREATE TABLE message (
    id        TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    time_created INTEGER NOT NULL,   -- Unix epoch in milliseconds
    data      TEXT NOT NULL          -- JSON containing role, parentID
);

CREATE TABLE part (
    id         TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    data       TEXT                  -- JSON containing type, text, tool, state
);
"""


def _create_opencode_db(path: Path) -> sqlite3.Connection:
    """Create a fresh OpenCode-shaped database at `path` and return the connection."""
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


@pytest.fixture
def opencode_db(tmp_path: Path) -> Iterator[Path]:
    """A real on-disk SQLite file with the OpenCode schema and one normal session.

    Uses tmp_path so the test is fully isolated and has no global state.
    """
    db_path = tmp_path / "opencode.db"
    conn = _create_opencode_db(db_path)

    # One session in /home/carlos/work/myapp (will be scrubbed by the reader).
    conn.execute(
        "INSERT INTO session (id, directory) VALUES (?, ?)",
        ("01ARZ3NDEKTSV4RRFFQ69G5FAV", "/home/carlos/work/myapp"),
    )
    # A user message with normal text.
    # 2024-01-15 10:00:00 UTC in ms = 1705312800000
    conn.execute(
        "INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
        (
            "msg-user-1",
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            1705312800000,
            '{"role": "user", "parentID": null}',
        ),
    )
    conn.execute(
        "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
        ("part-user-1", "msg-user-1", '{"type": "text", "text": "Refactor the scoring engine"}'),
    )
    # An assistant message with a tool call.
    conn.execute(
        "INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
        (
            "msg-asst-1",
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            1705312860000,
            '{"role": "assistant", "parentID": "msg-user-1"}',
        ),
    )
    text_json = '{"type": "text", "text": "I will read the file first."}'
    tool_call_json = (
        '{"type": "tool", "tool": "read", "state": {"input": {"path": "src/scorer.py"}}}'
    )
    conn.execute(
        "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
        ("part-asst-1-text", "msg-asst-1", text_json),
    )
    conn.execute(
        "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
        ("part-asst-1-tool", "msg-asst-1", tool_call_json),
    )
    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture
def opencode_db_with_oversized_text(tmp_path: Path) -> Iterator[Path]:
    """A database containing a message whose text exceeds MAX_HUMAN_PROMPT_CHARS.

    Boundary test: the reader must truncate to MAX_HUMAN_PROMPT_CHARS.
    """
    db_path = tmp_path / "opencode.db"
    conn = _create_opencode_db(db_path)

    conn.execute(
        "INSERT INTO session (id, directory) VALUES (?, ?)",
        ("01ARZ3NDEKTSV4RRFFQ69G5FAV", "/srv/apps/backend"),
    )
    huge_text = "x" * (MAX_HUMAN_PROMPT_CHARS + 500)
    import json

    conn.execute(
        "INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
        (
            "msg-huge",
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            1705312800000,
            '{"role": "user", "parentID": null}',
        ),
    )
    part_json = json.dumps({"type": "text", "text": huge_text})
    conn.execute(
        "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
        ("part-huge", "msg-huge", part_json),
    )
    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture
def opencode_db_empty(tmp_path: Path) -> Iterator[Path]:
    """A valid OpenCode database with zero sessions — boundary test for empty input."""
    db_path = tmp_path / "opencode.db"
    conn = _create_opencode_db(db_path)
    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture
def opencode_db_multi_session(tmp_path: Path) -> Iterator[Path]:
    """A database with two distinct sessions — boundary test for session_count."""
    db_path = tmp_path / "opencode.db"
    conn = _create_opencode_db(db_path)

    for sid, directory in [
        ("01ARZ3NDEKTSV4RRFFQ69G5FAV", "/home/carlos/projectA"),
        ("01BRZ3NDEKTSV4RRFFQ69G5FBV", "/home/carlos/projectB"),
    ]:
        conn.execute(
            "INSERT INTO session (id, directory) VALUES (?, ?)",
            (sid, directory),
        )
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
            (f"msg-{sid}", sid, 1705312800000, '{"role": "user", "parentID": null}'),
        )
        conn.execute(
            "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
            (f"part-{sid}", f"msg-{sid}", '{"type": "text", "text": "hello"}'),
        )
    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture
def symlinked_db(tmp_path: Path) -> Iterator[Path]:
    """A real DB plus a symlink pointing to it — boundary test for symlink rejection."""
    real = tmp_path / "real.db"
    conn = _create_opencode_db(real)
    conn.execute(
        "INSERT INTO session (id, directory) VALUES (?, ?)",
        ("01ARZ3NDEKTSV4RRFFQ69G5FAV", "/srv/test"),
    )
    conn.commit()
    conn.close()

    link = tmp_path / "link.db"
    link.symlink_to(real)
    yield link


# ---------------------------------------------------------------------------
# Helper: import OpenCodeReader lazily so a missing class fails the test
# in a clear way (ImportError or AttributeError), not a NameError at
# collection time.
# ---------------------------------------------------------------------------


def _opencode_reader_class() -> type:
    from src.reader.opencode import OpenCodeReader

    return OpenCodeReader


# ---------------------------------------------------------------------------
# Test 1: constructor rejects symlink paths via resolve_db_path
# ---------------------------------------------------------------------------


class TestOpenCodeReaderRejectsSymlinks:
    def test_symlink_path_raises_configuration_error(self, symlinked_db: Path):
        """OpenCodeReader must defer to resolve_db_path, which rejects symlinks."""
        from src.errors import ConfigurationError

        Reader = _opencode_reader_class()
        with pytest.raises(ConfigurationError, match="symbolic link"):
            Reader(symlinked_db)

    def test_nonexistent_path_raises(self, tmp_path: Path):
        """A missing DB file must surface as FileNotFoundError, not a cryptic SQLite error."""
        Reader = _opencode_reader_class()
        missing = tmp_path / "does_not_exist.db"
        with pytest.raises(FileNotFoundError):
            Reader(missing)


# ---------------------------------------------------------------------------
# Test 2: interactions() returns mapped and truncated records
# ---------------------------------------------------------------------------


class TestOpenCodeReaderInteractions:
    def test_returns_list_of_raw_interaction(self, opencode_db: Path):
        Reader = _opencode_reader_class()
        with Reader(opencode_db) as r:
            result = r.interactions()

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, RawInteraction) for item in result)

    def test_fields_are_correctly_mapped(self, opencode_db: Path):
        Reader = _opencode_reader_class()
        with Reader(opencode_db) as r:
            interactions = r.interactions()

        user_msg = next(i for i in interactions if i.role == "user")
        assert user_msg.session_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert user_msg.text == "Refactor the scoring engine"
        assert user_msg.timestamp == datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        assert user_msg.tool_calls == ()
        # Project label is scrubbed: "work/myapp", not "/home/carlos/work/myapp".
        assert "carlos" not in user_msg.project
        assert "myapp" in user_msg.project

    def test_tool_calls_are_parsed_into_tuple(self, opencode_db: Path):
        Reader = _opencode_reader_class()
        with Reader(opencode_db) as r:
            interactions = r.interactions()

        asst = next(i for i in interactions if i.role == "assistant")
        assert len(asst.tool_calls) == 1
        assert isinstance(asst.tool_calls[0], ToolCall)
        assert asst.tool_calls[0].name == "read"
        assert "scorer.py" in asst.tool_calls[0].arguments

    def test_oversized_text_is_truncated(self, opencode_db_with_oversized_text: Path):
        """Boundary: text > MAX_HUMAN_PROMPT_CHARS must be truncated to that limit."""
        Reader = _opencode_reader_class()
        with Reader(opencode_db_with_oversized_text) as r:
            interactions = r.interactions()

        assert len(interactions) == 1
        assert len(interactions[0].text) == MAX_HUMAN_PROMPT_CHARS
        assert interactions[0].text == "x" * MAX_HUMAN_PROMPT_CHARS

    def test_parent_id_is_propagated(self, opencode_db: Path):
        Reader = _opencode_reader_class()
        with Reader(opencode_db) as r:
            interactions = r.interactions()

        asst = next(i for i in interactions if i.role == "assistant")
        assert asst.parent_id == "msg-user-1"

        user = next(i for i in interactions if i.role == "user")
        assert user.parent_id is None

    def test_empty_database_returns_empty_list(self, opencode_db_empty: Path):
        """Boundary: no sessions → empty list, not an error."""
        Reader = _opencode_reader_class()
        with Reader(opencode_db_empty) as r:
            assert r.interactions() == []


# ---------------------------------------------------------------------------
# Test 3: session_count() returns the correct number
# ---------------------------------------------------------------------------


class TestOpenCodeReaderSessionCount:
    def test_single_session(self, opencode_db: Path):
        Reader = _opencode_reader_class()
        with Reader(opencode_db) as r:
            assert r.session_count() == 1

    def test_multiple_sessions(self, opencode_db_multi_session: Path):
        Reader = _opencode_reader_class()
        with Reader(opencode_db_multi_session) as r:
            assert r.session_count() == 2

    def test_empty_database_returns_zero(self, opencode_db_empty: Path):
        Reader = _opencode_reader_class()
        with Reader(opencode_db_empty) as r:
            assert r.session_count() == 0


# ---------------------------------------------------------------------------
# Test 4: context manager closes the connection
# ---------------------------------------------------------------------------


class TestOpenCodeReaderContextManager:
    def test_with_statement_closes_connection(self, opencode_db: Path, monkeypatch):
        """`with Reader(path) as r:` must close the underlying SQLite connection.

        We patch sqlite3.connect to record the connection object and verify
        `close()` was called on it after the with-block exits.
        """
        Reader = _opencode_reader_class()

        connections: list[sqlite3.Connection] = []
        original_connect = sqlite3.connect

        def tracking_connect(*args, **kwargs):
            conn = original_connect(*args, **kwargs)
            connections.append(conn)
            return conn

        monkeypatch.setattr(sqlite3, "connect", tracking_connect)

        with Reader(opencode_db) as r:
            assert isinstance(r, AbstractReader) or hasattr(r, "interactions")
            _ = r.interactions()

        assert connections, "OpenCodeReader must open a sqlite3 connection"
        # The connection object is closed (in __exit__) — check via an operation
        # that requires an open connection. The previous line
        # `connections[0]._conn.__exit__ is not None or True` was a typo
        # (sqlite3.Connection has no _conn attribute) and is removed.
        with pytest.raises((sqlite3.ProgrammingError, sqlite3.OperationalError)):
            connections[0].execute("SELECT 1")

    def test_explicit_close_is_idempotent(self, opencode_db: Path):
        Reader = _opencode_reader_class()
        reader = Reader(opencode_db)
        reader.close()
        # Calling close a second time must not raise.
        reader.close()
