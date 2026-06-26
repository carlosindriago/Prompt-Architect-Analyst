"""
OpenCodeReader — reads OpenCode sessions from opencode.db (SQLite).

SECURITY GUARANTEES
- The DB path is funneled through resolve_db_path() before any I/O,
  so symlinks, path traversal, and non-SQLite files are rejected at
  the constructor boundary.
- The SQLite connection is opened with ?mode=ro (read-only). No write
  operation is possible through this class.
- All SQL queries use parameterized placeholders (?); no f-strings or
.format() are used to build SQL.

RESOURCE MANAGEMENT
- The class implements the context-manager protocol (__enter__/__exit__).
- close() is idempotent and safe to call multiple times.
- Database errors are caught and wrapped in the project's DatabaseError
  hierarchy so callers can handle a single exception type.

NEW SCHEMA (OpenCode >= 4.x)
- message table stores metadata (role, parentID, timestamps) in a JSON
  `data` column; role / createdAt / parentID are no longer flat columns.
- part table stores all data in a JSON `data` column with a `type` field
  ("text", "reasoning", "tool", etc.) instead of flat `text`/`tool_call`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict

from src.config import MAX_HUMAN_PROMPT_CHARS, resolve_db_path
from src.errors import DatabaseError
from src.reader.base import RawInteraction, Role, ToolCall
from src.utils import project_label, safe_json_loads, ts_from_ms

# The only role values the reader emits. System/injected messages are
# filtered at the SQL level; the scorer never sees them.
_ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant"})


# SQL identifier allowlist. Identifiers (table/column names) cannot be
# parameterized in SQLite, so they must come from a hard-coded set.
_QUERY_INTERACTIONS: str = (
    "SELECT "
    "  m.id, m.session_id, "
    "  json_extract(m.data, '$.role'), "
    "  m.time_created, "
    "  json_extract(m.data, '$.parentID'), "
    "  s.directory, "
    "  p.id, p.data "
    "FROM message m "
    "JOIN session s ON s.id = m.session_id "
    "LEFT JOIN part p ON p.message_id = m.id "
    "WHERE json_extract(m.data, '$.role') IN (?, ?) "
    "ORDER BY m.time_created ASC, m.id ASC, p.id ASC"
)

_QUERY_SESSION_COUNT: str = "SELECT COUNT(*) FROM session"


# TypedDict for the per-message aggregation bucket. T1: replaces
# dict[str, Any] with a precise schema; mypy now catches typos
# like bucket["text"] vs bucket["texts"].
class _MessageBucket(TypedDict):
    session_id: str
    directory: str | None
    role: str
    created_at_ms: int
    parent_id: str | None
    texts: list[str]
    tool_calls: list[ToolCall]


class OpenCodeReader:
    """Read-only reader for the OpenCode SQLite database.

    The constructor validates the path through resolve_db_path() and
    opens a read-only SQLite connection. Use as a context manager:

        with OpenCodeReader(db_path) as reader:
            interactions = reader.interactions()
            count = reader.session_count()
    """

    __slots__ = ("_conn", "_closed")

    def __init__(self, db_path: Path | str) -> None:
        # resolve_db_path returns an absolute, validated string. We
        # accept Path or str; the resolver handles both.
        validated: str = resolve_db_path(str(db_path))
        self._conn: sqlite3.Connection = self._open_connection(validated)
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @staticmethod
    def _open_connection(validated_path: str) -> sqlite3.Connection:
        """Open a read-only SQLite connection, wrapping any error."""
        uri = f"file:{validated_path}?mode=ro"
        try:
            # uri=True is required for the mode=ro URI scheme.
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to open OpenCode database at {validated_path}: {exc}"
            ) from exc
        return conn

    def close(self) -> None:
        """Release the SQLite connection. Idempotent."""
        if self._closed:
            return
        try:
            self._conn.close()
        except sqlite3.Error as exc:
            # Closing a closed connection is fine; any other error is
            # surfaced but does not propagate as fatal.
            raise DatabaseError(f"Error closing database connection: {exc}") from exc
        finally:
            self._closed = True

    def __enter__(self) -> OpenCodeReader:
        if self._closed:
            raise DatabaseError("Cannot reopen a closed OpenCodeReader")
        return self

    # T6: precise type hints for the context-manager protocol.
    # Python passes the exception info from the with-block's
    # `__exit__`; the types are part of the language spec.
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # AbstractReader implementation
    # ------------------------------------------------------------------

    def session_count(self) -> int:
        """Return the number of distinct sessions in the source."""
        try:
            cursor = self._conn.execute(_QUERY_SESSION_COUNT)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to count sessions: {exc}") from exc
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def interactions(self) -> list[RawInteraction]:
        """Return every user/assistant interaction, in chronological order.

        The query joins session + message + part in a single round-trip.
        Python-side aggregation groups parts by message so each
        RawInteraction carries the union of that message's text and
        tool calls.

        New schema (OpenCode >= 4.x):
            - message.data  → JSON with role, timestamps, parentID
            - part.data     → JSON with type (text/reasoning/tool/...)
              and typed content fields rather than flat text/tool_call cols.
        """
        try:
            cursor = self._conn.execute(_QUERY_INTERACTIONS, ("user", "assistant"))
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to query interactions: {exc}") from exc

        # Group parts by message_id while iterating. Using a dict of
        # lists avoids re-querying and keeps memory bounded by the
        # number of parts (not messages × parts).
        by_message: dict[str, _MessageBucket] = {}

        for row in cursor.fetchall():
            (
                msg_id,  # m.id
                session_id,  # m.session_id
                role,  # json_extract(m.data, '$.role')
                created_at_ms,  # m.time_created
                parent_id,  # json_extract(m.data, '$.parentID')
                directory,  # s.directory
                part_id,  # p.id (NULL if no parts)
                part_data,  # p.data (JSON string or None)
            ) = row

            if role not in _ALLOWED_ROLES:
                # Defence in depth: the SQL already filters, but if a
                # future change widens the IN clause we still refuse
                # to emit unsupported roles.
                continue

            bucket: _MessageBucket | None = by_message.get(msg_id)
            if bucket is None:
                bucket = _MessageBucket(
                    session_id=session_id,
                    directory=directory,
                    role=role,
                    created_at_ms=created_at_ms,
                    parent_id=parent_id,
                    texts=[],
                    tool_calls=[],
                )
                by_message[msg_id] = bucket

            # Parse part.data JSON and dispatch by type.
            if part_data is not None:
                self._consume_part(part_data, part_id, msg_id, bucket)

        # Build immutable records. Iteration order follows the SQL
        # ORDER BY (time_created, msg_id) so the output is deterministic.
        result: list[RawInteraction] = []
        for msg_id, bucket in by_message.items():
            result.append(self._build_interaction(msg_id, bucket))
        return result

    # ------------------------------------------------------------------
    # Part-level dispatcher
    # ------------------------------------------------------------------

    @staticmethod
    def _consume_part(
        part_data: str,
        part_id: str | None,
        msg_id: str,
        bucket: _MessageBucket,
    ) -> None:
        """Parse a single part.data JSON blob and append to the bucket.

        The new schema stores all part content in a JSON `data` column
        with a `type` discriminator.  We handle the types that carry
        text or tool-call information and silently skip structural
        types (step-start, step-finish, compaction, patch, file, subtask)
        that the scorer does not need.
        """
        parsed: dict[str, Any] = safe_json_loads(
            part_data, context=f"part {part_id} of message {msg_id}"
        )
        if not parsed:
            return

        part_type: str | None = parsed.get("type")
        if part_type in ("text", "reasoning"):
            text: str | None = parsed.get("text")
            if text:
                bucket["texts"].append(text)
        elif part_type == "tool":
            tool_name: str | None = parsed.get("tool")
            if tool_name:
                # state.input holds the tool arguments as a dict.
                state: dict[str, Any] | None = parsed.get("state")
                arguments: Any = (state or {}).get("input", {})
                tool_call: ToolCall = ToolCall(
                    name=tool_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
                bucket["tool_calls"].append(tool_call)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_interaction(self, msg_id: str, bucket: _MessageBucket) -> RawInteraction:
        """Assemble a single RawInteraction from aggregated parts."""
        try:
            timestamp: datetime = ts_from_ms(bucket["created_at_ms"])
        except ValueError as exc:
            # Surface corrupt timestamps as DatabaseError, not ValueError,
            # so callers only need to handle one hierarchy.
            raise DatabaseError(f"Invalid timestamp in message {msg_id!r}: {exc}") from exc

        # Concat parts with a single space. Real OpenCode messages
        # typically have one text part, but the schema allows many.
        raw_text: str = " ".join(bucket["texts"])
        truncated: str = raw_text[:MAX_HUMAN_PROMPT_CHARS]

        # project_label already scrubs the username.
        project: str = project_label(str(bucket["directory"]))

        # The bucket invariant: role is one of _ALLOWED_ROLES, but
        # we still assert it for type-narrowing before constructing.
        role_str: str = bucket["role"]
        assert role_str in _ALLOWED_ROLES, f"Unexpected role: {role_str!r}"  # nosec B101
        # Narrow to the Role Literal for the RawInteraction constructor.
        role: Role = role_str  # type: ignore[assignment]

        return RawInteraction(
            session_id=bucket["session_id"],
            project=project,
            timestamp=timestamp,
            role=role,
            text=truncated,
            tool_calls=tuple(bucket["tool_calls"]),
            parent_id=bucket["parent_id"],
        )


__all__ = ["OpenCodeReader"]
