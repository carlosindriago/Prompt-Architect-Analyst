"""
AbstractReader — the contract every reader must fulfill.

This module defines the data model and the protocol that all concrete
readers (OpenCodeReader, future ClaudeCodeReader, etc.) must implement.

DESIGN INVARIANTS
- RawInteraction is immutable and hashable (frozen + slots).
- AbstractReader is a Protocol, not an ABC. No inheritance tax.
- No method on a reader mutates the source data. Read-only by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

# Valid message roles emitted by the reader. "system" is filtered out
# upstream because the scorer is not interested in injected context.
Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation embedded in an assistant message.

    Immutable so the same record can be shared across passes of the
    scoring engine without copy-on-write overhead.
    """

    name: str
    arguments: str  # raw JSON string, never parsed here


@dataclass(frozen=True, slots=True)
class RawInteraction:
    """One turn of conversation, normalized for the scorer.

    Invariants enforced by the reader (not the dataclass, which stays pure):
        - session_id is a valid ULID.
        - project is already scrubbed of the system username.
        - timestamp is UTC-aware.
        - text length <= MAX_HUMAN_PROMPT_CHARS (6000).
        - role is only "user" or "assistant".
        - tool_calls is empty for role="user".
    """

    session_id: str
    project: str
    timestamp: datetime
    role: Role
    text: str
    tool_calls: tuple[ToolCall, ...]
    parent_id: str | None


@runtime_checkable
class AbstractReader(Protocol):
    """Protocol every reader must satisfy.

    Readers are short-lived: construct, query, close. They are not
    designed to be reused across passes because SQLite connections
    hold OS resources and the scoring engine runs in a single pass
    over the in-memory list returned by `interactions()`.
    """

    def interactions(self) -> list[RawInteraction]:
        """Return every interaction across every session, in insertion order.

        The result is a list, not a generator, so the scorer can iterate
        it multiple times (per-dimension, confidence shrinkage) without
        re-querying. Memory is bounded by the caller's session cap.
        """
        ...

    def session_count(self) -> int:
        """Return the number of distinct sessions in the source."""
        ...

    def close(self) -> None:
        """Release the underlying file handle. Idempotent."""
        ...


__all__ = [
    "Role",
    "ToolCall",
    "RawInteraction",
    "AbstractReader",
]
