"""
Corpus — immutable value object that groups RawInteraction records into
chronologically-ordered Sessions.

DESIGN INVARIANTS
- Every dataclass is frozen and uses __slots__ (value semantics, hashable).
- Construction is a pure transformation: list[RawInteraction] → Corpus.
- Orphans and anomalies surface as CorpusIssue, never as exceptions.
- The Corpus never mutates after creation; the scorer can iterate it
  multiple times without defensive copies.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

from src.reader.base import RawInteraction, Role, ToolCall


@dataclass(frozen=True, slots=True)
class CorpusIssue:
    """A data-quality anomaly detected during corpus construction.

    Issues are first-class data, not log lines. The CLI/reporter can
    display them in a "data quality" footer without parsing strings.
    """

    kind: str
    session_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class Turn:
    """One turn of conversation inside a Session.

    A Turn is a projection of RawInteraction: the session_id and project
    have been hoisted into the parent Session, leaving only the per-turn
    data the scorer needs.
    """

    timestamp: datetime
    role: Role
    text: str
    tool_calls: tuple[ToolCall, ...]
    parent_id: str | None


@dataclass(frozen=True, slots=True)
class Session:
    """A single conversation, composed of chronologically-ordered Turns."""

    session_id: str
    project: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class Corpus:
    """Immutable container of Sessions, built from a list of RawInteraction.

    Usage (factory):
        corpus = Corpus.from_interactions(reader.interactions())

    Usage (iteration):
        for session in corpus:
            for turn in session.turns:
                ...

    Usage (flat):
        for turn in corpus.turns():
            ...
    """

    sessions: tuple[Session, ...]
    issues: tuple[CorpusIssue, ...]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_interactions(
        cls,
        interactions: Iterable[RawInteraction],
    ) -> Corpus:
        """Build a Corpus from a flat list of interactions.

        Steps:
        1. Validate each interaction (non-empty session_id).
        2. Group valid interactions by session_id.
        3. Sort each group's interactions by timestamp.
        4. Project RawInteraction → Turn.
        5. Build Session objects (sorted by first timestamp).
        6. Collect any anomalies as CorpusIssue records.
        """
        issues: list[CorpusIssue] = []
        buckets: dict[str, list[RawInteraction]] = defaultdict(list)

        for interaction in interactions:
            sid = interaction.session_id
            if not sid:
                issues.append(
                    CorpusIssue(
                        kind="orphan_session",
                        session_id=sid,
                        detail=f"Orphan interaction with empty session_id "
                        f"(project={interaction.project!r})",
                    )
                )
                continue
            buckets[sid].append(interaction)

        # Build sessions: sort each bucket by timestamp, then project.
        sessions: list[Session] = []
        for sid, bucket in buckets.items():
            bucket.sort(key=lambda i: i.timestamp)
            project = bucket[0].project
            turns = tuple(
                Turn(
                    timestamp=i.timestamp,
                    role=i.role,
                    text=i.text,
                    tool_calls=i.tool_calls,
                    parent_id=i.parent_id,
                )
                for i in bucket
            )
            sessions.append(Session(session_id=sid, project=project, turns=turns))

        # Sort sessions by the timestamp of their first turn for deterministic
        # iteration order.
        sessions.sort(key=lambda s: s.turns[0].timestamp if s.turns else datetime.min)

        return cls(sessions=tuple(sessions), issues=tuple(issues))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sessions)

    def __iter__(self) -> Iterator[Session]:
        return iter(self.sessions)

    def session(self, session_id: str) -> Session | None:
        """Lookup a session by id. O(n) — acceptable for small N."""
        for s in self.sessions:
            if s.session_id == session_id:
                return s
        return None

    def turns(self) -> Iterator[Turn]:
        """Yield every Turn across all sessions, in chronological order
        within each session, sessions ordered by first turn."""
        for s in self.sessions:
            yield from s.turns


__all__ = [
    "Corpus",
    "CorpusIssue",
    "Session",
    "Turn",
]
