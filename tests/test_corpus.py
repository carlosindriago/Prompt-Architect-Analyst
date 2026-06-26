"""
Tests for src/corpus.py — the Corpus value object.

These tests assume Corpus, Session, Turn, and CorpusIssue will be
implemented in src/corpus.py. This file is the RED phase of TDD:
the import of Corpus is expected to fail until the Green phase lands.

Coverage:
- Empty input produces an empty, valid Corpus.
- RawInteraction list is grouped into Session objects with correct Turns.
- Orphan interactions (invalid session_id) surface as CorpusIssue, never
  as exceptions.
- All dataclasses are frozen — mutation is rejected at runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

# These imports will fail with ImportError until src/corpus.py is created.
from src.corpus import Corpus, CorpusIssue, Session, Turn
from src.reader.base import RawInteraction, Role, ToolCall

# ---------------------------------------------------------------------------
# Fixtures — synthetic RawInteraction records
# ---------------------------------------------------------------------------


def _make_interaction(
    session_id: str,
    project: str,
    role: Role,
    text: str,
    timestamp: datetime | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
    parent_id: str | None = None,
) -> RawInteraction:
    """Factory for RawInteraction with sensible defaults."""
    return RawInteraction(
        session_id=session_id,
        project=project,
        timestamp=timestamp or datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        role=role,
        text=text,
        tool_calls=tool_calls,
        parent_id=parent_id,
    )


@pytest.fixture
def interactions_two_sessions() -> list[RawInteraction]:
    """Four interactions across two sessions, chronological within each."""
    base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    return [
        # Session A — user then assistant
        _make_interaction(
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "project_a",
            "user",
            "Refactor the engine",
            timestamp=base,
        ),
        _make_interaction(
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "project_a",
            "assistant",
            "I will read the file first.",
            timestamp=base.replace(minute=1),
            tool_calls=(ToolCall(name="read", arguments='{"path":"src/engine.py"}'),),
            parent_id="msg-user-a",
        ),
        # Session B — user then assistant
        _make_interaction(
            "01BRZ3NDEKTSV4RRFFQ69G5FBV",
            "project_b",
            "user",
            "Fix the bug",
            timestamp=base,
        ),
        _make_interaction(
            "01BRZ3NDEKTSV4RRFFQ69G5FBV",
            "project_b",
            "assistant",
            "Can you provide the traceback?",
            timestamp=base.replace(minute=2),
            parent_id="msg-user-b",
        ),
    ]


@pytest.fixture
def interactions_with_orphan() -> list[RawInteraction]:
    """Three valid interactions plus one orphan (empty session_id)."""
    base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    return [
        _make_interaction(
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "project_a",
            "user",
            "Valid message",
            timestamp=base,
        ),
        _make_interaction(
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "project_a",
            "assistant",
            "Reply",
            timestamp=base.replace(minute=1),
        ),
        # Orphan: empty session_id is invalid for grouping.
        _make_interaction(
            "",
            "orphan_project",
            "user",
            "I have no session",
            timestamp=base,
        ),
    ]


# ---------------------------------------------------------------------------
# Test 1: empty iterable
# ---------------------------------------------------------------------------


class TestCorpusEmpty:
    def test_corpus_empty_iterable(self) -> None:
        """An empty list of interactions must produce an empty Corpus."""
        corpus: Corpus = Corpus.from_interactions([])

        assert corpus.sessions == ()
        assert corpus.issues == ()
        assert len(corpus) == 0


# ---------------------------------------------------------------------------
# Test 2: successful grouping into sessions
# ---------------------------------------------------------------------------


class TestCorpusGrouping:
    def test_corpus_successful_grouping(
        self, interactions_two_sessions: list[RawInteraction]
    ) -> None:
        """Four interactions (2 + 2 across two sessions) → 2 Session objects."""
        corpus: Corpus = Corpus.from_interactions(interactions_two_sessions)

        assert len(corpus) == 2
        assert len(corpus.sessions) == 2
        assert len(corpus.issues) == 0

        # Identify sessions by id
        session_a = corpus.session("01ARZ3NDEKTSV4RRFFQ69G5FAV")
        session_b = corpus.session("01BRZ3NDEKTSV4RRFFQ69G5FBV")

        assert session_a is not None
        assert session_b is not None

        # Session metadata
        assert session_a.project == "project_a"
        assert session_b.project == "project_b"

        # Each session has exactly 2 turns
        assert len(session_a.turns) == 2
        assert len(session_b.turns) == 2

        # Turns are in chronological order
        assert session_a.turns[0].text == "Refactor the engine"
        assert session_a.turns[0].role == "user"
        assert session_a.turns[1].text == "I will read the file first."
        assert session_a.turns[1].role == "assistant"
        assert session_a.turns[1].tool_calls == (
            ToolCall(name="read", arguments='{"path":"src/engine.py"}'),
        )

        # Parent id is carried through
        assert session_a.turns[1].parent_id == "msg-user-a"
        assert session_b.turns[1].parent_id == "msg-user-b"

    def test_corpus_turns_flat_iterator(
        self, interactions_two_sessions: list[RawInteraction]
    ) -> None:
        """corpus.turns() yields every Turn across all sessions."""
        corpus: Corpus = Corpus.from_interactions(interactions_two_sessions)
        flat: list[Turn] = list(corpus.turns())

        assert len(flat) == 4
        texts = [t.text for t in flat]
        assert "Refactor the engine" in texts
        assert "Fix the bug" in texts


# ---------------------------------------------------------------------------
# Test 3: orphan policy — no exceptions, issues as data
# ---------------------------------------------------------------------------


class TestCorpusOrphanPolicy:
    def test_corpus_orphan_policy(self, interactions_with_orphan: list[RawInteraction]) -> None:
        """An orphan interaction must not crash; it becomes a CorpusIssue."""
        corpus: Corpus = Corpus.from_interactions(interactions_with_orphan)

        # The two valid interactions still form a session.
        assert len(corpus) == 1
        assert len(corpus.sessions) == 1
        assert corpus.session("01ARZ3NDEKTSV4RRFFQ69G5FAV") is not None

        # Exactly one issue records the orphan.
        assert len(corpus.issues) == 1
        issue: CorpusIssue = corpus.issues[0]
        assert issue.kind == "orphan_session"
        assert issue.session_id == ""
        assert "orphan" in issue.detail.lower()


# ---------------------------------------------------------------------------
# Test 4: immutability — frozen dataclasses reject mutation
# ---------------------------------------------------------------------------


class TestCorpusImmutability:
    def test_corpus_sessions_cannot_be_reassigned(self) -> None:
        """Reassigning corpus.sessions must raise FrozenInstanceError."""
        corpus: Corpus = Corpus.from_interactions([])

        with pytest.raises((AttributeError, TypeError)):
            corpus.sessions = ()  # type: ignore[misc]

    def test_session_turns_cannot_be_reassigned(self) -> None:
        """Reassigning session.turns must raise FrozenInstanceError."""
        _ = Corpus.from_interactions([])
        # corpus is empty, so we construct a standalone Session for the test.
        session: Session = Session(
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            project="test",
            turns=(),
        )

        with pytest.raises((AttributeError, TypeError)):
            session.turns = ()  # type: ignore[misc]

    def test_turn_text_cannot_be_mutated(self) -> None:
        """Reassigning turn.text must raise FrozenInstanceError."""
        turn: Turn = Turn(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            role="user",
            text="hello",
            tool_calls=(),
            parent_id=None,
        )

        with pytest.raises((AttributeError, TypeError)):
            turn.text = "goodbye"  # type: ignore[misc]
