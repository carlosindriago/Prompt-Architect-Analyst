from datetime import UTC, datetime

import pytest

from src.corpus import Corpus, Session, Turn
from src.scorer import Scorer


def _corpus_from_sessions(sessions: list[Session]) -> Corpus:
    return Corpus(sessions=tuple(sessions), issues=())


def test_scorer_basic():
    base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    t1 = Turn(timestamp=base, role="user", text="hello", tool_calls=(), parent_id=None)
    sess = Session("123", "p", (t1,))
    corpus = _corpus_from_sessions([sess])
    cards = Scorer().compute(corpus)
    assert len(cards) == 1
    assert cards[0].session_id == "123"


def test_scorecard_immutability():
    base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    t1 = Turn(timestamp=base, role="user", text="hello", tool_calls=(), parent_id=None)
    sess = Session("123", "p", (t1,))
    corpus = _corpus_from_sessions([sess])
    card = Scorer().compute(corpus)[0]
    with pytest.raises((AttributeError, TypeError)):
        card.overall = 0.99  # type: ignore
