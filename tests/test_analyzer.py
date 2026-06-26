from datetime import UTC, datetime

import pytest

from src.analyzer import Analyzer
from src.corpus import Corpus, Session, Turn
from src.llm import FakeLLMClient
from src.scorer import DimensionScore, ScoreCard


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(
        {
            "direction_score": 0.8,
            "direction_rationale": "Good",
            "verification_score": 0.8,
            "verification_rationale": "Good",
            "context_score": 0.8,
            "context_rationale": "Good",
            "iteration_score": 0.8,
            "iteration_rationale": "Good",
            "toolcraft_score": 0.8,
            "toolcraft_rationale": "Good",
            "workflow_level": "Profesional",
            "tips": ["Tip 1"],
        }
    )


@pytest.fixture
def fast_corpus() -> Corpus:
    sess = Session(
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "p",
        (Turn(timestamp=datetime.now(UTC), role="user", text="h", tool_calls=(), parent_id=None),),
    )
    return Corpus((sess,), ())


@pytest.fixture
def partial_scorecards() -> tuple[ScoreCard, ...]:
    return (
        ScoreCard(
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            dimensions=(
                DimensionScore("direction", 0.5, 1.0, "heuristic", "Good"),
                DimensionScore("verification", 0.5, 1.0, "heuristic", "Good"),
                DimensionScore("context", 0.5, 1.0, "heuristic", "Good"),
                DimensionScore("iteration", 0.5, 1.0, "heuristic", "Good"),
                DimensionScore("toolcraft", 0.5, 1.0, "heuristic", "Good"),
            ),
            overall=0.5,
            workflow_level=None,
            tips=(),
            corpus_issues=(),
            archetype="Builder",
        ),
    )


def test_analyzer_enriches_scorecard(fast_corpus, partial_scorecards, fake_llm):
    analyzer = Analyzer(client=fake_llm)
    enriched = analyzer.enrich(fast_corpus, partial_scorecards)
    card = enriched[0]
    assert card.dimension("direction").source == "llm"
    assert card.dimension("direction").score == 0.8


def test_analyzer_catches_generic_exception(fast_corpus, partial_scorecards):
    failing_llm = FakeLLMClient({}, exception=Exception("boom"))
    analyzer = Analyzer(client=failing_llm)
    enriched = analyzer.enrich(fast_corpus, partial_scorecards)
    card = enriched[0]
    assert card.dimension("direction").source == "heuristic"
    assert card.dimension("direction").score == 0.5


def test_analyzer_rejects_hallucinated_high_score(fast_corpus, partial_scorecards):
    hallucinated_llm = FakeLLMClient({"direction_score": 999.0})
    analyzer = Analyzer(client=hallucinated_llm)
    enriched = analyzer.enrich(fast_corpus, partial_scorecards)
    card = enriched[0]
    assert card.dimension("direction").source == "heuristic"
    assert card.dimension("direction").score == 0.5
