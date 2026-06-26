from pathlib import Path

from src.reporter import HTMLReporter
from src.scorer import DimensionScore, GlobalAnalysis, PromptSection, ScoreCard


def _card(session_id: str) -> ScoreCard:
    return ScoreCard(
        session_id=session_id,
        dimensions=(
            DimensionScore("direction", 0.8, 0.8, "llm", "Good"),
            DimensionScore("verification", 0.8, 0.8, "llm", "Good"),
            DimensionScore("context", 0.8, 0.8, "llm", "Good"),
            DimensionScore("iteration", 0.8, 0.8, "llm", "Good"),
            DimensionScore("toolcraft", 0.8, 0.8, "llm", "Good"),
        ),
        overall=0.8,
        workflow_level="Profesional",
        tips=("Use smaller functions",),
        corpus_issues=(),
        archetype="Builder",
    )


def _global() -> GlobalAnalysis:
    return GlobalAnalysis(
        recommendations=("Be safe",),
        user_feedback="Great job!",
        ideal_prompt=(PromptSection("H", "T", "E"),),
        security_score=0.9,
        security_rationale="Looks good",
        security_risks=(),
    )


def test_reporter_generates_file_successfully(tmp_path: Path):
    cards = (_card("session-1"),)
    output = tmp_path / "report.html"
    HTMLReporter().render(cards, _global(), output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_reporter_html_content_integrity(tmp_path: Path):
    cards = (_card("session-alpha"),)
    output = tmp_path / "report.html"
    HTMLReporter().render(cards, _global(), output)
    content = output.read_text(encoding="utf-8")
    assert "session-alpha" in content
    assert "direction" in content
