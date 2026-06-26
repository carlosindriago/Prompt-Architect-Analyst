"""
Scorer — deterministic scoring engine for the 5-dimension AI fluency model.

DESIGN INVARIANTS
- Pure Python, zero LLM calls, zero network I/O. Every metric is
  computable from the Corpus structure alone.
- The Scorer fills only the *heuristic* dimensions (speed, data_quality,
  debugging). Architecture and resolution are left as `pending` for
  the Analyzer (Phase 5) to enrich via LLM.
- All value objects are frozen with __slots__; mutation is rejected
  at runtime, so the Analyzer can safely extend a partial ScoreCard.
- `compute` returns one ScoreCard PER SESSION. A ScoreCard belongs
  to exactly one session_id; an aggregate card would violate the
  domain model and the Analyzer's contract.
- `overall` is None whenever any dimension is still pending — the
  CLI/reporter can use this to gate the final report.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

from src.corpus import Corpus, CorpusIssue, Session, Turn

Source = Literal["heuristic", "llm", "hybrid", "pending"]
_ALL_SOURCES: frozenset[str] = frozenset(get_args(Source))


# Patterns used by the debugging heuristic. Kept as a module-level
# frozenset so the regex compilation is done once.
_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Traceback", re.IGNORECASE),
    re.compile(r"\bError\b"),
    re.compile(r"\bException\b"),
    re.compile(r"\bTypeError\b"),
    re.compile(r"\bValueError\b"),
    re.compile(r"\bKeyError\b"),
    re.compile(r"\bAttributeError\b"),
    re.compile(r"\bImportError\b"),
)

# Resolution pattern is compiled once at module load (T3: was being
# recompiled on every call to _score_debugging). The word boundary
# is intentional — it avoids false positives like "unresolved".
_RESOLUTION_PATTERN: re.Pattern[str] = re.compile(
    r"\b(now it works|works now|fixed|resolved|solved|thanks|thank you)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Shrinkage Formula
# ---------------------------------------------------------------------------


def _shrinkage(raw_score: float, n_points: int, k: int = 3) -> float:
    """Shrinkage towards 0.5 for small sample sizes.

    Prevents assigning extreme scores (0 or 100) when the sample size
    is too small to be statistically significant.
    """
    if n_points == 0:
        return 0.5
    return (raw_score * n_points + 0.5 * k) / (n_points + k)


@dataclass(frozen=True, slots=True)
class DimensionScore:
    """One dimension of the fluency model.

    `score` and `confidence` are None when the dimension is pending
    (i.e. waiting for the LLM). `source` discriminates the origin so
    the reporter can label each dimension honestly.
    """

    name: str
    score: float | None
    confidence: float | None
    source: Source
    rationale: str

    def __post_init__(self) -> None:
        if self.source not in _ALL_SOURCES:
            raise ValueError(f"Invalid source: {self.source!r}")
        if self.source == "pending":
            if self.score is not None:
                raise ValueError("pending dimensions must have score=None")
            if self.confidence is not None:
                raise ValueError("pending dimensions must have confidence=None")
        else:
            if self.score is None or not 0.0 <= self.score <= 1.0:
                raise ValueError(
                    f"non-pending dimensions must have a score in [0, 1]; got {self.score!r}"
                )
            if self.confidence is None or not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    f"non-pending dimensions must have a confidence in [0, 1]; "
                    f"got {self.confidence!r}"
                )


@dataclass(frozen=True, slots=True)
class ScoreCard:
    """Immutable container of DimensionScore objects, one per dimension.

    The contract between Scorer (Phase 4) and Analyzer (Phase 5):
    the Scorer produces a partial card with `pending` placeholders,
    the Analyzer replaces the placeholders with LLM-derived verdicts.

    Each ScoreCard belongs to exactly one session_id.
    """

    session_id: str
    dimensions: tuple[DimensionScore, ...]
    overall: float | None
    workflow_level: str | None
    archetype: str | None
    tips: tuple[str, ...]
    corpus_issues: tuple[CorpusIssue, ...]

    def dimension(self, name: str) -> DimensionScore:
        """Lookup a dimension by name. Raises KeyError if absent."""
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError(f"Unknown dimension: {name!r}")


@dataclass(frozen=True, slots=True)
class PromptSection:
    """A personalized educational section of the ideal prompt."""

    header: str
    theory: str
    example: str


@dataclass(frozen=True, slots=True)
class GlobalAnalysis:
    """Global recommendations and security report across all sessions."""

    recommendations: tuple[str, ...]
    user_feedback: str | None
    ideal_prompt: tuple[PromptSection, ...] | None
    security_score: float | None
    security_rationale: str
    security_risks: tuple[str, ...]
    security_leaks: tuple[str, ...] = ()
    debug_raw_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class Scorer:
    """Deterministic scoring engine.

    Stateless: every call to `compute` is a pure function of the input
    Corpus. The class exists for symmetry with the future Analyzer
    (which will need configuration for LLM provider, model, etc.).
    """

    # Weights for the overall computation.
    weights: dict[str, float] | None = None

    def __post_init__(self):
        if self.weights is None:
            # Default weights derived from claude-insight model
            object.__setattr__(
                self,
                "weights",
                {
                    "Direction": 0.24,
                    "Verification": 0.22,
                    "Context": 0.22,
                    "Iteration": 0.18,
                    "Toolcraft": 0.14,
                },
            )

    def compute(self, corpus: Corpus) -> tuple[ScoreCard, ...]:
        """Compute one ScoreCard per Session in the Corpus.

        Each session is scored in isolation — heuristics are computed
        from that session's own Turn list, never aggregated across
        sessions. This preserves the domain invariant that a
        ScoreCard.session_id identifies a single conversation.

        An empty Corpus yields an empty tuple. A Corpus with N
        sessions yields N ScoreCards, in the same order as
        `corpus.sessions`.
        """
        return tuple(self._score_session(session, corpus.issues) for session in corpus.sessions)

    def _score_session(
        self,
        session: Session,
        corpus_issues: tuple[CorpusIssue, ...],
    ) -> ScoreCard:
        """Compute a single ScoreCard for one Session."""
        turns: tuple[Turn, ...] = session.turns

        direction: DimensionScore = _score_direction(turns)
        verification: DimensionScore = _score_verification(turns)
        context: DimensionScore = _score_context(turns)
        iteration: DimensionScore = _score_iteration(turns)
        toolcraft: DimensionScore = _score_toolcraft(turns)

        dimensions: tuple[DimensionScore, ...] = (
            direction,
            verification,
            context,
            iteration,
            toolcraft,
        )

        overall: float | None = _compute_overall(
            direction,
            verification,
            context,
            iteration,
            toolcraft,
            self.weights or {},
        )

        from src.archetypes import classify_archetype

        scores_dict = {
            "Direction": direction.score or 0.5,
            "Verification": verification.score or 0.5,
            "Context": context.score or 0.5,
            "Iteration": iteration.score or 0.5,
            "Toolcraft": toolcraft.score or 0.5,
        }
        archetype = classify_archetype(scores_dict)

        return ScoreCard(
            session_id=session.session_id,
            dimensions=dimensions,
            overall=overall,
            workflow_level=None,
            archetype=archetype,
            tips=(),
            corpus_issues=corpus_issues,
        )


# ---------------------------------------------------------------------------
# Private helpers — pure functions, no I/O, no LLM
# ---------------------------------------------------------------------------


def _placeholder(name: str) -> DimensionScore:
    """A dimension waiting for the Analyzer (Phase 5)."""
    return DimensionScore(
        name=name,
        score=None,
        confidence=None,
        source="pending",
        rationale="Awaiting LLM enrichment in Phase 5.",
    )


def _score_direction(turns: Iterable[Turn]) -> DimensionScore:
    turn_list = list(turns)
    user_turns = [t for t in turn_list if t.role == "user"]
    if not user_turns:
        return DimensionScore(
            name="direction",
            score=0.5,
            confidence=0.0,
            source="heuristic",
            rationale="No user turns.",
        )

    hits = 0
    for t in user_turns:
        text = t.text.lower()
        has_intent = any(
            w in text
            for w in ("create", "add", "fix", "implement", "change", "update", "must", "do not")
        )
        has_file = "/" in text or "." in text
        if has_intent or has_file:
            hits += 1

    raw_score = hits / len(user_turns)
    score = _shrinkage(raw_score, len(user_turns), k=3)
    return DimensionScore(
        name="direction",
        score=score,
        confidence=1.0,
        source="heuristic",
        rationale=f"{hits}/{len(user_turns)} turns had clear direction. Shrunk to {score:.2f}.",
    )


def _score_verification(turns: Iterable[Turn]) -> DimensionScore:
    turn_list = list(turns)
    write_turns = 0
    verification_hits = 0

    # We look for test/build commands
    verification_tools = ("run_command", "pytest", "npm test")

    i = 0
    while i < len(turn_list):
        t = turn_list[i]
        if any(
            tc.name in ("write_to_file", "replace_file_content", "multi_replace_file_content")
            for tc in t.tool_calls
        ):
            write_turns += 1
            # Check if within next 3 turns there is a verification
            verified = False
            for j in range(i + 1, min(i + 4, len(turn_list))):
                if any(
                    tc.name in verification_tools
                    or "test" in tc.arguments.lower()
                    or "build" in tc.arguments.lower()
                    for tc in turn_list[j].tool_calls
                ):
                    verified = True
                    break
                if turn_list[j].role == "user" and any(
                    w in turn_list[j].text.lower() for w in ("test", "verify", "check", "run")
                ):
                    verified = True
                    break
            if verified:
                verification_hits += 1
        i += 1

    if write_turns == 0:
        return DimensionScore(
            name="verification",
            score=0.5,
            confidence=0.0,
            source="heuristic",
            rationale="No write actions found.",
        )

    raw_score = verification_hits / write_turns
    score = _shrinkage(raw_score, write_turns, k=2)
    return DimensionScore(
        name="verification",
        score=score,
        confidence=1.0,
        source="heuristic",
        rationale=f"{verification_hits} verifications out of {write_turns} write bursts. Shrunk to {score:.2f}.",
    )


def _score_context(turns: Iterable[Turn]) -> DimensionScore:
    turn_list = list(turns)
    write_events = 0
    context_hits = 0
    read_tools = ("view_file", "read_file", "grep_search", "list_dir")
    write_tools = ("write_to_file", "replace_file_content", "multi_replace_file_content")

    has_read_recently = False
    for t in turn_list:
        if any(tc.name in read_tools for tc in t.tool_calls):
            has_read_recently = True

        if any(tc.name in write_tools for tc in t.tool_calls):
            write_events += 1
            if has_read_recently:
                context_hits += 1
            has_read_recently = False  # reset after write

    if write_events == 0:
        return DimensionScore(
            name="context",
            score=0.5,
            confidence=0.0,
            source="heuristic",
            rationale="No write actions found.",
        )

    raw_score = context_hits / write_events
    score = _shrinkage(raw_score, write_events, k=2)
    return DimensionScore(
        name="context",
        score=score,
        confidence=1.0,
        source="heuristic",
        rationale=f"{context_hits} context reads out of {write_events} writes. Shrunk to {score:.2f}.",
    )


def _score_iteration(turns: Iterable[Turn]) -> DimensionScore:
    turn_list = list(turns)
    user_turns = [t for t in turn_list if t.role == "user"]
    if not user_turns:
        return DimensionScore(
            name="iteration",
            score=0.5,
            confidence=0.0,
            source="heuristic",
            rationale="No user turns.",
        )

    good_iterations = 0
    for t in user_turns:
        # A good iteration provides details, not just "it failed"
        text = t.text.strip()
        if (
            len(text) > 40
            or "error" in text.lower()
            or "line" in text.lower()
            or "expected" in text.lower()
        ):
            good_iterations += 1

    raw_score = good_iterations / len(user_turns)
    score = _shrinkage(raw_score, len(user_turns), k=3)
    return DimensionScore(
        name="iteration",
        score=score,
        confidence=1.0,
        source="heuristic",
        rationale=f"{good_iterations}/{len(user_turns)} detailed user turns. Shrunk to {score:.2f}.",
    )


def _score_toolcraft(turns: Iterable[Turn]) -> DimensionScore:
    import math

    turn_list = list(turns)
    tool_counts: dict[str, int] = {}
    total_tools = 0
    agent_turns = 0
    user_micromanage = 0

    for t in turn_list:
        if t.role == "assistant" and t.tool_calls:
            agent_turns += 1
            for tc in t.tool_calls:
                tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
                total_tools += 1
        elif t.role == "user":
            text = t.text.lower()
            # Check for micromanagement "use grep", "read this file", "run this command"
            if any(w in text for w in ("use ", "run ", "grep ", "cat ", "view ", "search ")):
                user_micromanage += 1

    if total_tools == 0:
        return DimensionScore(
            name="toolcraft",
            score=0.5,
            confidence=0.0,
            source="heuristic",
            rationale="No tools used.",
        )

    # Shannon Evenness
    entropy = 0.0
    for count in tool_counts.values():
        p = count / total_tools
        entropy -= p * math.log(p)

    max_entropy = math.log(len(tool_counts)) if len(tool_counts) > 1 else 1.0
    evenness = entropy / max_entropy if max_entropy > 0 else 0.5

    # Delegation
    total_instructions = agent_turns + user_micromanage
    delegation = agent_turns / total_instructions if total_instructions > 0 else 0.5

    # Combine (50/50 split)
    raw_score = (evenness * 0.5) + (delegation * 0.5)
    score = _shrinkage(raw_score, total_tools, k=5)

    return DimensionScore(
        name="toolcraft",
        score=score,
        confidence=1.0,
        source="heuristic",
        rationale=f"Evenness: {evenness:.2f}, Delegation: {delegation:.2f}. Shrunk to {score:.2f}.",
    )


def _compute_overall(
    direction: DimensionScore,
    verification: DimensionScore,
    context: DimensionScore,
    iteration: DimensionScore,
    toolcraft: DimensionScore,
    weights: dict[str, float],
) -> float | None:
    dims = {
        "Direction": direction,
        "Verification": verification,
        "Context": context,
        "Iteration": iteration,
        "Toolcraft": toolcraft,
    }
    total_score = 0.0
    total_weight = 0.0

    for name, d in dims.items():
        if d.score is not None:
            w = weights.get(name, 0.2)
            total_score += d.score * w
            total_weight += w

    if total_weight == 0:
        return None
    return total_score / total_weight


__all__ = ["DimensionScore", "GlobalAnalysis", "ScoreCard", "Scorer", "Source"]
