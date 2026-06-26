# ruff: noqa: E501, S324
"""
Analyzer — enriches partial ScoreCards with LLM-derived verdicts.

DESIGN INVARIANTS
- Pure orchestration: the Analyzer does not compute metrics; it
  delegates heuristic computation to the Scorer and qualitative
  evaluation to the LLMClient.
- Immutable: enrich() returns NEW ScoreCard instances. The input
  tuple and its cards are never mutated (guaranteed by ScoreCard
  being frozen + dataclasses.replace()).
- Safe extraction: every LLM response value is read via dict.get().
  Missing keys, non-float scores, or non-string rationales leave
  the dimension pending rather than crashing.
- overall is the simple average of all 5 scores, or None if any
  dimension is still pending. This differs from the Scorer's
  weighted heuristic average; the Analyzer has the full picture.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from src.corpus import Corpus, Session
from src.llm import LLMClient
from src.llm_cache import LLMCache
from src.scorer import DimensionScore, GlobalAnalysis, ScoreCard

_logger: logging.Logger = logging.getLogger(__name__)

# Default confidence assigned to LLM-derived dimensions. The LLM
# response format does not include confidence; the Analyzer assigns
# a moderate default (0.8) reflecting "the LLM is usually right but
# not infallible". The reporter can surface this in the UI.
_LLM_CONFIDENCE: float = 0.8


@dataclass(frozen=True, slots=True)
class Analyzer:
    """Enriches partial ScoreCards by calling an LLM for pending dimensions.

    The LLMClient is held as a field so the same Analyzer can be reused
    across multiple enrich() calls without re-injecting the client.
    """

    client: LLMClient
    cache: LLMCache | None = None
    language: str = "en"
    api_delay_seconds: float = 0.0

    def enrich(
        self,
        corpus: Corpus,
        cards: tuple[ScoreCard, ...],
        on_progress: Callable[[], None] | None = None,
    ) -> tuple[ScoreCard, ...]:
        """Return new ScoreCards with pending dimensions filled from the LLM.

        For each card:
        - If no dimension is pending, return the card unchanged.
        - Otherwise, locate the corresponding Session in the Corpus,
          build a prompt, call the LLM, and update architecture /
          resolution where the response carries valid float scores.
        - If all 5 dimensions end up with scores, compute overall
          as a simple average; otherwise leave overall as None.
        """
        sessions_by_id: dict[str, Session] = {s.session_id: s for s in corpus.sessions}

        def process_card(card: ScoreCard) -> ScoreCard:
            try:
                if not _has_pending(card):
                    return card

                session: Session | None = sessions_by_id.get(card.session_id)
                if session is None:
                    # Session not found in corpus; leave the card unchanged.
                    return card

                turn_count = len(session.turns)
                response: dict[str, Any] | None = None

                # Try cache first
                if self.cache is not None:
                    prompt = _build_prompt(session, self.language)
                    import hashlib

                    fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    response = self.cache.get(card.session_id, turn_count, fingerprint=fingerprint)

                # If cache miss, call LLM
                if response is None:
                    if "prompt" not in locals():
                        prompt = _build_prompt(session, self.language)
                        import hashlib

                        fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

                    try:
                        response = self.client.analyze(prompt)
                        if self.api_delay_seconds > 0:
                            time.sleep(self.api_delay_seconds)
                        if self.cache is not None:
                            self.cache.set(
                                card.session_id, turn_count, response, fingerprint=fingerprint
                            )
                    except Exception as exc:  # noqa: BLE001 — graceful degradation
                        _log_llm_failure(card.session_id, exc)
                        if self.api_delay_seconds > 0:
                            time.sleep(self.api_delay_seconds)
                        return card

                new_dimensions: tuple[DimensionScore, ...] = _update_dimensions(
                    card.dimensions, response
                )
                new_overall: float | None = _compute_overall(new_dimensions)

                workflow_level_raw = response.get("workflow_level")
                workflow_level: str | None = (
                    workflow_level_raw if isinstance(workflow_level_raw, str) else None
                )

                tips_raw = response.get("tips", [])
                tips: tuple[str, ...] = (
                    tuple(str(t) for t in tips_raw) if isinstance(tips_raw, list) else ()
                )

                return replace(
                    card,
                    dimensions=new_dimensions,
                    overall=new_overall,
                    workflow_level=workflow_level,
                    tips=tips,
                )
            finally:
                if on_progress is not None:
                    on_progress()

        enriched: list[ScoreCard] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for result_card in executor.map(process_card, cards):
                enriched.append(result_card)

        return tuple(enriched)

    def analyze_global(self, corpus: Corpus, cards: tuple[ScoreCard, ...]) -> GlobalAnalysis:
        """Run a single LLM pass over the aggregated data of the entire corpus to generate a global report."""
        import hashlib
        import time
        from collections import Counter

        # Build a cache key based on the exact state of all sessions
        state_repr = "|".join(
            f"{s.session_id}:{len(s.turns)}"
            for s in sorted(corpus.sessions, key=lambda x: x.session_id)
        )
        state_hash = hashlib.md5(state_repr.encode("utf-8")).hexdigest()  # nosec B324

        cached = None
        if self.cache is not None:
            cached = self.cache.get("__global__", 0)
            if cached and cached.get("_state_hash") != state_hash:
                cached = None

        tool_counter: Counter[str] = Counter()
        for session in corpus.sessions:
            for turn in session.turns:
                for tc in turn.tool_calls:
                    tool_counter[tc.name] += 1
        tools_str = "\n".join(
            f"- {t_name} (used {count} times)" for t_name, count in tool_counter.most_common()
        )

        # 1. Security Analysis
        security_resp = None
        if cached:
            security_resp = cached.get("security")
        if security_resp is None:
            prompt = _build_security_prompt(corpus, tools_str, self.language)
            try:
                security_resp = self.client.analyze(prompt)
                if self.api_delay_seconds > 0:
                    time.sleep(self.api_delay_seconds)
            except Exception as exc:
                _logger.warning("Global security analysis failed: %s", exc)
                security_resp = {}

        # 2. Workflow Analysis
        workflow_resp = None
        if cached:
            workflow_resp = cached.get("workflow")
        if workflow_resp is None:
            prompt = _build_workflow_prompt(corpus, tools_str, self.language)
            try:
                workflow_resp = self.client.analyze(prompt)
                if self.api_delay_seconds > 0:
                    time.sleep(self.api_delay_seconds)
            except Exception as exc:
                _logger.warning("Global workflow analysis failed: %s", exc)
                workflow_resp = {}

        # 3. Ideal Prompt Analysis
        ideal_prompt_resp = None
        if cached:
            ideal_prompt_resp = cached.get("ideal_prompt")
        if ideal_prompt_resp is None:
            prompt = _build_ideal_prompt(corpus, self.language)
            try:
                ideal_prompt_resp = self.client.analyze(prompt)
                if self.api_delay_seconds > 0:
                    time.sleep(self.api_delay_seconds)
            except Exception as exc:
                _logger.warning("Global ideal prompt analysis failed: %s", exc)
                ideal_prompt_resp = {}

        if self.cache is not None and not cached:
            self.cache.set(
                "__global__",
                0,
                {
                    "_state_hash": state_hash,
                    "security": security_resp,
                    "workflow": workflow_resp,
                    "ideal_prompt": ideal_prompt_resp,
                },
            )

        recs_raw = workflow_resp.get("recommendations", [])
        recs = tuple(str(r) for r in recs_raw) if isinstance(recs_raw, list) else ()

        score_raw = security_resp.get("security_score")
        score = score_raw if isinstance(score_raw, float) and 0.0 <= score_raw <= 1.0 else None

        from src.scorer import PromptSection

        ideal_prompt_raw = ideal_prompt_resp.get("ideal_prompt_sections") or ideal_prompt_resp.get(
            "ideal_prompt"
        )

        # Sometimes models nest the array under another key if they get confused by the prompt format.
        if isinstance(ideal_prompt_raw, dict):
            for val in ideal_prompt_raw.values():
                if isinstance(val, list):
                    ideal_prompt_raw = val
                    break

        ideal_prompt = None
        if isinstance(ideal_prompt_raw, list) and len(ideal_prompt_raw) > 0:
            sections = []
            for item in ideal_prompt_raw:
                if isinstance(item, dict):
                    sections.append(
                        PromptSection(
                            header=str(item.get("header", "")),
                            theory=str(item.get("theory", "")),
                            example=str(item.get("example", "")),
                        )
                    )
            if sections:
                ideal_prompt = tuple(sections)

        user_feedback_raw = workflow_resp.get("user_feedback")
        user_feedback = str(user_feedback_raw) if user_feedback_raw else None

        rationale = _coerce_rationale(security_resp.get("security_rationale"))

        risks_raw = security_resp.get("security_risks", [])
        risks = tuple(str(r) for r in risks_raw) if isinstance(risks_raw, list) else ()

        leaks_raw = security_resp.get("security_leaks", [])
        leaks = tuple(str(leak) for leak in leaks_raw) if isinstance(leaks_raw, list) else ()

        import json

        debug_raw = json.dumps(ideal_prompt_resp, indent=2) if ideal_prompt_resp else None

        return GlobalAnalysis(
            recommendations=recs,
            user_feedback=user_feedback,
            ideal_prompt=ideal_prompt,
            security_score=score,
            security_rationale=rationale,
            security_risks=risks,
            security_leaks=leaks,
            debug_raw_prompt=debug_raw,
        )


# ---------------------------------------------------------------------------
# Private helpers — pure functions, no I/O
# ---------------------------------------------------------------------------


def _has_pending(card: ScoreCard) -> bool:
    """True if any dimension in the card needs LLM enrichment."""
    return any(d.source in ("pending", "heuristic", "hybrid") for d in card.dimensions)


def _build_prompt(session: Session, language: str) -> str:
    """Build a prompt from the session including truncated conversational turns."""
    from src.config import REPORT_PROMPT_TRUNCATE

    lines: list[str] = [
        f"Analyze the following session {session.session_id} on project {session.project!r} "
        f"which has {len(session.turns)} turns.\n"
    ]

    for idx, turn in enumerate(session.turns):
        text: str = turn.text[:REPORT_PROMPT_TRUNCATE]
        # In a real pipeline, we'd include tool calls here too, but text gives us a great start.
        lines.append(f"Turn {idx} [{turn.role}]: {text}")

    lines.append("\nEvaluate the conversation and return a JSON object with exactly these fields:")
    lines.append("- direction_score: float (0.0 to 1.0, precision of the user's instructions)")
    lines.append("- direction_rationale: string")
    lines.append("- verification_score: float (0.0 to 1.0, how well the user validates output)")
    lines.append("- verification_rationale: string")
    lines.append("- context_score: float (0.0 to 1.0, quality of file reading/context setup)")
    lines.append("- context_rationale: string")
    lines.append("- iteration_score: float (0.0 to 1.0, quality of step-by-step progress)")
    lines.append("- iteration_rationale: string")
    lines.append("- toolcraft_score: float (0.0 to 1.0, proficiency using AI tools)")
    lines.append("- toolcraft_rationale: string")
    lines.append("- workflow_level: string (must be exactly 'Novato', 'Profesional', or 'Senior')")
    lines.append("- tips: list of strings (3 to 5 actionable tips to improve the user's workflow)")
    lines.append("")
    lines.append(
        f"CRITICAL: You MUST output your analysis entirely in the '{language}' language (e.g. 'en', 'es', 'pt'). This applies to all text fields like rationale, tips, security_rationale, etc., EXCEPT for ideal_prompt which MUST ALWAYS be written in English for precision."
    )

    return "\n".join(lines)


def _build_security_prompt(corpus: Corpus, tools_str: str, language: str) -> str:
    lines: list[str] = [
        "You are a Senior Security Auditor analyzing an AI coding assistant workflow.",
        f"Total sessions analyzed: {len(corpus.sessions)}",
        "\nTools used across all sessions:",
        tools_str,
        "\nEvaluate the GLOBAL SECURITY risks of the user's workflow. Pay special attention to the tools used.",
        "Return a JSON object with EXACTLY these fields:",
        "- security_score: float (0.0 to 1.0, where 1.0 is completely secure, and lower means risks detected)",
        "- security_rationale: string (explain the security score in detail)",
        "- security_risks: list of strings (specific risks found, e.g. using dangerous tools like raw bash, executing untrusted code. Empty if none)",
        "- security_leaks: list of strings (CRITICAL: Look carefully for any exposed API keys, passwords, or secrets in the session metadata or tool usage. If found, describe what was leaked and in which session, partially obfuscating the secret for safety. Empty if none)",
        "\n" + f"CRITICAL: Output entirely in the '{language}' language.",
    ]
    return "\n".join(lines)


def _build_workflow_prompt(corpus: Corpus, tools_str: str, language: str) -> str:
    lines: list[str] = [
        "You are a Senior Software Architect analyzing a developer's workflow with an AI assistant.",
        f"Total sessions analyzed: {len(corpus.sessions)}",
        "\nTools used across all sessions:",
        tools_str,
        "\nEvaluate the GLOBAL workflow and architectural habits of the user.",
        "Return a JSON object with EXACTLY these fields:",
        "- recommendations: list of strings (3 to 5 high-level architectural/workflow recommendations for the user)",
        "- user_feedback: string (A direct, constructive feedback paragraph directed at the user. First, explicitly praise what they are currently doing right in their workflow or prompting. Then, emphasize clearly what they need to improve to reach a senior/architect level.)",
        "\n" + f"CRITICAL: Output entirely in the '{language}' language.",
    ]
    return "\n".join(lines)


def _build_ideal_prompt(corpus: Corpus, language: str) -> str:
    lines: list[str] = [
        "You are a Prompt Engineering Expert specializing in AI coding assistants.",
        "Based on the user's workflow, your task is to write the 'ideal prompt' they should use next time.",
        "Return a JSON object with EXACTLY this field:",
        "- ideal_prompt_sections: a list of exactly 7 JSON objects representing the sections of the ideal prompt.",
        "Each object in the list MUST have EXACTLY these fields:",
        "  - header: string (MUST be one of: '[SCENARIO]', '[ROLE]', '[CONTEXT]', '[TASK]', '[CONSTRAINTS]', '[FORMAT]', '[ACCEPTANCE]')",
        "  - theory: string (A personalized explanation addressed to the user detailing WHY this section is important, WHAT it is for, and HOW to write it properly. CRITICAL: Adapt this explanation to the user's specific workflow flaws or habits you analyzed. For example, if they act like it's a chatbot, explain why this section prevents that.)",
        "  - example: string (The actual fragment of the prompt for this section. CRITICAL: It MUST be a concrete, real-world professional software development example based on their actual project context, written in English.)",
        "CRITICAL: The 'theory' field MUST be written in the '{language}' language. The 'header' and 'example' fields MUST ALWAYS be written in English regardless of the requested language.",
    ]
    return "\n".join(lines)


def _update_dimensions(
    dimensions: tuple[DimensionScore, ...],
    response: dict[str, Any],
) -> tuple[DimensionScore, ...]:
    """Replace heuristic dimensions with LLM-derived values if available and valid."""
    new_dims: list[DimensionScore] = []
    for dim in dimensions:
        name = dim.name.lower()
        score_raw: str | float | None = response.get(f"{name}_score")
        if isinstance(score_raw, float) and 0.0 <= score_raw <= 1.0:
            rationale_raw: str | float | None = response.get(f"{name}_rationale", "")
            new_dims.append(
                DimensionScore(
                    name=dim.name,
                    score=score_raw,
                    confidence=_LLM_CONFIDENCE,
                    source="llm",
                    rationale=_coerce_rationale(rationale_raw),
                )
            )
        else:
            new_dims.append(dim)
    return tuple(new_dims)


def _coerce_rationale(value: str | float | None) -> str:
    """Coerce a rationale value to a string. None becomes empty string.

    T4: removed the redundant `cast(str, str(value))`. `str(value)`
    is already a `str`; the outer `cast` was dead code. The
    function signature narrows the input to `str | float | None`
    so we know the post-`str()` call returns a real string.
    """
    if value is None:
        return ""
    return str(value)


def _compute_overall(
    dimensions: tuple[DimensionScore, ...],
) -> float | None:
    """Simple average of all 5 scores, or None if any is missing."""
    scores: list[float] = []
    for d in dimensions:
        if d.score is None:
            return None
        scores.append(d.score)
    return sum(scores) / len(scores)


def _log_llm_failure(session_id: str, exc: BaseException) -> None:
    """Record a single LLM failure without leaking the exception message.

    The exception message may contain prompt content or API details;
    we log only the type and a short context so the report can still
    be generated (graceful degradation) while leaving a trail for
    post-mortem debugging.
    """
    _logger.warning(
        "LLM enrichment failed for session %s: %s: %s. Leaving heuristic scores untouched.",
        session_id,
        type(exc).__name__,
        str(exc),
    )


__all__ = ["Analyzer"]
