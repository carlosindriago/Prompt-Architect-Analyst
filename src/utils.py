"""
Pure utility helpers for prompt-architect-analyst.

All functions are stateless and side-effect-free.
They may be called from any module without risk of circular imports.

SECURITY NOTES:
- scrub_paths() must be applied to ALL user-facing output before display.
- safe_json_loads() never logs the content of the string it failed to parse.
- run_fingerprint() uses SHA-256 — never MD5 or SHA-1.
- ts_from_ms() validates timestamp range to prevent crashes on corrupt DB data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import UTC, datetime
from typing import TypedDict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path scrubbing  (privacy: remove system username from any displayed string)
# ---------------------------------------------------------------------------

# Matches /home/<user> and /Users/<user> on Unix/macOS
_UNIX_HOME_RE = re.compile(r"(/home|/Users)/([^/\s\\\"']+)")
# Matches C:\Users\<user> on Windows
_WIN_HOME_RE = re.compile(r"(?i)C:\\Users\\([^\\\s\"']+)")


def scrub_paths(text: str) -> str:
    """
    Replace absolute paths containing a username with an anonymous form.

    Examples:
        /Users/carlos/projects  →  ~/projects
        /home/carlos/projects   →  ~/projects
        C:\\Users\\carlos\\proj  →  ~/proj

    IMPORTANT: Apply only to PRESENTATION output — never to the raw data
    used for scoring, so that scores remain deterministic and reproducible.

    Args:
        text: A string that may contain absolute paths.

    Returns:
        The string with usernames removed from paths.
    """
    # Unix / macOS: /home/user/... or /Users/user/...
    text = _UNIX_HOME_RE.sub(r"~", text)
    # Windows: C:\Users\user\...
    text = _WIN_HOME_RE.sub("~", text)
    return text


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

# Valid timestamp range: 2020-01-01 – 2100-01-01 in milliseconds
_TS_MIN_MS = 1_577_836_800_000
_TS_MAX_MS = 4_102_444_800_000


def ts_from_ms(unix_ms: int | float) -> datetime:
    """
    Convert a Unix millisecond timestamp to a UTC-aware datetime.

    SECURITY: Validates the timestamp is in a plausible range (2020–2100)
    to prevent crashes or nonsensical output from corrupt DB values.

    Args:
        unix_ms: Unix timestamp in milliseconds.

    Returns:
        A timezone-aware datetime in UTC.

    Raises:
        ValueError: If the timestamp is outside the plausible range.
    """
    unix_ms = int(unix_ms)
    if not (_TS_MIN_MS <= unix_ms <= _TS_MAX_MS):
        raise ValueError(
            f"Timestamp {unix_ms} is outside the expected range "
            f"({_TS_MIN_MS}–{_TS_MAX_MS} ms). The database may be corrupt."
        )
    return datetime.fromtimestamp(unix_ms / 1000.0, tz=UTC)


# ---------------------------------------------------------------------------
# Safe JSON parsing
# ---------------------------------------------------------------------------


def safe_json_loads(text: str, context: str = "") -> dict[str, str | float | int | bool | None]:
    """
    Parse a JSON string, returning an empty dict on failure.

    SECURITY: The content of `text` is NEVER included in log or error
    messages, because it may contain user prompt data.
    Only the `context` label (e.g. session ID) is logged.

    Args:
        text:    The raw JSON string to parse.
        context: A safe label for log messages (e.g. "message abc123").

    Returns:
        Parsed dict, or {} if parsing fails.
    """
    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            log.warning("Expected JSON object in %s, got %s", context, type(result).__name__)
            return {}
        return result
    except (json.JSONDecodeError, ValueError):
        log.warning("Invalid JSON in %s — skipping (content not logged)", context)
        return {}


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Restrict x to the closed interval [lo, hi].

    >>> clamp(1.5, 0.0, 1.0)
    1.0
    >>> clamp(-0.1, 0.0, 1.0)
    0.0
    """
    return max(lo, min(hi, x))


def squash(x: float, target: float) -> float:
    """
    Saturating curve: reaching `target` maxes the signal at 1.0.
    Values beyond `target` do not add further benefit.

    Formula: min(1.0, x / target)  with target > 0 guard.

    >>> squash(0.5, 1.0)
    0.5
    >>> squash(2.0, 1.0)
    1.0
    """
    if target <= 0:
        return 1.0
    return min(1.0, x / target)


def shrink(raw: float, n: int, full_n: int) -> float:
    """
    Confidence-weighted shrinkage: pull sparse scores toward 0.5.

    When n == 0 → returns 0.5 (fully uncertain).
    When n >= full_n → returns raw unchanged (full confidence).
    In between → linearly interpolates between 0.5 and raw.

    Args:
        raw:    Raw score in [0, 1].
        n:      Number of opportunities observed.
        full_n: Threshold for full confidence.

    Returns:
        Shrunk score in [0, 1].

    >>> shrink(1.0, 0, 10)
    0.5
    >>> shrink(1.0, 10, 10)
    1.0
    >>> shrink(1.0, 5, 10)
    0.75
    """
    if full_n <= 0:
        return raw
    confidence = clamp(n / full_n, 0.0, 1.0)
    return 0.5 + (raw - 0.5) * confidence


# ---------------------------------------------------------------------------
# Corpus fingerprinting
# ---------------------------------------------------------------------------


# TypedDict for run_fingerprint input. T1: replaces list[dict[str, Any]]
# with a precise schema; the function only reads the prompt text
# (and would crash on non-string values), so the structural type
# documents the contract.
class _PromptRecord(TypedDict, total=False):
    text: str
    timestamp: str
    session_id: str


def run_fingerprint(prompts: list[_PromptRecord]) -> str:
    """
    Compute a short SHA-256 fingerprint of the corpus prompt list.

    The fingerprint binds an LLM analysis to the exact data it was
    generated from. A different corpus — even from the same user —
    produces a different fingerprint, so stale analyses are rejected
    at report-render time.

    SECURITY: Uses hashlib.sha256 explicitly. MD5 and SHA-1 are
    prohibited (collision-vulnerable; could be exploited to forge
    fingerprint matches).

    Args:
        prompts: List of PromptRecord dicts from Corpus.real_prompts.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    payload = json.dumps(prompts, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Project label helper
# ---------------------------------------------------------------------------


def project_label(raw_path: str) -> str:
    """
    Convert an OpenCode project path to a display-safe label.

    OpenCode stores the project directory as an absolute path.
    This function extracts the final two path components (parent/name)
    and scrubs any username, producing a label safe for the report.

    Examples:
        /home/carlos/work/myapp   →  work/myapp
        /Users/carlos/projects/x  →  projects/x
        /srv/apps/backend         →  apps/backend

    Args:
        raw_path: The raw directory string from session.directory.

    Returns:
        A short, username-free display label.
    """
    if not raw_path:
        return "unknown"

    parts = [p for p in raw_path.replace("\\", "/").split("/") if p]

    # Drop username segments that are immediately after /home or /Users
    cleaned: list[str] = []
    skip_next = False
    for _i, part in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        if part.lower() in ("home", "users"):
            skip_next = True  # skip the username that follows
            continue
        cleaned.append(part)

    if not cleaned:
        return "unknown"

    # Return the last two segments for readability
    return "/".join(cleaned[-2:]) if len(cleaned) >= 2 else cleaned[-1]


# ---------------------------------------------------------------------------
# Shannon entropy (used by Toolcraft dimension)
# ---------------------------------------------------------------------------


def shannon_evenness(counts: dict[str, int]) -> float:
    """
    Compute Pielou's evenness index (Shannon entropy / log(S)).

    Returns a value in [0, 1]:
      0 = only one tool used
      1 = all tools used equally

    Returns 0.0 if there are fewer than 2 distinct tools.

    Args:
        counts: Mapping of tool name → usage count.
    """
    values = [v for v in counts.values() if v > 0]
    s = len(values)
    if s < 2:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    entropy = -sum((v / total) * math.log(v / total) for v in values)
    max_entropy = math.log(s)
    return entropy / max_entropy if max_entropy > 0 else 0.0


__all__ = [
    "scrub_paths",
    "ts_from_ms",
    "safe_json_loads",
    "clamp",
    "squash",
    "shrink",
    "run_fingerprint",
    "project_label",
    "shannon_evenness",
]
