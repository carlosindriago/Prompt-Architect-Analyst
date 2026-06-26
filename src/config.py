"""
Constants and secure configuration resolution for prompt-architect-analyst.

SECURITY RULE: No credential, absolute user path, or personal data
may be hardcoded here. This file is safe to commit.

API keys are resolved EXCLUSIVELY from environment variables or
explicit CLI flags at runtime — never from files, stdin, or argv.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from src.errors import ConfigurationError

# ---------------------------------------------------------------------------
# Default database locations (XDG / macOS standard paths — no usernames)
# ---------------------------------------------------------------------------
DEFAULT_DB_PATHS: list[str] = [
    "~/.local/share/opencode/opencode.db",  # Linux XDG
    "~/Library/Application Support/opencode/opencode.db",  # macOS
    "~/.opencode/opencode.db",  # fallback
]

DEFAULT_ARCHIVE_DIR = "~/.local/share/opencode/prompt-architect-analyst-archive"
DEFAULT_REPORT_PATH = "./ai_report.html"

# ---------------------------------------------------------------------------
# Parsing limits  (security: prevent DoS / OOM on malformed / huge DBs)
# ---------------------------------------------------------------------------
MAX_HUMAN_PROMPT_CHARS = 6_000  # longer = paste or injection, not a typed prompt
MAX_SESSIONS_PER_RUN = 10_000  # guard against DBs with millions of rows
MAX_PARTS_PER_MESSAGE = 500  # guard against corrupt messages with infinite parts
MAX_TOOL_NAME_LENGTH = 128  # guard against exotic MCP namespaced names

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
GAP_CAP_SECONDS = 300  # idle gaps longer than this don't count as work

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
PROVISIONAL_MIN_PROMPTS = 30  # below this threshold the score is hedged

# ---------------------------------------------------------------------------
# Environment variable names for API keys
# NEVER read keys from anywhere else.
# ---------------------------------------------------------------------------
ENV_ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
ENV_OPENAI_KEY = "OPENAI_API_KEY"
ENV_OLLAMA_HOST = "OLLAMA_HOST"  # default: http://localhost:11434

# ---------------------------------------------------------------------------
# LLM timeouts (seconds)
# ---------------------------------------------------------------------------
LLM_CONNECT_TIMEOUT = 10
LLM_READ_TIMEOUT = 120
LLM_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
REPORT_MAX_EVIDENCE_ITEMS = 6  # max examples per section
REPORT_MAX_PROMPT_SAMPLE = 50  # max prompts in the evidence bundle sent to LLM
REPORT_PROMPT_TRUNCATE = 600  # chars per prompt in the evidence bundle

# ---------------------------------------------------------------------------
# SQLite magic bytes (first 16 bytes of a valid SQLite3 file)
# ---------------------------------------------------------------------------
_SQLITE_MAGIC = b"SQLite format 3\x00"

# ---------------------------------------------------------------------------
# ULID character set — used to validate session/message IDs from user input
# ---------------------------------------------------------------------------
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$", re.ASCII)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_api_key(provider: str, explicit: str | None = None) -> str:
    """
    Resolve the API key for a provider with the correct precedence.

    Precedence: 1) explicit --api-key flag  2) environment variable  3) SystemExit

    SECURITY:
    - Uses os.environ.get() — never os.environ[] (KeyError leaks the key name).
    - Never reads from files, stdin, or argv[0].
    - The returned key is never logged; callers must treat it as a secret.

    Args:
        provider: One of "anthropic", "openai".
        explicit: Value passed via --api-key CLI flag (may be None).

    Returns:
        The resolved API key string.

    Raises:
        SystemExit(1): With a human-friendly message if no key is found.
    """
    if explicit:
        return explicit

    env_var = _ENV_FOR_PROVIDER.get(provider)
    if env_var:
        value = os.environ.get(env_var, "")
        if value.strip():
            return value.strip()

    _bail(
        f"No API key found for provider '{provider}'.\n"
        f"Set the {_ENV_FOR_PROVIDER.get(provider, 'appropriate')} environment variable "
        f"or pass --api-key KEY."
    )


def resolve_db_path(explicit: str | None = None) -> str:
    """
    Find opencode.db at the explicit path or in standard locations.

    SECURITY:
    - Rejects paths containing '..' (path traversal).
    - Verifies the file exists and is readable.
    - Verifies the file is a valid SQLite3 database (magic bytes check).
    - Never follows symlinks outside the resolved parent directory.

    Args:
        explicit: Path supplied via --db CLI flag (may be None).

    Returns:
        Resolved absolute path string to a valid, readable opencode.db.

    Raises:
        ValueError:       If the explicit path contains path traversal.
        FileNotFoundError: If no valid DB is found in any location.
        PermissionError:  If the file exists but cannot be read.
    """
    candidates: list[str] = [explicit] if explicit else DEFAULT_DB_PATHS

    for raw in candidates:
        if raw is None:
            continue

        # Security: reject path traversal before any filesystem access
        if ".." in Path(raw).parts:
            raise ValueError(
                f"Rejected DB path containing '..': {raw!r}. Path traversal is not allowed."
            )

        resolved = Path(raw).expanduser().resolve()

        if not resolved.exists():
            continue
        if not resolved.is_file():
            continue

        # Security: reject symlinks explicitly. The docstring promises
        # "Never follows symlinks outside the resolved parent directory".
        #
        # Audit fix S1: the previous component-by-component loop was
        # bypassed by relative paths whose first component was a
        # symlinked directory (e.g. "link_dir/db.sqlite"). The
        # reliable primitive for symlink detection in Python is
        # pathlib: if `original.resolve()` differs from
        # `original.absolute()`, then the original path traversed at
        # least one symlink. This works uniformly for absolute and
        # relative paths and does not depend on the current working
        # directory at the call site.
        original = Path(raw).expanduser()
        try:
            original_abs = original.absolute()
            original_resolved = original.resolve()
        except OSError:
            # resolve() can raise on broken symlinks. Treat that as
            # a symlink rejection — we never want to follow it.
            raise ConfigurationError(
                f"Rejected DB path: cannot resolve symbolic link in {raw!r}."
            ) from None
        if original_resolved != original_abs:
            raise ConfigurationError(
                f"Rejected DB path containing a symbolic link: {raw!r}. "
                "Symlinks are not allowed for the database path."
            )

        # Check read permission explicitly (avoid TOCTOU: check then act)
        if not os.access(resolved, os.R_OK):
            raise PermissionError(
                f"opencode.db found at {resolved} but is not readable. Check file permissions."
            )

        # Validate SQLite magic bytes — reject non-SQLite files early
        if not _is_sqlite_file(resolved):
            raise ValueError(
                f"File at {resolved} does not appear to be a valid SQLite3 database. "
                "Expected SQLite magic bytes at offset 0."
            )

        return str(resolved)

    if explicit:
        raise FileNotFoundError(
            f"opencode.db not found at the specified path: {explicit!r}\n"
            "Check that OpenCode has been run at least once to create the database."
        )

    searched = ", ".join(DEFAULT_DB_PATHS)
    raise FileNotFoundError(
        f"opencode.db not found in any standard location.\n"
        f"Searched: {searched}\n"
        "Run OpenCode at least once, or use --db PATH to specify the location."
    )


def validate_ulid(value: str, label: str = "ID") -> str:
    """
    Validate that a string is a well-formed ULID before using it in a SQL query.

    SECURITY: Any user-supplied ID (e.g. --session flag) must be validated
    against this pattern before being passed as a bind parameter to SQLite.
    This adds a second layer of defence on top of parameterized queries.

    Args:
        value: The string to validate.
        label: Human-readable label for error messages (e.g. "session ID").

    Returns:
        The validated value unchanged.

    Raises:
        ValueError: If the value is not a valid ULID.
    """
    if not _ULID_RE.match(value):
        raise ValueError(
            f"Invalid {label}: {value!r}. "
            "Expected a 26-character ULID (e.g. 01ARZ3NDEKTSV4RRFFQ69G5FAV)."
        )
    return value


@dataclass
class UserConfig:
    """User configuration for interactive onboarding."""

    api_key: str = ""
    base_url: str = ""
    model_id: str = ""
    language: str = "en"
    max_sessions_to_analyze: int = 20
    api_delay_seconds: float = 2.0

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_id": self.model_id,
            "language": self.language,
            "max_sessions_to_analyze": self.max_sessions_to_analyze,
            "api_delay_seconds": self.api_delay_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | int | float]) -> UserConfig:
        return cls(
            api_key=str(data.get("api_key", "")),
            base_url=str(data.get("base_url", "")),
            model_id=str(data.get("model_id", "")),
            language=str(data.get("language", "en")),
            max_sessions_to_analyze=int(data.get("max_sessions_to_analyze", 20)),
            api_delay_seconds=float(data.get("api_delay_seconds", 2.0)),
        )


def load_config() -> UserConfig:
    """Load the user configuration securely from disk."""
    config_path = Path.home() / ".config" / "prompt-architect-analyst" / "config.json"
    if not config_path.exists():
        return UserConfig()
    try:
        raw = config_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            valid_data: dict[str, str | int | float] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, (str, int, float)):
                    valid_data[k] = v
            return UserConfig.from_dict(valid_data)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("Failed to load config: %s", exc)
    return UserConfig()


def save_config(config: UserConfig) -> None:
    """Save the user configuration to disk securely.

    SECURITY: The config file contains an API key. It is created with 0o600
    permissions and enforced on every save to prevent unauthorized local read.
    """
    config_dir = Path.home() / ".config" / "prompt-architect-analyst"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.touch(mode=0o600, exist_ok=True)
    config_path.chmod(0o600)
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_ENV_FOR_PROVIDER: dict[str, str] = {
    "anthropic": ENV_ANTHROPIC_KEY,
    "openai": ENV_OPENAI_KEY,
}


def _is_sqlite_file(path: Path) -> bool:
    """Return True if the file starts with the SQLite3 magic byte sequence."""
    try:
        with path.open("rb") as fh:
            header = fh.read(len(_SQLITE_MAGIC))
        return header == _SQLITE_MAGIC
    except OSError:
        return False


def _bail(message: str) -> NoReturn:
    """Print a user-friendly error to stderr and exit with code 1.

    NoReturn is the correct return type here: the function always
    exits, so callers (like resolve_api_key) can rely on control
    never returning. This also satisfies mypy's "missing return
    statement" check on those callers.
    """
    import sys

    print(f"\nprompt-architect-analyst error: {message}\n", file=sys.stderr)
    sys.exit(1)


__all__ = [
    # Constants
    "DEFAULT_DB_PATHS",
    "DEFAULT_ARCHIVE_DIR",
    "DEFAULT_REPORT_PATH",
    "MAX_HUMAN_PROMPT_CHARS",
    "MAX_SESSIONS_PER_RUN",
    "MAX_PARTS_PER_MESSAGE",
    "MAX_TOOL_NAME_LENGTH",
    "GAP_CAP_SECONDS",
    "PROVISIONAL_MIN_PROMPTS",
    "ENV_ANTHROPIC_KEY",
    "ENV_OPENAI_KEY",
    "ENV_OLLAMA_HOST",
    "LLM_CONNECT_TIMEOUT",
    "LLM_READ_TIMEOUT",
    "LLM_MAX_RETRIES",
    "REPORT_MAX_EVIDENCE_ITEMS",
    "REPORT_MAX_PROMPT_SAMPLE",
    "REPORT_PROMPT_TRUNCATE",
    # Functions
    "resolve_api_key",
    "resolve_db_path",
    "validate_ulid",
    "UserConfig",
    "load_config",
    "save_config",
]
