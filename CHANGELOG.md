# Changelog

All notable changes to prompt-architect-analyst are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Pipeline: Phases 1–7 implemented end-to-end

The full 7-phase pipeline is now implemented, tested (177/177 passing),
and free of `mypy --strict` errors.

| Phase | Module | Status |
|---|---|---|
| 1. Config & secrets | `src/config.py` | ✅ |
| 2. Reader (SQLite, read-only) | `src/reader/opencode.py` | ✅ |
| 3. Corpus (immutable grouping) | `src/corpus.py` | ✅ |
| 4. Scorer (5 deterministic heuristics) | `src/scorer.py` | ✅ |
| 5A. LLM clients (Strategy + Fallback) | `src/llm.py` | ✅ |
| 5B. Analyzer (LLM enrichment) | `src/analyzer.py` | ✅ |
| 6. Reporter (HTML, Jinja2 autoescape) | `src/reporter.py` | ✅ |
| 7. CLI (Typer, graceful degradation) | `src/cli.py` | ✅ |

### Hotfix 1 — Critical security audit (4 findings)

- **A1** `LLMFallbackError` now inherits from `LLMProviderError` (transitive `InsightError`).
  The CLI's `except InsightError` now catches it and exits with code 1.
- **A2** CLI detects empty `api_key` and degrades to `FakeLLMClient({})` instead of
  constructing a broken `OpenAICompatibleClient`. Heuristics still produce a report.
- **A3** `Analyzer.enrich()` wraps `client.analyze()` in `try/except Exception` with
  `noqa: BLE001` and an explicit comment. On failure, the dimension stays "pending"
  and the loop continues with the next card.
- **S1** `resolve_db_path()` now uses the standard `Path.resolve() != Path.absolute()`
  idiom for symlink detection, working uniformly for absolute and relative paths.
  `OSError` on a broken symlink is mapped to `ConfigurationError`.

### Hotfix 2 — Security hardening (4 findings)

- **S2** `_extract_architecture` and `_extract_resolution` now validate
  `0.0 <= score <= 1.0` before constructing the `DimensionScore`. Hallucinated
  values like `999.0` no longer abort the pipeline.
- **S3** `HTMLReporter.render()` calls `os.chmod(output_path, 0o600)` after writing
  the file. The chmod is best-effort (`try/except OSError`) so Windows filesystems
  do not crash the report.
- **S4** `SensitiveDataFilter` now pre-formats `record.exc_info`, scrubs it via the
  same `_scrub()` used for `msg`, and assigns the result to `record.exc_text`. The
  original `exc_info` is cleared so the handler does not re-format and overwrite.
- **S5** `OpenAICompatibleClient` now caches the `openai.OpenAI` instance on the
  first call to `analyze()` and reuses it for subsequent calls. The cache lives in
  a declared dataclass field with `init=False`; `object.__setattr__` installs it
  under the frozen guard. Connection-pool exhaustion (S5) is closed.

### Hotfix 3 — Technical debt elimination (8 findings)

- **T1** 12 uses of `Any` eradicated. Replaced with `TypedDict` (`_MessageBucket`,
  `_ToolCallJson`, `_PromptRecord`) and `Protocol` (`_OpenAIModule`, `_OpenAIClient`,
  `_Message`, `_Choice`, `_Completion`) plus `TYPE_CHECKING` for `openai`.
- **T2** `Turn.tool_calls: tuple[ToolCall, ...]` (was `tuple[object, ...]`).
- **T3** `_RESOLUTION_PATTERN` promoted to a module-level constant in `scorer.py`,
  compiled once at import time.
- **T4** Redundant `cast(str, str(value))` removed from `_coerce_rationale` in
  `analyzer.py`. `str(value)` already returns a `str`.
- **T5** Dead `try/except` in `_parse_tool_call` removed (the inner `ToolCall()`
  constructor cannot raise the caught types).
- **T6** `OpenCodeReader.__exit__` now uses precise types:
  `type[BaseException] | None`, `BaseException | None`, `TracebackType | None`.
- **T7** `_score_speed` with an empty session now returns `0.5` (neutral) instead
  of `1.0`. New test `test_scorer_speed_empty_session_returns_neutral_0_5`
  enforces the contract.
- **T8** `FallbackLLMClient` no longer uses `object.__setattr__` in `__init__`.
  The `init=False` trick is replaced by a classmethod `FallbackLLMClient.of(*clients)`
  that delegates to the standard dataclass constructor. All call sites
  (`src/cli.py`, `tests/test_llm.py`, `tests/test_cli.py`) updated.

### External audit follow-up — zero remaining findings (N1, N2, N3)

An external auditor found three gaps in Hotfix 3. They are now closed:

- **N1** `src/logging_config.py` is now free of `mypy` errors without `# type: ignore`
  workarounds. The redundant `cast(TracebackType | None, tb)` in `_format_exception`
  is removed; `record.args = args_out` no longer carries an unused `# type: ignore`.
  `mypy src/` reports 0 errors.
- **N2** `_log_llm_failure` in `src/analyzer.py` now passes `str(exc)` to the logger
  instead of the raw exception object. This ensures the `SensitiveDataFilter`
  `_scrub_args` string path can redact any API key leaked in the exception message.
- **N3** Missing tests added for the Hotfix 2 security behaviours:
  - **S2** `test_analyzer_rejects_hallucinated_high_score` and
    `test_analyzer_rejects_negative_score` assert that out-of-range LLM scores
    leave the dimension pending.
  - **S3** `test_reporter_sets_owner_only_permissions` asserts
    `output.stat().st_mode & 0o777 == 0o600`.
  - **S4** `test_filter_scrubs_api_key_from_exception_traceback` builds a
    `LogRecord` with `exc_info`, runs `SensitiveDataFilter`, and asserts the
    resulting `exc_text` is scrubbed and `exc_info` is cleared.
  - **S5** `test_openai_client_is_reused_across_calls` mocks `openai.OpenAI`,
    calls `analyze()` twice, and asserts the constructor is invoked exactly once.
  - **T7** `test_scorer_speed_empty_session_returns_neutral_0_5` asserts an
    empty session scores `0.5`.

### Remaining `Any` instances (justified)

After Hotfix 3, **5 instances of `Any` remain in production code**. They are
intentional and bounded:

1. `src/logging_config.py`: `args_in: Any = record.args` — the stdlib types
   `LogRecord.args` as the private union `_LogRecordArgs`, which cannot be
   imported. We use `Any` for the narrow bidirectional assignment; the actual
   contract is enforced by `_scrub_args` at runtime.
2. `src/logging_config.py`: `args_out: Any = self._scrub_args(args_in)` —
   symmetrical to (1); the return value is immediately assigned back to the
   stdlib attribute.
3. `src/llm.py`: `_OpenAIModule.OpenAI(...) -> Any` — the `openai` SDK
   constructor signature is large and changes between releases. A Protocol
   return of `Any` is the pragmatic boundary; the returned object is only
   used through the narrower `_OpenAIClient` Protocol.
4. `src/llm.py`: `_OpenAIClient.chat: Any` — same reasoning as (3): the
   SDK's nested `chat.completions.create` chain is vendor-specific and
   unstable, so the Protocol exposes it as `Any` and we narrow usage at
   the call site via the `_Completion` Protocol.
5. `src/llm.py`: the `_MockChatResponse` test fixture and SDK-interaction
   layer inherently deal with an external dynamic API. The production code
   does not leak this `Any` into the `LLMClient.analyze()` return contract,
  which remains `dict[str, str | float]`.

No other `Any` remains in `src/`.

### Additional cleanup

- **A5** Deleted unused stubs `src/tools.py` and `src/archetype.py`.
- **Bonus** `mypy --strict` reports **0 errors**. The two pre-existing
  errors (in `config.py:87` and `logging_config.py:64`) are fixed:
  `_bail` returns `NoReturn`; `record.args` is narrowed via local `Any`.

### Migration notes

- Public API: `FallbackLLMClient(*clients)` is deprecated; use
  `FallbackLLMClient.of(*clients)` for variadic construction. The
  dataclass-style `FallbackLLMClient(clients=(...))` still works.
- All hotfix changes are backwards-compatible at the test layer.
  The suite grew from 163 to 177 tests, with every new test
  enforcing a security or typing contract.

### Verification snapshot

```
$ pytest tests/
============================= 177 passed in 3.40s ==============================

$ ruff check src/ tests/
All checks passed!

$ mypy src/ tests/
Success: no issues found in 26 source files
```

### CI

GitHub Actions workflow `.github/workflows/ci.yml` runs on every push to
`main` and every pull request. It installs dependencies with pip caching,
then fails fast in this order: format check, lint check, strict type check
(`mypy src/ tests/`), and finally `pytest`.

---
