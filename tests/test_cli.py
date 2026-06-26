"""
Tests for src/cli.py — the Typer-based CLI orchestrator (Phase 7).

The CLI is the "dumb" glue: it wires Reader → Corpus → Scorer →
Analyzer → Reporter, handles argument parsing, and translates
domain errors into clean exit codes.

Exit code contract:
    0 — success
    1 — user / domain error (caught InsightError)
    2 — Typer / click argument error
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli import app
from src.errors import DatabaseError


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fresh CliRunner per test for isolation."""
    return CliRunner()


# ---------------------------------------------------------------------------
# Test 1: auto-detect fails when no DB path provided and default missing
# ---------------------------------------------------------------------------


class TestCLIDefaultAutoDetectFails:
    def test_cli_default_auto_detect_fails(self, cli_runner: CliRunner) -> None:
        """Invoking the CLI with no args must try auto-detect, fail, and exit 1."""
        with patch(
            "src.cli.load_config",
            return_value=MagicMock(
                api_key="",
                base_url="",
                model_id="",
                api_delay_seconds=0.0,
                max_sessions_to_analyze=20,
            ),
        ):
            with patch(
                "src.cli.resolve_db_path",
                side_effect=FileNotFoundError("Could not find standard location"),
            ):
                result = cli_runner.invoke(app, ["--api-key", "test-key"])

        assert result.exit_code == 1, (
            f"Expected exit code 1, got {result.exit_code}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )

        combined: str = (result.stdout + result.stderr).lower()
        assert "could not find standard location" in combined, (
            f"Expected FileNotFoundError message in output, got:\n{combined}"
        )


# ---------------------------------------------------------------------------
# Test 2: graceful domain error — no stacktrace, exit code 1
# ---------------------------------------------------------------------------


class TestCLIGracefulDomainError:
    def test_cli_graceful_domain_error(self, cli_runner: CliRunner) -> None:
        """A DatabaseError from the pipeline must be caught, not crash with a traceback."""
        with patch(
            "src.cli.resolve_db_path",
            side_effect=DatabaseError("simulated DB failure"),
        ):
            with patch(
                "src.cli.load_config",
                return_value=MagicMock(
                    api_key="",
                    base_url="",
                    model_id="",
                    api_delay_seconds=0.0,
                    max_sessions_to_analyze=20,
                ),
            ):
                result = cli_runner.invoke(app, ["--db-path", "/fake/db.sqlite"])

        # Domain errors map to exit code 1.
        assert result.exit_code == 1, (
            f"Expected exit code 1, got {result.exit_code}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )

        combined: str = result.stdout + result.stderr

        # The user-friendly error message is in the output.
        assert "simulated DB failure" in combined, (
            f"Expected the domain error message in output, got:\n{combined}"
        )

        # No full stacktrace — defence in depth for clean UX.
        assert "Traceback" not in combined, f"CLI leaked a stacktrace:\n{combined}"
        assert "raise " not in combined or "Traceback" not in combined


# ---------------------------------------------------------------------------
# Test 3: successful end-to-end execution (fully mocked pipeline)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Security audit A2: CLI must gracefully handle an empty --api-key
# instead of crashing with a raw stacktrace. The expected behaviour
# is exit code 0 and a partial report with 'pending' LLM dimensions.
# ---------------------------------------------------------------------------


class TestCLIEmptyApiKey:
    """The CLI must not crash when --api-key is empty.

    The empty string is the default when neither the flag nor the env
    vars (OPENAI_API_KEY, NIM_API_KEY) are set. A user running the
    CLI locally without configuring a key should still get a usable
    (partial) report, not a stacktrace.
    """

    def test_cli_empty_api_key_does_not_crash(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """An empty --api-key must exit 0 and produce a partial report.

        The audit found that an empty api_key crashes the CLI because
        the OpenAICompatibleClient is constructed unconditionally and
        the empty key propagates as an unhandled exception. The expected
        fix is graceful degradation: the CLI should detect the empty
        key, skip the LLM stage, and produce a heuristic-only report.
        """
        output_html: Path = tmp_path / "report.html"

        with (
            patch(
                "src.cli.resolve_db_path",
                return_value="/fake/db.sqlite",
            ),
            patch("src.cli.OpenCodeReader") as m_reader_cls,
            patch("src.cli.Corpus") as m_corpus_cls,
            patch("src.cli.Scorer") as m_scorer_cls,
            patch("src.cli.Analyzer") as m_analyzer_cls,
            patch("src.cli.HTMLReporter") as m_reporter_cls,
            patch("src.cli.OpenAICompatibleClient") as m_openai_cls,
            patch("src.cli.FallbackLLMClient") as m_fallback_llm_cls,
            patch("src.cli.load_config") as m_load_config,
        ):
            m_load_config.return_value = MagicMock(
                api_key="",
                base_url="",
                model_id="",
                api_delay_seconds=0.0,
                max_sessions_to_analyze=20,
            )
            m_reader_instance = MagicMock(name="OpenCodeReader")
            m_reader_instance.__enter__ = MagicMock(return_value=m_reader_instance)
            m_reader_instance.__exit__ = MagicMock(return_value=False)
            m_reader_instance.interactions = MagicMock(return_value=[])
            m_reader_cls.return_value = m_reader_instance

            m_corpus_instance = MagicMock(name="Corpus")
            m_corpus_cls.from_interactions = MagicMock(return_value=m_corpus_instance)

            m_scorer_instance = MagicMock(name="Scorer")
            m_scorer_instance.compute = MagicMock(return_value=())
            m_scorer_cls.return_value = m_scorer_instance

            m_analyzer_instance = MagicMock(name="Analyzer")
            m_analyzer_instance.enrich = MagicMock(return_value=())
            m_analyzer_instance.analyze_global = MagicMock(return_value=None)
            m_analyzer_cls.return_value = m_analyzer_instance

            m_reporter_instance = MagicMock(name="HTMLReporter")
            m_reporter_instance.render = MagicMock(return_value=None)
            m_reporter_cls.return_value = m_reporter_instance

            # Simulate the OpenAICompatibleClient rejecting the empty
            # api_key at construction time (the real openai SDK raises
            # an auth error on the first API call; here we simulate
            # that failure at the construction site). The audit
            # vulnerability: the CLI currently doesn't catch this,
            # so the whole run aborts with a stacktrace.
            m_openai_cls.side_effect = ValueError("api_key cannot be empty")
            m_fallback_llm_cls.return_value = MagicMock(name="FallbackLLMClient")

            result = cli_runner.invoke(
                app,
                [
                    "--db-path",
                    "/fake/db.sqlite",
                    "--api-key",
                    "",
                    "--output",
                    str(output_html),
                ],
            )

        # The CLI must exit cleanly (0), not crash.
        assert result.exit_code == 0, (
            f"Expected exit 0 for empty api_key, got {result.exit_code}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )

        # No raw stacktrace in the output.
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined, f"CLI leaked a stacktrace on empty api_key:\n{combined}"

        # The reporter was still called — a partial report is produced.
        m_reporter_instance.render.assert_called_once()


class TestCLISuccessfulExecution:
    def test_cli_successful_execution(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """A valid invocation must exit 0 and run the full pipeline in order."""
        output_html: Path = tmp_path / "report.html"

        # Call-order log: each side_effect appends its stage name and
        # returns the canned value. `unittest.mock._Call` objects compare
        # as tuples (NOT by call order), so the only reliable way to
        # verify sequence across different mocks is to record it
        # explicitly via side_effect.
        call_order: list[str] = []

        def _track(stage: str, return_value: object) -> object:
            def _side_effect(*args: object, **kwargs: object) -> object:
                call_order.append(stage)
                return return_value

            return _side_effect

        # The mocks below patch the names AS IMPORTED in src.cli.
        # If the CLI imports them under different aliases, these patches
        # will not intercept and the test will fail loudly — which is
        # the correct behaviour (defence in depth against silent misses).
        with (
            patch(
                "src.cli.resolve_db_path",
                side_effect=_track("resolve", "/fake/db.sqlite"),
            ) as m_resolve,
            patch("src.cli.OpenCodeReader") as m_reader_cls,
            patch("src.cli.Corpus") as m_corpus_cls,
            patch("src.cli.Scorer") as m_scorer_cls,
            patch("src.cli.Analyzer") as m_analyzer_cls,
            patch("src.cli.HTMLReporter") as m_reporter_cls,
            patch("src.cli.OpenAICompatibleClient") as m_openai_cls,
            patch("src.cli.FallbackLLMClient") as m_fallback_llm_cls,
            patch("src.cli.load_config") as m_load_config,
        ):
            m_load_config.return_value = MagicMock(
                api_key="",
                base_url="",
                model_id="",
                api_delay_seconds=0.0,
                max_sessions_to_analyze=20,
            )
            # Wire up the reader's context manager protocol.
            m_reader_instance = MagicMock(name="OpenCodeReader")
            m_reader_instance.__enter__ = MagicMock(return_value=m_reader_instance)
            m_reader_instance.__exit__ = MagicMock(return_value=False)
            m_reader_instance.interactions = MagicMock(
                side_effect=_track("reader", []),
            )
            m_reader_cls.side_effect = _track("reader_ctor", m_reader_instance)
            m_reader_cls.return_value = m_reader_instance

            # Corpus.from_interactions is a classmethod on the production class.
            m_corpus_instance = MagicMock(name="Corpus")
            m_corpus_cls.from_interactions = MagicMock(
                side_effect=_track("corpus", m_corpus_instance),
            )

            # Scorer().compute(corpus) returns a tuple of ScoreCards.
            m_scorer_instance = MagicMock(name="Scorer")
            m_scorer_instance.compute = MagicMock(
                side_effect=_track("scorer", ()),
            )
            m_scorer_cls.side_effect = _track("scorer_ctor", m_scorer_instance)
            m_scorer_cls.return_value = m_scorer_instance

            # Analyzer(client).enrich(corpus, cards) returns enriched cards.
            m_analyzer_instance = MagicMock(name="Analyzer")
            m_analyzer_instance.enrich = MagicMock(
                side_effect=_track("analyzer", ()),
            )
            m_analyzer_instance.analyze_global = MagicMock(
                side_effect=_track("analyze_global", None),
            )
            m_analyzer_cls.side_effect = _track("analyzer_ctor", m_analyzer_instance)
            m_analyzer_cls.return_value = m_analyzer_instance

            # HTMLReporter().render(cards, path) writes the report.
            m_reporter_instance = MagicMock(name="HTMLReporter")
            m_reporter_instance.render = MagicMock(
                side_effect=_track("reporter", None),
            )
            m_reporter_cls.side_effect = _track("reporter_ctor", m_reporter_instance)
            m_reporter_cls.return_value = m_reporter_instance

            # OpenAICompatibleClient is constructed; FallbackLLMClient.of()
            # returns the mock instance (T8: of() is a classmethod, not
            # a constructor call, so we mock the bound method).
            m_openai_instance = MagicMock(name="OpenAICompatibleClient")
            m_openai_cls.side_effect = _track("openai_client", m_openai_instance)
            m_openai_cls.return_value = m_openai_instance
            m_fallback_llm_instance = MagicMock(name="FallbackLLMClient")
            m_fallback_llm_cls.of = MagicMock(
                side_effect=_track("fallback_llm", m_fallback_llm_instance)
            )
            m_fallback_llm_cls.return_value = m_fallback_llm_instance

            result = cli_runner.invoke(
                app,
                [
                    "--db-path",
                    "/fake/db.sqlite",
                    "--api-key",
                    "test-key",
                    "--output",
                    str(output_html),
                ],
            )

        # The command exited cleanly.
        assert result.exit_code == 0, (
            f"Expected exit 0, got {result.exit_code}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )

        # Every stage of the pipeline was called exactly once.
        m_resolve.assert_called_once()
        m_reader_cls.assert_called_once()
        m_reader_instance.interactions.assert_called_once()
        m_corpus_cls.from_interactions.assert_called_once()
        m_scorer_cls.return_value.compute.assert_called_once()
        m_analyzer_cls.assert_called_once()  # constructed with the LLM client
        m_analyzer_instance.enrich.assert_called_once()
        m_analyzer_instance.analyze_global.assert_called_once()
        m_reporter_cls.assert_called_once()
        m_reporter_instance.render.assert_called_once()

        # The pipeline ran in the expected order. The constructor stages
        # (`*_ctor`) and the call stages are interleaved; the meaningful
        # sequence is:
        #   resolve -> reader -> corpus -> scorer
        #   -> openai_client -> fallback_llm -> analyzer -> reporter
        pipeline_stages: list[str] = [stage for stage in call_order if not stage.endswith("_ctor")]
        assert pipeline_stages == [
            "resolve",
            "reader",
            "corpus",
            "scorer",
            "openai_client",
            "fallback_llm",
            "analyzer",
            "analyze_global",
            "reporter",
        ], f"Pipeline ran out of order: {pipeline_stages!r}"
