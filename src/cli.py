# ruff: noqa: E501
"""
CLI entry point — the "dumb" orchestrator that wires every phase.

Exit code contract:
    0 — success
    1 — user / domain error (caught InsightError)
    2 — Typer / Click argument error

The CLI delegates ALL real work to the library modules. It only
catches domain errors and translates them into clean exit codes;
no business logic lives here.

PIPELINE
    Reader → Corpus → Scorer → Analyzer → Reporter
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.analyzer import Analyzer
from src.cli_onboarding import run_main_menu, run_onboarding
from src.config import UserConfig, load_config, resolve_api_key, resolve_db_path
from src.corpus import Corpus
from src.errors import InsightError
from src.i18n import get_text
from src.llm import FakeLLMClient, FallbackLLMClient, LLMClient, OpenAICompatibleClient
from src.llm_cache import LLMCache
from src.logging_config import setup_logging
from src.reader.opencode import OpenCodeReader
from src.reporter import HTMLReporter
from src.scorer import ScoreCard, Scorer

app: typer.Typer = typer.Typer(
    help="prompt-architect-analyst: Interactive AI coding fluency analyzer",
)

# Default NVIDIA NIM endpoint. Override with --base-url for OpenAI,
# Groq, Together, OpenRouter, or any other OpenAI-compatible provider.
_DEFAULT_BASE_URL: str = "https://integrate.api.nvidia.com/v1"


@app.callback(invoke_without_command=True)
def main_loop(
    ctx: typer.Context,
    db_path: str = typer.Option(  # noqa: B008
        None,
        "--db-path",
        help="Path to the opencode.db SQLite file. If omitted, searches default locations.",
    ),
    api_key: str = typer.Option(  # noqa: B008
        "",
        "--api-key",
        envvar=["OPENAI_API_KEY", "NIM_API_KEY"],
        help=("LLM provider API key. Falls back to config file or environment variables."),
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("report.html"),
        "--output",
        help="Output HTML report path.",
    ),
    model: str = typer.Option(  # noqa: B008
        "",
        "--model",
        help="LLM model name. Defaults to value in config or 'gpt-4o-mini'.",
    ),
    base_url: str = typer.Option(  # noqa: B008
        "",
        "--base-url",
        help=("API base URL. Defaults to value in config or NVIDIA NIM."),
    ),
    json_mode: bool = typer.Option(  # noqa: B008
        False,
        "--json",
        help="Output raw JSON to stdout instead of HTML report.",
    ),
) -> None:
    """prompt-architect-analyst: Interactive AI coding fluency analyzer."""
    if ctx.invoked_subcommand is not None:
        # If someone accidentally uses old subcommands (like `init`), ignore the main loop
        # and let Typer show the error or handle it.
        return

    setup_logging()

    user_config = load_config()

    # If any CLI flags are explicitly passed, we assume headless CI/CD execution.
    api_key_source = ctx.get_parameter_source("api_key")
    api_key_from_cli = api_key_source is not None and api_key_source.name == "COMMANDLINE"
    headless_mode = bool(db_path or api_key_from_cli or model or base_url or json_mode)

    if not headless_mode and not user_config.api_key:
        # First-time interactive setup
        new_config = run_onboarding(allow_cancel=False, language=user_config.language)
        if new_config is None:
            import sys

            sys.exit(0)
        user_config = new_config

    if headless_mode:
        _do_analyze(db_path, api_key, output, model, base_url, json_mode, user_config)
        return

    while True:
        choice = run_main_menu(user_config.language)
        if choice == "analyze":
            _do_analyze(db_path, api_key, output, model, base_url, json_mode, user_config)
            import rich.prompt

            rich.prompt.Prompt.ask("\n[cyan]Press Enter to return to menu...[/cyan]")
        elif choice == "reconfigure":
            new_config = run_onboarding(allow_cancel=True, language=user_config.language)
            if new_config:
                user_config = new_config
        elif choice == "clear_cache":
            LLMCache().clear()
            typer.secho(get_text("cli_cache_cleared", user_config.language), fg=typer.colors.GREEN)
        elif choice == "change_lang":
            from rich.prompt import IntPrompt

            from src.config import save_config

            prompt_text = f"0. {get_text('cli_menu_cancel', user_config.language)}\n" + get_text(
                "cli_select_lang", user_config.language
            )
            lang_choice = IntPrompt.ask(prompt_text, choices=["0", "1", "2", "3"], default=1)
            if lang_choice != 0:
                lang_map = {1: "en", 2: "es", 3: "pt"}
                user_config.language = lang_map[lang_choice]
                save_config(user_config)
        else:
            typer.secho("Exiting.", fg=typer.colors.CYAN)
            break


def _do_analyze(
    db_path: str | None,
    api_key: str,
    output: Path,
    model: str,
    base_url: str,
    json_mode: bool,
    user_config: UserConfig,
) -> None:
    """Perform the actual analysis pipeline."""
    if output == Path("report.html"):
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = Path(f"reports/report_{now}.html")

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        validated_path: str = resolve_db_path(db_path)
        with OpenCodeReader(validated_path) as reader:
            interactions = reader.interactions()

        corpus = Corpus.from_interactions(interactions)

        # Truncate corpus to the most recent max_sessions_to_analyze sessions
        if len(corpus.sessions) > user_config.max_sessions_to_analyze:
            # Sessions are sorted oldest to newest, so we take the tail
            truncated_sessions = corpus.sessions[-user_config.max_sessions_to_analyze :]
            corpus = Corpus(sessions=truncated_sessions, issues=corpus.issues)
            typer.secho(
                f"[yellow]Limiting analysis to the {user_config.max_sessions_to_analyze} most recent sessions.[/yellow]"
            )

        scorer: Scorer = Scorer()
        partial_cards = scorer.compute(corpus)

        # Corpus sessions are already sorted chronologically (oldest to newest).
        # We want the most recent cards first.
        session_order = {s.session_id: idx for idx, s in enumerate(corpus.sessions)}
        sorted_cards = sorted(
            partial_cards,
            key=lambda c: session_order.get(c.session_id, -1),
            reverse=True,
        )

        env_key: str = _probe_env_api_key()
        explicit_key: str = api_key.strip() or user_config.api_key

        resolved_model: str = model.strip() or user_config.model_id or "gpt-4o-mini"
        resolved_base_url: str = base_url.strip() or user_config.base_url or _DEFAULT_BASE_URL

        if explicit_key or env_key:
            resolved_key: str = resolve_api_key("openai", explicit=explicit_key or None)
            primary_client: LLMClient = OpenAICompatibleClient(
                api_key=resolved_key,
                model_id=resolved_model,
                base_url=resolved_base_url,
            )
            llm_client: LLMClient = FallbackLLMClient.of(primary_client)
        else:
            typer.secho(
                "No API key configured. Producing heuristic-only report "
                "(architecture and resolution will be marked pending).",
                fg=typer.colors.YELLOW,
            )
            llm_client = FallbackLLMClient.of(FakeLLMClient(canned_response={}))

        analyzer: Analyzer = Analyzer(
            client=llm_client,
            cache=LLMCache(),
            language=user_config.language,
            api_delay_seconds=user_config.api_delay_seconds,
        )

        preview_cards = tuple(sorted_cards[:3])
        rest_cards = tuple(sorted_cards[3:])

        from rich.console import Console

        console = Console()
        sessions_by_id = {s.session_id: s for s in corpus.sessions}

        # Determine how many preview cards are uncached to calculate real LLM speed
        preview_uncached_count = 0
        if analyzer.cache is not None:
            for c in preview_cards:
                s = sessions_by_id.get(c.session_id)
                if s and analyzer.cache.get(c.session_id, len(s.turns)) is None:
                    preview_uncached_count += 1
        else:
            preview_uncached_count = len(preview_cards)

        import time

        start_preview_time = time.time()
        if json_mode:
            enriched_preview = analyzer.enrich(corpus, preview_cards)
            preview_duration = time.time() - start_preview_time
        else:
            with console.status(
                f"[bold cyan]{get_text('cli_analyzing_preview', user_config.language)}[/bold cyan]"
            ):
                enriched_preview = analyzer.enrich(corpus, preview_cards)

            preview_duration = time.time() - start_preview_time
            # Show preliminary dashboard
            _print_preview_dashboard(enriched_preview, console, user_config.language)

        if rest_cards:
            total_rest = len(rest_cards)

            uncached_count = 0
            if analyzer.cache is not None:
                for c in rest_cards:
                    s = sessions_by_id.get(c.session_id)
                    if s and analyzer.cache.get(c.session_id, len(s.turns)) is None:
                        uncached_count += 1
            else:
                uncached_count = total_rest

            import math

            # Since ThreadPoolExecutor has 5 workers, we process up to 5 cards in parallel.
            # If we evaluated any uncached cards in preview, that took ~1 batch time.
            if preview_uncached_count > 0:
                time_per_batch = preview_duration
            else:
                time_per_batch = 3.0  # Default fallback if preview was fully cached

            batches = math.ceil(uncached_count / 5.0)
            estimated_seconds = int(batches * time_per_batch)

            if json_mode:
                enriched_rest = analyzer.enrich(corpus, rest_cards)
            else:
                from rich.panel import Panel

                console.print(
                    Panel(
                        f"[bold]{get_text('cli_stats_total', user_config.language)}[/bold] {total_rest} {get_text('html_sessions_analyzed', user_config.language).lower()}\n"
                        f"[bold]{get_text('cli_stats_cached', user_config.language)}[/bold] {total_rest - uncached_count}\n"
                        f"[bold]{get_text('cli_stats_new', user_config.language)}[/bold] {uncached_count}\n"
                        f"[bold]{get_text('cli_stats_eta', user_config.language)}[/bold] ~{estimated_seconds} {get_text('cli_stats_seconds', user_config.language)}",
                        title=f"[bold blue]{get_text('cli_stats_title', user_config.language)}[/bold blue]",
                        border_style="blue",
                        expand=False,
                    )
                )

                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    TimeRemainingColumn,
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task_id = progress.add_task(
                        f"[cyan]{get_text('cli_evaluating', user_config.language)}",
                        total=total_rest,
                    )

                    def advance():
                        progress.advance(task_id)

                    enriched_rest = analyzer.enrich(corpus, rest_cards, on_progress=advance)
        else:
            enriched_rest = ()

        all_enriched = enriched_preview + enriched_rest

        if json_mode:
            global_analysis = analyzer.analyze_global(corpus, all_enriched)
            import dataclasses
            import json

            # Simple recursive dataclass to dict function
            def _to_dict(obj):
                if dataclasses.is_dataclass(obj):
                    return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
                elif isinstance(obj, (list, tuple)):
                    return [_to_dict(v) for v in obj]
                else:
                    return obj

            output_data = {
                "global_analysis": _to_dict(global_analysis),
                "sessions": [_to_dict(c) for c in all_enriched],
            }
            print(json.dumps(output_data, indent=2))
            return

        with console.status(
            f"[bold magenta]{get_text('cli_global_report', user_config.language)}[/bold magenta]"
        ):
            global_analysis = analyzer.analyze_global(corpus, all_enriched)

        reporter: HTMLReporter = HTMLReporter()
        with console.status(
            f"[bold magenta]{get_text('cli_html_report', user_config.language)}[/bold magenta]"
        ):
            reporter.render(all_enriched, global_analysis, output, user_config.language)

    except InsightError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    typer.secho(
        f"{get_text('cli_done', user_config.language)} {output.resolve()}", fg=typer.colors.GREEN
    )


def _print_preview_dashboard(preview_cards: tuple[ScoreCard, ...], console, language: str) -> None:
    """Print a quick terminal dashboard for the preview cards."""
    from rich.panel import Panel
    from rich.table import Table

    if not preview_cards:
        return

    table = Table(title=get_text("cli_top_3_title", language), header_style="bold cyan")
    table.add_column(get_text("cli_session", language), style="dim")
    table.add_column(get_text("cli_level", language), justify="center")
    table.add_column(get_text("cli_score", language), justify="right")

    all_tips: list[str] = []

    for c in preview_cards:
        session_id = c.session_id[:8]
        level = c.workflow_level or get_text("cli_pending", language)
        score = f"{int(c.overall * 100)}" if c.overall is not None else "N/A"

        level_color = (
            "green" if level == "Senior" else "yellow" if level == "Profesional" else "red"
        )

        table.add_row(session_id, f"[{level_color}]{level}[/]", score)
        if c.tips:
            all_tips.extend(c.tips)

    console.print(table)

    # Deduplicate tips preserving order
    unique_tips = []
    seen = set()
    for tip in all_tips:
        if tip not in seen:
            unique_tips.append(tip)
            seen.add(tip)

    if unique_tips:
        # Show top 3 unique tips
        tips_text = "\n".join(f"[yellow]•[/] {tip}" for tip in unique_tips[:3])
        panel = Panel(
            tips_text,
            title=f"[bold green]{get_text('cli_tips_title', language)}[/bold green]",
            border_style="green",
            expand=False,
        )
        console.print(panel)

    console.print()


def _probe_env_api_key() -> str:
    """Return the first non-empty OPENAI_API_KEY or NIM_API_KEY, or empty.

    Used to decide whether to invoke resolve_api_key (which exits on
    failure) or skip directly to the graceful-degradation path. We
    avoid calling resolve_api_key when no key is present because that
    function raises SystemExit(1) by design when validation fails —
    not what we want for a soft-degradation UX.
    """
    import os

    for name in ("OPENAI_API_KEY", "NIM_API_KEY"):
        value: str = os.environ.get(name, "")
        if value.strip():
            return value.strip()
    return ""


def main() -> None:
    """Entry point for the CLI (used by pyproject.toml script)."""
    app()


if __name__ == "__main__":
    main()
