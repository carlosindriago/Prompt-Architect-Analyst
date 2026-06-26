# ruff: noqa: E501
"""
Interactive CLI onboarding for prompt-architect-analyst.
"""

from __future__ import annotations

import typer
from rich.columns import Columns
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

from src.config import UserConfig, save_config
from src.errors import LLMProviderError
from src.i18n import get_text
from src.llm import fetch_available_models, verify_connection

console = Console()

# Pre-defined providers for quick setup
PROVIDERS = [
    {
        "name": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
    },
    {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
    },
    {
        "name": "Custom OpenAI-compatible",
        "base_url": "",
    },
]


def run_onboarding(allow_cancel: bool = False, language: str = "en") -> UserConfig | None:
    """Run the interactive onboarding flow to configure the LLM."""
    console.print("\n[bold cyan]Welcome to prompt-architect-analyst![/bold cyan]")
    console.print("Let's configure your AI provider to analyze your OpenCode history.\n")

    while True:
        console.print("[bold]Select your AI Provider:[/bold]")
        if allow_cancel:
            console.print(f"  0. {get_text('cli_menu_cancel', language)}")
        for i, provider in enumerate(PROVIDERS, 1):
            console.print(f"  {i}. {provider['name']}")

        choices = [str(i) for i in range(1, len(PROVIDERS) + 1)]
        if allow_cancel:
            choices.insert(0, "0")

        provider_choice = IntPrompt.ask(
            "Choice",
            choices=choices,
            default=1,
        )

        if allow_cancel and provider_choice == 0:
            return None

        selected_provider = PROVIDERS[provider_choice - 1] if provider_choice != 0 else PROVIDERS[0]

        if selected_provider["base_url"]:
            base_url = selected_provider["base_url"]
        else:
            base_url = Prompt.ask("Enter the API Base URL (e.g. https://api.openai.com/v1)")

        api_key = Prompt.ask("Enter your API Key", password=True)

        console.print("\n[italic]Fetching available models...[/italic]")
        try:
            models = fetch_available_models(api_key, base_url)
        except LLMProviderError as e:
            console.print(f"\n[bold red]Error connecting to provider:[/bold red] {e}")
            console.print("Let's try again.\n")
            continue

        if not models:
            console.print("\n[bold red]No models found for this provider.[/bold red]")
            console.print("Let's try again.\n")
            continue

        console.print("\n[bold]Select a model to use:[/bold]")
        model_panels = [f"[{i}] {m}" for i, m in enumerate(models, 1)]
        console.print(Columns(model_panels, equal=True, expand=True))

        model_choice = IntPrompt.ask(
            "Model choice",
            choices=[str(i) for i in range(1, len(models) + 1)],
            default=1,
        )
        model_id = models[model_choice - 1]

        console.print("\n[italic]Verifying connection...[/italic]")
        try:
            verify_connection(api_key, base_url, model_id)
            console.print("[bold green]Connection verified successfully![/bold green]")
        except LLMProviderError as e:
            console.print(f"\n[bold red]Verification failed:[/bold red] {e}")
            console.print("Let's try again.\n")
            continue

        console.print(
            "\n[bold]Select your language / Selecciona tu idioma / Selecione seu idioma:[/bold]"
        )
        console.print("  1. English (en)")
        console.print("  2. Español (es)")
        console.print("  3. Português (pt)")

        lang_choice = IntPrompt.ask("Language choice", choices=["1", "2", "3"], default=1)
        lang_map = {1: "en", 2: "es", 3: "pt"}
        language = lang_map[lang_choice]

        config = UserConfig(
            api_key=api_key, base_url=base_url, model_id=model_id, language=language
        )

        try:
            save_config(config)
            console.print("\n[bold green]Configuration saved securely![/bold green]")
        except Exception as e:
            console.print(f"\n[bold red]Failed to save configuration:[/bold red] {e}")
            raise typer.Exit(1) from e

        return config


def run_main_menu(language: str = "en") -> str:
    """Show the interactive main menu and return the user's choice."""
    console.print("\n[bold cyan]AI Coding Insight[/bold cyan]")
    console.print(f"  1. {get_text('cli_menu_analyze', language)}")
    console.print(f"  2. {get_text('cli_menu_clear_cache', language)}")
    console.print(f"  3. {get_text('cli_menu_config', language)}")
    console.print(f"  4. {get_text('cli_menu_lang', language)}")
    console.print(f"  5. {get_text('cli_menu_exit', language)}")

    prompt_text = get_text("cli_menu_prompt", language)
    choice = IntPrompt.ask(prompt_text, choices=["1", "2", "3", "4", "5"], default=1)

    if choice == 1:
        return "analyze"
    elif choice == 2:
        return "clear_cache"
    elif choice == 3:
        return "reconfigure"
    elif choice == 4:
        return "change_lang"
    else:
        return "exit"
