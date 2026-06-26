# ruff: noqa: E501
"""
Tests for src/cli_onboarding.py
"""

from __future__ import annotations

from unittest import mock

import pytest

from src.errors import LLMProviderError


@pytest.fixture
def mock_fetch():
    with mock.patch("src.cli_onboarding.fetch_available_models") as m:
        m.return_value = ["model-1", "model-2", "model-3"]
        yield m


@pytest.fixture
def mock_verify():
    with mock.patch("src.cli_onboarding.verify_connection") as m:
        yield m


@pytest.fixture
def mock_save():
    with mock.patch("src.cli_onboarding.save_config") as m:
        yield m


def test_onboarding_success_flow(mock_fetch, mock_verify, mock_save):
    """Test a successful onboarding flow with NIM selected."""
    from src.cli_onboarding import run_onboarding

    # Provider 1 (NVIDIA NIM), skip base_url, api_key = "test-key", model choice 2 ("model-2"), language choice 1 ("en")
    with mock.patch("src.cli_onboarding.IntPrompt.ask", side_effect=[1, 2, 1]):
        with mock.patch("src.cli_onboarding.Prompt.ask", return_value="test-key"):
            config = run_onboarding()

    assert config.api_key == "test-key"
    assert config.base_url == "https://integrate.api.nvidia.com/v1"
    assert config.model_id == "model-2"
    assert config.language == "en"
    mock_save.assert_called_once_with(config)


def test_onboarding_custom_provider(mock_fetch, mock_verify, mock_save):
    """Test custom provider selection."""
    from src.cli_onboarding import run_onboarding

    # Provider 3 (Custom), base_url="http://localhost:8000/v1", api_key="sk-local", model 1, lang 1
    with mock.patch("src.cli_onboarding.IntPrompt.ask", side_effect=[3, 1, 1]):
        with mock.patch(
            "src.cli_onboarding.Prompt.ask", side_effect=["http://localhost:8000/v1", "sk-local"]
        ):
            config = run_onboarding()

    assert config.base_url == "http://localhost:8000/v1"
    assert config.api_key == "sk-local"
    assert config.model_id == "model-1"
    assert config.language == "en"
    mock_save.assert_called_once_with(config)


def test_onboarding_fetch_fails(mock_fetch, mock_save):
    """If fetch fails, the CLI should retry (we break with KeyboardInterrupt)."""
    from src.cli_onboarding import run_onboarding

    mock_fetch.side_effect = LLMProviderError("Connection refused")

    with mock.patch("src.cli_onboarding.IntPrompt.ask", side_effect=[1, KeyboardInterrupt]):
        with mock.patch("src.cli_onboarding.Prompt.ask", return_value="test-key"):
            with pytest.raises(KeyboardInterrupt):
                run_onboarding()

    mock_save.assert_not_called()


def test_onboarding_no_models(mock_fetch, mock_save):
    """If fetch returns empty list, the CLI should retry (break with KeyboardInterrupt)."""
    from src.cli_onboarding import run_onboarding

    mock_fetch.return_value = []

    with mock.patch("src.cli_onboarding.IntPrompt.ask", side_effect=[1, KeyboardInterrupt]):
        with mock.patch("src.cli_onboarding.Prompt.ask", return_value="test-key"):
            with pytest.raises(KeyboardInterrupt):
                run_onboarding()

    mock_save.assert_not_called()


def test_onboarding_verify_fails(mock_fetch, mock_verify, mock_save):
    """If verify_connection fails, the CLI should retry (we break with KeyboardInterrupt)."""
    from src.cli_onboarding import run_onboarding

    mock_verify.side_effect = LLMProviderError("Invalid API key")

    with mock.patch("src.cli_onboarding.IntPrompt.ask", side_effect=[1, 1, 1, KeyboardInterrupt]):
        with mock.patch("src.cli_onboarding.Prompt.ask", return_value="test-key"):
            with pytest.raises(KeyboardInterrupt):
                run_onboarding()

    mock_save.assert_not_called()
