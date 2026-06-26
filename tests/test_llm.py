"""
Tests for src/llm.py — the LLM client layer (Strategy + Composite + Fallback).

These tests assume LLMClient, FakeLLMClient, FallbackLLMClient, and
LLMFallbackError will be implemented in src/llm.py. This file is the
RED phase of TDD: the import is expected to fail until the Green
phase lands.

Coverage:
- FakeLLMClient implements the LLMClient protocol and echoes its
  canned response.
- FallbackLLMClient returns the first client's result on success.
- FallbackLLMClient falls back to the next client on failure.
- FallbackLLMClient raises LLMFallbackError when every client fails.
- OpenAICompatibleClient wraps the official openai SDK, parses the
  JSON response, and wraps provider errors in the project's
  LLMProviderError / LLMResponseError hierarchy.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import openai
import pytest

from src.errors import LLMProviderError
from src.llm import (
    FakeLLMClient,
    FallbackLLMClient,
    LLMClient,
    LLMFallbackError,
    OpenAICompatibleClient,
    fetch_available_models,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AlwaysFailsClient:
    """A minimal stand-in LLMClient that always raises.

    Used to exercise the fallback path without depending on the
    production LLMClient implementation.
    """

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc: Exception = exc or RuntimeError("simulated LLM failure")
        self.call_count: int = 0

    def analyze(self, prompt: str) -> dict[str, str | float]:
        self.call_count += 1
        raise self._exc


class _CountingClient:
    """A client that records how many times analyze() was invoked.

    Useful for asserting that a fallback path actually delegates to
    the secondary client.
    """

    def __init__(self, response: dict[str, str | float]) -> None:
        self._response: dict[str, str | float] = response
        self.call_count: int = 0

    def analyze(self, prompt: str) -> dict[str, str | float]:
        self.call_count += 1
        return self._response


# ---------------------------------------------------------------------------
# Test 1: FakeLLMClient — Strategy stub
# ---------------------------------------------------------------------------


class TestFakeLLMClient:
    def test_fake_llm_client(self) -> None:
        """FakeLLMClient must implement LLMClient and return its canned response."""
        canned: dict[str, str | float] = {"score": 0.85, "rationale": "good architecture"}
        client: LLMClient = FakeLLMClient(canned)

        result: dict[str, str | float] = client.analyze("any prompt goes here")

        assert result == canned
        assert result is canned  # Fake returns the exact same object (deterministic)

    def test_fake_llm_client_satisfies_protocol(self) -> None:
        """FakeLLMClient must pass an isinstance() check against LLMClient."""
        client: LLMClient = FakeLLMClient({"score": 0.5})
        # runtime_checkable Protocol allows this assertion at test time.
        assert isinstance(client, LLMClient)

    def test_fake_llm_client_ignores_prompt(self) -> None:
        """The fake returns its canned response regardless of the prompt text."""
        canned: dict[str, str | float] = {"answer": 42}
        client: LLMClient = FakeLLMClient(canned)

        assert client.analyze("first prompt") == canned
        assert client.analyze("totally different prompt") == canned
        assert client.analyze("") == canned


# ---------------------------------------------------------------------------
# Test 2: FallbackLLMClient — success on the first client
# ---------------------------------------------------------------------------


class TestFallbackSuccess:
    def test_fallback_client_success_on_first(self) -> None:
        """When the first client succeeds, the fallback returns its result."""
        primary_response: dict[str, str | float] = {"source": "primary", "score": 0.9}
        primary: _CountingClient = _CountingClient(primary_response)

        secondary_response: dict[str, str | float] = {"source": "secondary", "score": 0.7}
        secondary: _CountingClient = _CountingClient(secondary_response)

        fallback: LLMClient = FallbackLLMClient.of(primary, secondary)

        result: dict[str, str | float] = fallback.analyze("prompt")

        assert result == primary_response
        # The secondary must NOT have been called when the primary succeeds.
        assert primary.call_count == 1
        assert secondary.call_count == 0


# ---------------------------------------------------------------------------
# Test 3: FallbackLLMClient — delegates to the fallback on failure
# ---------------------------------------------------------------------------


class TestFallbackDelegation:
    def test_fallback_client_uses_fallback(self) -> None:
        """When the first client raises, the fallback uses the second client."""
        failing: _AlwaysFailsClient = _AlwaysFailsClient(RuntimeError("primary is down"))
        recovery_response: dict[str, str | float] = {"source": "recovery", "score": 0.6}
        recovery: _CountingClient = _CountingClient(recovery_response)

        fallback: LLMClient = FallbackLLMClient.of(failing, recovery)

        result: dict[str, str | float] = fallback.analyze("prompt")

        assert result == recovery_response
        assert failing.call_count == 1
        assert recovery.call_count == 1

    def test_fallback_skips_multiple_failing_clients(self) -> None:
        """Two failures in a row still surface the third client's result."""
        first_fail: _AlwaysFailsClient = _AlwaysFailsClient(RuntimeError("first"))
        second_fail: _AlwaysFailsClient = _AlwaysFailsClient(RuntimeError("second"))
        third: _CountingClient = _CountingClient({"source": "third", "score": 0.4})

        fallback: LLMClient = FallbackLLMClient.of(first_fail, second_fail, third)

        result: dict[str, str | float] = fallback.analyze("prompt")

        assert result == {"source": "third", "score": 0.4}
        assert first_fail.call_count == 1
        assert second_fail.call_count == 1
        assert third.call_count == 1


# ---------------------------------------------------------------------------
# Test 4: FallbackLLMClient — all clients fail → LLMFallbackError
# ---------------------------------------------------------------------------


class TestFallbackAllFail:
    def test_fallback_client_all_fail(self) -> None:
        """When every client raises, the fallback raises LLMFallbackError."""
        exc1: RuntimeError = RuntimeError("provider 1 timeout")
        exc2: ConnectionError = ConnectionError("provider 2 unreachable")
        exc3: ValueError = ValueError("provider 3 bad response")

        fallback: LLMClient = FallbackLLMClient.of(
            _AlwaysFailsClient(exc1),
            _AlwaysFailsClient(exc2),
            _AlwaysFailsClient(exc3),
        )

        with pytest.raises(LLMFallbackError):
            fallback.analyze("prompt")

    def test_fallback_error_preserves_underlying_causes(self) -> None:
        """LLMFallbackError must chain the original exceptions (raise ... from)."""
        first_exc: RuntimeError = RuntimeError("primary boom")
        second_exc: RuntimeError = RuntimeError("secondary boom")

        fallback: LLMClient = FallbackLLMClient.of(
            _AlwaysFailsClient(first_exc),
            _AlwaysFailsClient(second_exc),
        )

        with pytest.raises(LLMFallbackError) as exc_info:
            fallback.analyze("prompt")

        # At least one of the original exceptions must be reachable via __cause__
        # or __context__ (the last `raise ... from exc` sets __cause__).
        cause_chain: list[BaseException] = []
        current: BaseException | None = exc_info.value
        while current is not None:
            cause_chain.append(current)
            current = current.__cause__ or current.__context__

        assert any(e is first_exc or e is second_exc for e in cause_chain), (
            f"Neither original exception found in chain: {cause_chain!r}"
        )

    def test_fallback_empty_client_list_raises(self) -> None:
        """An empty fallback list is a configuration error, not a silent pass."""
        with pytest.raises(LLMFallbackError):
            FallbackLLMClient.of().analyze("prompt")


# ---------------------------------------------------------------------------
# Security audit A1: LLMFallbackError must inherit from InsightError so the
# CLI's `except InsightError` catch block can translate it into a clean
# exit code 1 instead of leaking an unhandled Exception.
# ---------------------------------------------------------------------------


class TestLLMFallbackErrorHierarchy:
    """The CLI catches `InsightError` to produce exit code 1.

    If `LLMFallbackError` inherits from a bare `Exception`, it bypasses
    that catch and crashes the CLI with a raw stacktrace — exactly the
    UX regression the graceful-error test guards against.
    """

    def test_llm_fallback_error_inherits_from_insight_error(self) -> None:
        """LLMFallbackError MUST be a subclass of InsightError (transitively)."""
        from src.errors import InsightError

        assert issubclass(LLMFallbackError, InsightError), (
            "LLMFallbackError must inherit from InsightError so the CLI's "
            "except InsightError clause can translate it to exit code 1. "
            f"Current MRO: {LLMFallbackError.__mro__!r}"
        )

    def test_llm_fallback_error_inherits_from_llm_provider_error(self) -> None:
        """LLMFallbackError MUST also be a subclass of LLMProviderError.

        LLMProviderError extends InsightError, so satisfying this check
        guarantees the previous one. It also lets the reporter/CLI
        distinguish LLM-specific failures from other InsightErrors
        (e.g. DatabaseError) when needed.
        """
        from src.errors import LLMProviderError

        assert issubclass(LLMFallbackError, LLMProviderError), (
            "LLMFallbackError must inherit from LLMProviderError so the "
            "LLM subsystem has a single error category. "
            f"Current MRO: {LLMFallbackError.__mro__!r}"
        )

    def test_fallback_error_is_caught_by_insight_error_handler(self) -> None:
        """An `except InsightError` clause MUST catch LLMFallbackError.

        This is the user-facing contract: when every LLM provider fails,
        the CLI exits cleanly with code 1 and a human-readable message,
        not with a raw traceback.
        """
        from src.errors import InsightError

        failing = _AlwaysFailsClient(RuntimeError("downstream outage"))
        fallback = FallbackLLMClient.of(failing, failing, failing)

        # The `except InsightError` clause catches LLMFallbackError
        # ONLY IF LLMFallbackError inherits (transitively) from it.
        with pytest.raises(InsightError):
            try:
                fallback.analyze("prompt")
            except InsightError:
                # The CLI's catch block runs here; re-raise so pytest
                # sees the same exception type and the assertion holds.
                raise
            except LLMFallbackError:
                # If we reach this branch, the audit vulnerability is
                # confirmed: LLMFallbackError is NOT an InsightError.
                pytest.fail(
                    "LLMFallbackError escaped the `except InsightError` "
                    "handler — the CLI would crash with a stacktrace."
                )


# ---------------------------------------------------------------------------
# Test 5: OpenAICompatibleClient — production HTTP client (Phase 8)
# ---------------------------------------------------------------------------


class _MockChatResponse:
    """Minimal stand-in for openai.types.chat.ChatCompletion.

    Only the attributes the client reads are implemented; everything
    else is irrelevant for the unit test.
    """

    def __init__(self, content: str) -> None:
        self.choices: list[_MockChoice] = [_MockChoice(content)]


class _MockChoice:
    def __init__(self, content: str) -> None:
        self.message: _MockMessage = _MockMessage(content)


class _MockMessage:
    def __init__(self, content: str) -> None:
        self.content: str = content


class TestOpenAICompatibleClient:
    """Unit tests for the production OpenAICompatibleClient.

    These tests assume OpenAICompatibleClient is implemented in
    src/llm.py and that it:

    1. Accepts (api_key: str, model_id: str, base_url: str) in the
       constructor (base_url defaults to NVIDIA NIM).
    2. Calls openai.OpenAI(...).chat.completions.create(...) with
       response_format={"type": "json_object"} so the LLM is forced
       to return parseable JSON.
    3. Parses the JSON content of the first choice's message.
    4. Wraps openai.APIError and openai.APIConnectionError in
       LLMProviderError; wraps json.JSONDecodeError in
       LLMResponseError — both subclasses of LLMProviderError so
       FallbackLLMClient catches them.

    All HTTP I/O is mocked. No real network calls are made.
    """

    def test_successful_api_call_returns_parsed_dict(self) -> None:
        """A well-formed JSON response is parsed and returned as a dict."""
        canned_json: str = (
            '{"architecture_score": 0.9, "resolution_score": 0.8, '
            '"extra_key": "ignored_by_typed_protocol"}'
        )
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.chat.completions.create.return_value = _MockChatResponse(canned_json)
            mock_openai_cls.return_value = mock_client

            client = OpenAICompatibleClient(
                api_key="test-key",
                model_id="gpt-4o-mini",
            )
            result: dict[str, str | float] = client.analyze("test prompt")

        # The parsed dict matches the JSON content. Extra keys (beyond
        # what the Analyzer cares about) are passed through unchanged;
        # the client is a thin wrapper, the Analyzer does the filtering.
        assert result == {
            "architecture_score": 0.9,
            "resolution_score": 0.8,
            "extra_key": "ignored_by_typed_protocol",
        }

        # The HTTP call was made exactly once with the expected args.
        # base_url defaults to NVIDIA NIM when not provided.
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
        )
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]
        # JSON output is requested so the LLM is forced to return parseable JSON.
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_successful_api_call_with_custom_base_url(self) -> None:
        """A custom base_url (e.g. OpenAI) overrides the NVIDIA NIM default."""
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.chat.completions.create.return_value = _MockChatResponse(
                '{"architecture_score": 0.5}'
            )
            mock_openai_cls.return_value = mock_client

            client = OpenAICompatibleClient(
                api_key="openai-key",
                model_id="gpt-4o",
                base_url="https://api.openai.com/v1",
            )
            _ = client.analyze("p")

        mock_openai_cls.assert_called_once_with(
            api_key="openai-key",
            base_url="https://api.openai.com/v1",
        )

    def test_api_error_is_wrapped_in_llm_provider_error(self) -> None:
        """An openai.APIError is wrapped in LLMProviderError so the Fallback catches it."""
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.chat.completions.create.side_effect = openai.APIError(
                message="rate limit exceeded",
                request=MagicMock(name="Request"),
                body=None,
            )
            mock_openai_cls.return_value = mock_client

            client = OpenAICompatibleClient(api_key="test-key", model_id="gpt-4o-mini")
            with pytest.raises(LLMProviderError) as exc_info:
                client.analyze("p")

        # The original openai.APIError is chained via __cause__ for debuggability.
        assert isinstance(exc_info.value.__cause__, openai.APIError)

    def test_connection_error_is_wrapped_in_llm_provider_error(self) -> None:
        """An openai.APIConnectionError is wrapped in LLMProviderError."""
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
                request=MagicMock(name="Request")
            )
            mock_openai_cls.return_value = mock_client

            client = OpenAICompatibleClient(api_key="test-key", model_id="gpt-4o-mini")
            with pytest.raises(LLMProviderError) as exc_info:
                client.analyze("p")

        assert isinstance(exc_info.value.__cause__, openai.APIConnectionError)

    def test_malformed_json_is_wrapped_in_llm_response_error(self) -> None:
        """A non-JSON response is wrapped in LLMResponseError (subclass of LLMProviderError)."""
        from src.errors import LLMResponseError

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.chat.completions.create.return_value = _MockChatResponse("not json at all")
            mock_openai_cls.return_value = mock_client

            client = OpenAICompatibleClient(api_key="test-key", model_id="gpt-4o-mini")
            with pytest.raises(LLMResponseError) as exc_info:
                client.analyze("p")

        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)

    def test_wrapped_error_is_caught_by_fallback(self) -> None:
        """End-to-end: a failing OpenAICompatibleClient is caught by FallbackLLMClient."""
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_failing_client = MagicMock(name="openai.OpenAI")
            mock_failing_client.chat.completions.create.side_effect = openai.APIConnectionError(
                request=MagicMock(name="Request")
            )
            mock_openai_cls.return_value = mock_failing_client

            failing = OpenAICompatibleClient(api_key="bad-key", model_id="gpt-4o-mini")
            # The Fallback catches the wrapped LLMProviderError and tries
            # the next client. The second client succeeds.
            fallback = FallbackLLMClient.of(
                failing,
                FakeLLMClient({"architecture_score": 0.7}),
            )
            result: dict[str, str | float] = fallback.analyze("p")

        assert result == {"architecture_score": 0.7}


# ---------------------------------------------------------------------------
# Security audit S5: OpenAICompatibleClient must reuse the openai.OpenAI
# instance across multiple analyze() calls instead of creating a new one
# each time (socket / connection pool leak).
# ---------------------------------------------------------------------------


class TestOpenAICompatibleClientCachesSdkInstance:
    def test_openai_client_is_reused_across_calls(self) -> None:
        """Calling analyze() twice must construct openai.OpenAI only once."""
        canned_json: str = '{"architecture_score": 0.9}'

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.chat.completions.create.return_value = _MockChatResponse(canned_json)
            mock_openai_cls.return_value = mock_client

            client = OpenAICompatibleClient(
                api_key="test-key",
                model_id="gpt-4o-mini",
            )

            result1 = client.analyze("first prompt")
            result2 = client.analyze("second prompt")

        # The SDK class constructor is invoked exactly once (lazy cache).
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
        )
        # Both calls return parsed results.
        assert result1 == {"architecture_score": 0.9}
        assert result2 == {"architecture_score": 0.9}
        # The underlying completion endpoint was invoked twice.
        assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Test 6: fetch_available_models
# ---------------------------------------------------------------------------


class _MockModelData:
    def __init__(self, id: str) -> None:
        self.id = id


class _MockModelResponse:
    def __init__(self, data: list[_MockModelData]) -> None:
        self.data = data


class TestFetchAvailableModels:
    def test_fetch_available_models_success(self) -> None:
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.models.list.return_value = _MockModelResponse(
                [
                    _MockModelData("z-model"),
                    _MockModelData("a-model"),
                ]
            )
            mock_openai_cls.return_value = mock_client

            models = fetch_available_models("test-key", "https://api.openai.com/v1")

        # Models should be returned sorted alphabetically
        assert models == ["a-model", "z-model"]
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.openai.com/v1"
        )

    def test_fetch_available_models_api_error(self) -> None:
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.models.list.side_effect = openai.APIError(
                message="auth failed",
                request=MagicMock(),
                body=None,
            )
            mock_openai_cls.return_value = mock_client

            with pytest.raises(LLMProviderError) as exc_info:
                fetch_available_models("bad-key")

        assert isinstance(exc_info.value.__cause__, openai.APIError)

    def test_fetch_available_models_unexpected_error(self) -> None:
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock(name="openai.OpenAI")
            mock_client.models.list.side_effect = ValueError("Something else broke")
            mock_openai_cls.return_value = mock_client

            with pytest.raises(LLMProviderError) as exc_info:
                fetch_available_models("test-key")

        assert isinstance(exc_info.value.__cause__, ValueError)
