"""
LLM client layer — Strategy + Composite + Fallback.

DESIGN INVARIANTS
- LLMClient is a Protocol with one method: `analyze(prompt) -> dict`.
- FakeLLMClient is a dataclass frozen that returns a canned response
  (and optionally raises). Deterministic, no I/O, safe in tests.
- FallbackLLMClient is a dataclass frozen that delegates to the
  first client that succeeds. SystemExit and KeyboardInterrupt
  bypass the fallback (they are user-initiated, not provider
  failures). Any other exception is caught and the next client
  is tried. If every client fails, LLMFallbackError is raised
  chained from the last underlying exception.
- OpenAICompatibleClient wraps the official `openai` SDK and works
  against any OpenAI-compatible endpoint (OpenAI, NVIDIA NIM, Groq,
  Together, etc.). Provider exceptions are wrapped in the project's
  LLMProviderError / LLMResponseError so the FallbackLLMClient can
  catch them via its `except Exception` clause.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import openai

from src.errors import LLMProviderError, LLMResponseError

if TYPE_CHECKING:
    # Only needed for type-checkers. The actual `openai` package is
    # imported lazily inside analyze() to keep the test surface
    # free of the heavy dependency.
    import openai  # noqa: F401


# T1: Protocol describing the openai SDK methods we actually use.
# This lets us type the cached client and the lazy module without
# forcing an import of the openai package at module load.
class _OpenAIModule(Protocol):
    """Structural type for the subset of the openai module we use."""

    def OpenAI(  # noqa: N802 — mirrors the openai module's API
        self, *, api_key: str, base_url: str
    ) -> Any: ...  # noqa: ANN401


class _OpenAIClient(Protocol):
    """Structural type for the openai.OpenAI client we cache."""

    chat: Any  # noqa: ANN401


# Structural type for the response shape we care about. `content` is
# `str | None` in the real SDK (it can be None for tool-only responses).
class _Message(Protocol):
    content: str | None


class _Choice(Protocol):
    message: _Message


class _Completion(Protocol):
    choices: list[_Choice]


# Default endpoint for OpenAI-compatible APIs. NVIDIA NIM is the
# primary target for this project, but the URL is overridable.
_DEFAULT_BASE_URL: str = "https://integrate.api.nvidia.com/v1"


@runtime_checkable
class LLMClient(Protocol):
    """Protocol every LLM client must satisfy.

    The return type is `dict[str, Any]` because LLM responses
    are inherently JSON-shaped and vary by provider. Callers narrow
    the type when consuming the result; the protocol keeps the
    surface intentionally narrow to avoid leaking provider-specific
    structures across the layer.
    """

    def analyze(self, prompt: str) -> dict[str, Any]:
        """Send a prompt to the LLM and return the parsed response."""
        ...


class LLMFallbackError(LLMProviderError):
    """Raised when every LLM client in a FallbackLLMClient chain fails.

    Inherits from LLMProviderError (which inherits from InsightError) so
    the CLI's `except InsightError` catch block can translate it into a
    clean exit code 1 instead of leaking an unhandled Exception.

    The original exception (if any) is reachable via __cause__
    so callers can introspect what went wrong upstream.
    """


@dataclass(frozen=True, slots=True)
class FakeLLMClient(LLMClient):
    """A canned-response LLM client for testing.

    Returns the exact same dict on every call. Optionally raises a
    pre-configured exception to simulate provider failures.
    Deterministic, zero I/O — safe in unit tests and CI.
    """

    canned_response: dict[str, Any]
    exception: Exception | None = None

    def analyze(self, prompt: str) -> dict[str, Any]:
        if self.exception is not None:
            raise self.exception
        return self.canned_response


@dataclass(frozen=True, slots=True)
class OpenAICompatibleClient(LLMClient):
    """Production LLM client backed by the official `openai` SDK.

    Works against any OpenAI-compatible endpoint:
    - OpenAI (https://api.openai.com/v1) — set base_url explicitly
    - NVIDIA NIM (https://integrate.api.nvidia.com/v1) — the default
    - Groq, Together, OpenRouter, etc. — override base_url

    The client requests JSON output via `response_format=json_object`
    so the LLM is forced to produce a parseable JSON string. The
    content is then parsed with `json.loads` and the resulting dict
    is returned.

    Error mapping:
    - openai.APIError         -> LLMProviderError  (rate limit, auth, etc.)
    - openai.APIConnectionError -> LLMProviderError (network failure)
    - json.JSONDecodeError    -> LLMResponseError  (malformed LLM output)
    - Any other SDK error     -> LLMProviderError  (defensive fallback)

    The original SDK exception is always chained via `__cause__`
    for debuggability, but is NEVER included in the message string
    (it may contain the user's prompt).
    """

    api_key: str
    model_id: str
    base_url: str = _DEFAULT_BASE_URL
    # Cached openai.OpenAI instance (security audit S5). Declared as a
    # field with init=False so dataclass(slots=True) creates a slot for
    # it; the frozen guard is bypassed in _get_or_create_sdk_client via
    # object.__setattr__ so the cache can be installed on first use.
    _sdk_client: _OpenAIClient | None = field(default=None, init=False, repr=False, compare=False)

    def analyze(self, prompt: str) -> dict[str, Any]:
        # Lazy import: keeps the test surface free of the heavy
        # `openai` dependency when the client is never used (e.g.
        # in unit tests that only exercise FakeLLMClient).

        # Security audit S5: the openai SDK owns an httpx.Client
        # connection pool. Constructing a fresh openai.OpenAI() on
        # every call leaks sockets and exhausts the pool. We cache
        # the SDK client on the first successful construction and
        # reuse it across subsequent calls.
        #
        # The dataclass is frozen=True, so we must use
        # object.__setattr__ to bypass the frozen guard for the
        # cache slot. We use _sdk_client as the slot name to
        # keep the cache off the public surface.
        sdk_client: _OpenAIClient = self._get_or_create_sdk_client(openai)

        try:
            completion: _Completion = sdk_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except (openai.APIError, openai.APIConnectionError) as exc:
            raise LLMProviderError("OpenAI-compatible provider returned an error") from exc
        except Exception as exc:
            # Defensive: any other SDK exception (timeout, auth, etc.)
            # is wrapped so the FallbackLLMClient can catch it.
            raise LLMProviderError("OpenAI-compatible provider raised an unexpected error") from exc

        try:
            content: str | None = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "OpenAI-compatible provider returned an unexpected response shape"
            ) from exc

        if content is None:
            # The response had no text content (e.g. tool-only response).
            raise LLMResponseError("OpenAI-compatible provider returned no text content")

        try:
            parsed: object = json.loads(content)
        except json.JSONDecodeError as exc:
            # The LLM was asked for JSON but returned garbage. Wrap it
            # so the Fallback can try the next client.
            raise LLMResponseError("OpenAI-compatible provider returned malformed JSON") from exc

        if not isinstance(parsed, dict):
            raise LLMResponseError("OpenAI-compatible provider returned JSON but not an object")

        return cast(dict[str, Any], parsed)

    def _get_or_create_sdk_client(self, openai_module: _OpenAIModule) -> _OpenAIClient:
        """Return the cached openai.OpenAI instance, creating it on first use.

        The cache lives in the private `_sdk_client` attribute. Because
        the dataclass is frozen=True, we use `object.__setattr__` to
        install the cache exactly once. Subsequent calls reuse the
        same SDK client (and therefore the same httpx connection pool).
        """
        cached: _OpenAIClient | None = getattr(self, "_sdk_client", None)
        if cached is not None:
            return cached
        created: _OpenAIClient = openai_module.OpenAI(api_key=self.api_key, base_url=self.base_url)
        # Bypass the frozen guard for the cache slot. This is safe
        # because the attribute is private (underscore prefix) and
        # never appears in the public surface of the dataclass.
        object.__setattr__(self, "_sdk_client", created)
        return created


@dataclass(frozen=True, slots=True)
class FallbackLLMClient(LLMClient):
    """Composite LLM client that tries each client in order.

    Construction: use the classmethod `of()` for variadic args, or
    the standard dataclass constructor for a pre-built tuple.

        FallbackLLMClient.of(primary, secondary, tertiary)
        FallbackLLMClient(clients=(primary, secondary))

    Behavior:
    - Each client is tried in order.
    - If a client raises SystemExit or KeyboardInterrupt, the fallback
      re-raises immediately (these are user-initiated, not provider
      failures, and aborting the fallback chain is the right call).
    - If a client raises any other exception, the fallback swallows
      it, remembers it as the last cause, and tries the next client.
    - If every client fails, raises LLMFallbackError chained from
      the last exception.
    - An empty client list raises LLMFallbackError immediately —
      a silent pass would hide configuration bugs.
    """

    clients: tuple[LLMClient, ...]

    @classmethod
    def of(cls, *clients: LLMClient) -> FallbackLLMClient:
        """Construct a FallbackLLMClient from variadic positional args.

        T8: this classmethod replaces the previous __init__ that
        relied on object.__setattr__ to bypass the frozen guard. The
        standard dataclass-generated __init__ now handles assignment
        through the declared slot — no frozen-bypass required.
        """
        return cls(clients=tuple(clients))

    def analyze(self, prompt: str) -> dict[str, Any]:
        if not self.clients:
            raise LLMFallbackError("FallbackLLMClient has no clients configured")

        last_exc: Exception | None = None
        for client in self.clients:
            try:
                return client.analyze(prompt)
            except (SystemExit, KeyboardInterrupt):
                # User-initiated control flow — never swallow.
                raise
            except Exception as exc:
                # Provider failure — remember and try the next client.
                last_exc = exc
                continue

        # All clients failed. Chain the last exception so callers can
        # inspect what went wrong upstream.
        raise LLMFallbackError(f"All {len(self.clients)} LLM clients failed") from last_exc


def fetch_available_models(api_key: str, base_url: str = _DEFAULT_BASE_URL) -> list[str]:
    """Fetch the list of available model IDs from the given provider.

    Used during interactive onboarding to let the user select a valid model.
    Instantiates a temporary openai.OpenAI client just for the probe.

    Returns:
        A sorted list of model ID strings.

    Raises:
        LLMProviderError: If the API key is invalid, the endpoint is
            unreachable, or the provider returns an error.
    """

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.models.list()
        return sorted([model.id for model in response.data])
    except (openai.APIError, openai.APIConnectionError) as exc:
        raise LLMProviderError(f"Failed to fetch models: {exc}") from exc
    except Exception as exc:
        raise LLMProviderError(f"Unexpected error fetching models: {exc}") from exc


def verify_connection(api_key: str, base_url: str, model_id: str) -> None:
    """Verify that the API key and model work by making a minimal request.

    Raises:
        LLMProviderError: If the connection or authentication fails.
    """

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except (openai.APIError, openai.APIConnectionError) as exc:
        raise LLMProviderError(f"Verification failed: {exc}") from exc
    except Exception as exc:
        raise LLMProviderError(f"Unexpected verification error: {exc}") from exc


__all__ = [
    "LLMClient",
    "LLMFallbackError",
    "FakeLLMClient",
    "FallbackLLMClient",
    "OpenAICompatibleClient",
    "fetch_available_models",
    "verify_connection",
]
