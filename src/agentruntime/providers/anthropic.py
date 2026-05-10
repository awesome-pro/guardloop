"""Anthropic Messages API wrapper."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Mapping
from typing import Any, Protocol, cast

from agentruntime.budget import BudgetController
from agentruntime.telemetry.conventions import llm_request_attributes, llm_response_attributes
from agentruntime.telemetry.tracer import Telemetry
from agentruntime.tokenization import estimate_anthropic_tokens


class _MessagesAPI(Protocol):
    def create(self, **kwargs: Any) -> Awaitable[object] | object: ...


class _AnthropicClient(Protocol):
    @property
    def messages(self) -> _MessagesAPI: ...


class WrappedAnthropicClient:
    """Anthropic client facade that currently wraps `messages.create`."""

    def __init__(
        self,
        client: object,
        budget: BudgetController,
        telemetry: Telemetry,
    ) -> None:
        typed_client = cast(_AnthropicClient, client)
        self.messages = WrappedAnthropicMessages(typed_client.messages, budget, telemetry)


class WrappedAnthropicMessages:
    def __init__(
        self,
        messages: _MessagesAPI,
        budget: BudgetController,
        telemetry: Telemetry,
    ) -> None:
        self._messages = messages
        self._budget = budget
        self._telemetry = telemetry

    async def create(self, **kwargs: Any) -> object:
        model = _require_str(kwargs, "model")
        max_tokens = _optional_positive_int(kwargs.get("max_tokens"))
        estimated_input_tokens = estimate_anthropic_tokens(
            {"system": kwargs.get("system"), "messages": kwargs.get("messages")}
        )
        preflight = self._budget.check_llm_call(
            provider="anthropic",
            model=model,
            estimated_input_tokens=estimated_input_tokens,
            reserved_output_tokens=max_tokens,
        )

        with self._telemetry.start_span(
            "llm_call anthropic.messages.create",
            llm_request_attributes(
                provider="anthropic",
                model=model,
                estimated_input_tokens=estimated_input_tokens,
                reserved_output_tokens=preflight.reserved_output_tokens,
                estimated_cost_usd=preflight.estimated_cost_usd,
            ),
        ) as span:
            try:
                maybe_response = self._messages.create(**kwargs)
                response = (
                    await maybe_response if inspect.isawaitable(maybe_response) else maybe_response
                )
                input_tokens, output_tokens = _anthropic_usage_tokens(
                    response,
                    fallback_input_tokens=estimated_input_tokens,
                )
                actual_cost = self._budget.record_llm_call(
                    provider="anthropic",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                self._telemetry.set_attributes(
                    span,
                    llm_response_attributes(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=actual_cost,
                    ),
                )
                self._telemetry.mark_ok(span)
                return response
            except Exception as exc:
                self._telemetry.record_exception(span, exc)
                raise


def _require_str(kwargs: Mapping[str, Any], key: str) -> str:
    value = kwargs.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Anthropic messages.create requires a non-empty {key!r}.")
    return value


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    try:
        parsed = int(str(value))
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _get(obj: object, key: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return cast(Mapping[str, object], obj).get(key, default)
    return getattr(obj, key, default)


def _anthropic_usage_tokens(response: object, *, fallback_input_tokens: int) -> tuple[int, int]:
    usage = _get(response, "usage")
    if usage is None:
        return fallback_input_tokens, 0
    input_tokens = _as_int(
        _get(usage, "input_tokens", fallback_input_tokens),
        fallback_input_tokens,
    )
    cache_creation = _as_int(_get(usage, "cache_creation_input_tokens", 0), 0)
    cache_read = _as_int(_get(usage, "cache_read_input_tokens", 0), 0)
    output_tokens = _as_int(_get(usage, "output_tokens", 0), 0)
    return input_tokens + cache_creation + cache_read, output_tokens


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    return default
