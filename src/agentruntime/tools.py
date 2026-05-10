"""Runtime-aware tool wrappers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from opentelemetry.trace import Span

from agentruntime.budget import BudgetController
from agentruntime.circuit_breaker import CircuitBreakerDecision, CircuitBreakerRegistry
from agentruntime.exceptions import AgentRuntimeError, CircuitBreakerOpen
from agentruntime.telemetry.conventions import tool_attributes
from agentruntime.telemetry.tracer import Telemetry


class ToolRunner:
    """Wrap arbitrary sync or async Python callables with runtime checks."""

    def __init__(
        self,
        budget: BudgetController,
        telemetry: Telemetry,
        circuit_breakers: CircuitBreakerRegistry,
    ) -> None:
        self._budget = budget
        self._telemetry = telemetry
        self._circuit_breakers = circuit_breakers

    def wrap(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            return await self.call(name, func, *args, **kwargs)

        return wrapped

    async def call(self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._telemetry.start_span(
            f"tool_call {name}",
            tool_attributes(tool_name=name, calls_used=self._budget.tool_calls),
        ) as span:
            try:
                decision = self._circuit_breakers.before_call(name)
                _apply_breaker_decision(
                    span,
                    self._telemetry,
                    decision,
                    calls_used=self._budget.tool_calls,
                    blocked=False,
                )
            except CircuitBreakerOpen as exc:
                _apply_breaker_open(
                    span,
                    self._telemetry,
                    name,
                    exc,
                    calls_used=self._budget.tool_calls,
                )
                self._telemetry.record_exception(span, exc)
                raise

            self._budget.record_tool_call_started(name)
            self._telemetry.set_attributes(
                span,
                tool_attributes(tool_name=name, calls_used=self._budget.tool_calls),
            )

            try:
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                decision = self._circuit_breakers.record_success(name)
                _apply_breaker_decision(
                    span,
                    self._telemetry,
                    decision,
                    calls_used=self._budget.tool_calls,
                    blocked=False,
                )
                self._telemetry.mark_ok(span)
                return result
            except AgentRuntimeError as exc:
                self._telemetry.record_exception(span, exc)
                raise
            except (asyncio.CancelledError, TimeoutError) as exc:
                self._telemetry.record_exception(span, exc)
                raise
            except Exception as exc:
                decision = self._circuit_breakers.record_failure(name)
                _apply_breaker_decision(
                    span,
                    self._telemetry,
                    decision,
                    calls_used=self._budget.tool_calls,
                    blocked=False,
                )
                self._telemetry.record_exception(span, exc)
                raise


def _apply_breaker_decision(
    span: Span,
    telemetry: Telemetry,
    decision: CircuitBreakerDecision | None,
    *,
    calls_used: int,
    blocked: bool,
) -> None:
    if decision is None:
        return

    snapshot = decision.snapshot
    telemetry.set_attributes(
        span,
        tool_attributes(
            tool_name=snapshot.tool_name,
            calls_used=calls_used,
            breaker_state=snapshot.state.value,
            breaker_failure_count=snapshot.failure_count,
            breaker_blocked=blocked,
            breaker_remaining_open_seconds=snapshot.remaining_open_seconds,
        ),
    )
    if not telemetry.config.enabled:
        return
    for event_name in decision.events:
        span.add_event(
            event_name,
            tool_attributes(
                tool_name=snapshot.tool_name,
                calls_used=calls_used,
                breaker_state=snapshot.state.value,
                breaker_failure_count=snapshot.failure_count,
                breaker_blocked=blocked,
                breaker_remaining_open_seconds=snapshot.remaining_open_seconds,
            ),
        )


def _apply_breaker_open(
    span: Span,
    telemetry: Telemetry,
    tool_name: str,
    exc: CircuitBreakerOpen,
    *,
    calls_used: int,
) -> None:
    attributes = tool_attributes(
        tool_name=tool_name,
        calls_used=calls_used,
        breaker_state=str(exc.details.get("state", "open")),
        breaker_failure_count=_int_detail(exc.details.get("failure_count")),
        breaker_blocked=True,
        breaker_remaining_open_seconds=_float_detail(exc.details.get("remaining_open_seconds")),
    )
    telemetry.set_attributes(span, attributes)
    if telemetry.config.enabled:
        span.add_event("agentruntime.circuit_breaker.blocked", attributes)


def _int_detail(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _float_detail(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
