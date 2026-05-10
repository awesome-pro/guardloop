"""Main AgentRuntime entry point."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from opentelemetry.trace import Span, Tracer

from agentruntime.budget import BudgetController
from agentruntime.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerSnapshot,
)
from agentruntime.context import RunContext
from agentruntime.exceptions import AgentRuntimeError
from agentruntime.models import BudgetConfig, RunResult, TelemetryConfig
from agentruntime.pricing import ModelPricing, PricingCatalog
from agentruntime.telemetry.conventions import (
    AGENTRUNTIME_TERMINATED_REASON,
    run_attributes,
)
from agentruntime.telemetry.tracer import Telemetry

AgentCallable = Callable[..., Awaitable[object] | object]


class AgentRuntime:
    """Execution wrapper that enforces runtime guardrails."""

    def __init__(
        self,
        *,
        budget: BudgetConfig | None = None,
        telemetry: TelemetryConfig | None = None,
        circuit_breakers: CircuitBreakerConfig | None = None,
        pricing: Iterable[ModelPricing] | None = None,
        include_default_pricing: bool = True,
        openai_client: Any | None = None,
        anthropic_client: Any | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.budget_config = budget or BudgetConfig()
        self.telemetry_config = telemetry or TelemetryConfig()
        self.pricing_catalog = PricingCatalog(pricing, include_defaults=include_default_pricing)
        self._circuit_breakers = CircuitBreakerRegistry(circuit_breakers)
        self._openai_client = openai_client
        self._anthropic_client = anthropic_client
        self._telemetry = Telemetry(self.telemetry_config, tracer=tracer)

    def circuit_breaker_snapshots(self) -> dict[str, CircuitBreakerSnapshot]:
        """Return current per-tool circuit breaker state."""

        return self._circuit_breakers.snapshots()

    def reset_circuit_breakers(self, tool_name: str | None = None) -> None:
        """Reset all circuit breakers or one named tool breaker."""

        self._circuit_breakers.reset(tool_name)

    async def run(self, agent: AgentCallable, *args: object, **kwargs: object) -> RunResult:
        budget = BudgetController(self.budget_config, self.pricing_catalog)
        ctx = RunContext(
            budget=budget,
            telemetry=self._telemetry,
            circuit_breakers=self._circuit_breakers,
            openai_client=self._openai_client,
            anthropic_client=self._anthropic_client,
        )

        with self._telemetry.start_span("agent_run", run_attributes()) as span:
            trace_id = self._telemetry.trace_id(span)
            try:
                result = await self._run_with_optional_timeout(agent, ctx, *args, **kwargs)
                self._telemetry.mark_ok(span)
                return self._result(
                    budget=budget,
                    span=span,
                    trace_id=trace_id,
                    success=True,
                    output=None if result is None else str(result),
                )
            except TimeoutError as exc:
                self._telemetry.record_exception(span, exc)
                span.set_attribute(AGENTRUNTIME_TERMINATED_REASON, "timeout")
                return self._result(
                    budget=budget,
                    span=span,
                    trace_id=trace_id,
                    success=False,
                    terminated_reason="timeout",
                    error_type=type(exc).__name__,
                    error_message=f"Run exceeded time limit of "
                    f"{self.budget_config.time_limit_seconds:.3f}s.",
                )
            except AgentRuntimeError as exc:
                self._telemetry.record_exception(span, exc)
                span.set_attribute(AGENTRUNTIME_TERMINATED_REASON, exc.terminated_reason)
                return self._result(
                    budget=budget,
                    span=span,
                    trace_id=trace_id,
                    success=False,
                    terminated_reason=exc.terminated_reason,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    metadata={"details": _json_safe_details(exc.details)},
                )
            except Exception as exc:
                self._telemetry.record_exception(span, exc)
                span.set_attribute(AGENTRUNTIME_TERMINATED_REASON, "error")
                return self._result(
                    budget=budget,
                    span=span,
                    trace_id=trace_id,
                    success=False,
                    terminated_reason="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

    async def _run_with_optional_timeout(
        self,
        agent: AgentCallable,
        ctx: RunContext,
        *args: object,
        **kwargs: object,
    ) -> object:
        if self.budget_config.time_limit_seconds is None:
            return await _call_agent(agent, ctx, *args, **kwargs)
        async with asyncio.timeout(self.budget_config.time_limit_seconds):
            return await _call_agent(agent, ctx, *args, **kwargs)

    @staticmethod
    def _result(
        *,
        budget: BudgetController,
        span: Span,
        trace_id: str | None,
        success: bool,
        output: str | None = None,
        terminated_reason: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunResult:
        return RunResult(
            output=output,
            success=success,
            cost_usd=budget.cost_usd,
            estimated_cost_usd=budget.estimated_cost_usd,
            tokens_used=budget.tokens_used,
            input_tokens=budget.input_tokens,
            output_tokens=budget.output_tokens,
            duration_seconds=budget.duration_seconds,
            tool_calls=budget.tool_calls,
            trace_id=trace_id or Telemetry.trace_id(span),
            terminated_reason=terminated_reason,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata or {},
        )


async def _call_agent(
    agent: AgentCallable,
    ctx: RunContext,
    *args: object,
    **kwargs: object,
) -> object:
    result = agent(ctx, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


JsonSafeDetail = str | int | float | bool | None


def _json_safe_details(details: dict[str, Any]) -> dict[str, JsonSafeDetail]:
    return {key: _json_safe_detail(value) for key, value in details.items() if value is not None}


def _json_safe_detail(value: Any) -> JsonSafeDetail:
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)
