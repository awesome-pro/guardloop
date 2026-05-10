"""Main GuardLoop entry point."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Span, Tracer

from guardloop.budget import BudgetController
from guardloop.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerSnapshot,
)
from guardloop.context import RunContext
from guardloop.exceptions import GuardLoopError, VerificationFailed
from guardloop.models import BudgetConfig, RunResult, TelemetryConfig
from guardloop.pricing import ModelPricing, PricingCatalog
from guardloop.telemetry.conventions import (
    GUARDLOOP_TERMINATED_REASON,
    run_attributes,
    verification_summary_attributes,
    verifier_attributes,
)
from guardloop.telemetry.tracer import Telemetry
from guardloop.verifier import (
    EVENT_RETRYING,
    EVENT_VERIFICATION_EXHAUSTED,
    EVENT_VERIFICATION_FAILED,
    Verifier,
    VerifierChain,
    VerifierConfig,
    feedback_for,
)

AgentCallable = Callable[..., Awaitable[object] | object]


@dataclass(slots=True)
class _RunOutcome:
    """Internal carrier for the result of the agent attempt loop."""

    output: object | None
    attempts: int
    verification_passed: bool | None
    feedback: list[str]
    last_verifier_name: str | None = None


class GuardLoop:
    """Execution wrapper that enforces runtime guardrails."""

    def __init__(
        self,
        *,
        budget: BudgetConfig | None = None,
        telemetry: TelemetryConfig | None = None,
        circuit_breakers: CircuitBreakerConfig | None = None,
        verifiers: Iterable[Verifier] | None = None,
        verifier_config: VerifierConfig | None = None,
        pricing: Iterable[ModelPricing] | None = None,
        include_default_pricing: bool = True,
        openai_client: Any | None = None,
        anthropic_client: Any | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.budget_config = budget or BudgetConfig()
        self.telemetry_config = telemetry or TelemetryConfig()
        self.verifier_config = verifier_config or VerifierConfig()
        self.pricing_catalog = PricingCatalog(pricing, include_defaults=include_default_pricing)
        self._circuit_breakers = CircuitBreakerRegistry(circuit_breakers)
        self._verifiers: list[Verifier] = list(verifiers or [])
        self._openai_client = openai_client
        self._anthropic_client = anthropic_client
        self._telemetry = Telemetry(self.telemetry_config, tracer=tracer)

    def circuit_breaker_snapshots(self) -> dict[str, CircuitBreakerSnapshot]:
        """Return current per-tool circuit breaker state."""

        return self._circuit_breakers.snapshots()

    def reset_circuit_breakers(self, tool_name: str | None = None) -> None:
        """Reset all circuit breakers or one named tool breaker."""

        self._circuit_breakers.reset(tool_name)

    def add_verifier(self, verifier: Verifier) -> None:
        """Append a verifier to this runtime's chain.

        The verifier runs after any already registered. ``run()`` snapshots the
        chain when it starts, so adding a verifier mid-run only affects later runs.
        """

        self._verifiers.append(verifier)

    async def run(self, agent: AgentCallable, *args: object, **kwargs: object) -> RunResult:
        budget = BudgetController(self.budget_config, self.pricing_catalog)
        ctx = RunContext(
            budget=budget,
            telemetry=self._telemetry,
            circuit_breakers=self._circuit_breakers,
            openai_client=self._openai_client,
            anthropic_client=self._anthropic_client,
        )
        chain = VerifierChain.from_iterable(self._verifiers)
        verifiers_active = bool(chain) and self.verifier_config.enabled

        with self._telemetry.start_span("agent_run", run_attributes()) as span:
            trace_id = self._telemetry.trace_id(span)
            try:
                outcome = await self._run_attempts_with_timeout(
                    agent, ctx, chain, verifiers_active, args, kwargs, span
                )
            except TimeoutError as exc:
                self._telemetry.record_exception(span, exc)
                span.set_attribute(GUARDLOOP_TERMINATED_REASON, "timeout")
                return self._result(
                    budget=budget,
                    span=span,
                    trace_id=trace_id,
                    success=False,
                    terminated_reason="timeout",
                    error_type=type(exc).__name__,
                    error_message=(
                        f"Run exceeded time limit of {self.budget_config.time_limit_seconds:.3f}s."
                    ),
                    verification_attempts=ctx.attempt,
                    verification_feedback=list(ctx.retry_feedback),
                )
            except GuardLoopError as exc:
                self._telemetry.record_exception(span, exc)
                span.set_attribute(GUARDLOOP_TERMINATED_REASON, exc.terminated_reason)
                return self._result(
                    budget=budget,
                    span=span,
                    trace_id=trace_id,
                    success=False,
                    terminated_reason=exc.terminated_reason,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    metadata={"details": _json_safe_details(exc.details)},
                    verification_attempts=ctx.attempt,
                    verification_feedback=list(ctx.retry_feedback),
                )
            except Exception as exc:
                self._telemetry.record_exception(span, exc)
                span.set_attribute(GUARDLOOP_TERMINATED_REASON, "error")
                return self._result(
                    budget=budget,
                    span=span,
                    trace_id=trace_id,
                    success=False,
                    terminated_reason="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    verification_attempts=ctx.attempt,
                    verification_feedback=list(ctx.retry_feedback),
                )

            if outcome.verification_passed is False:
                return self._verification_failed_result(
                    budget=budget, span=span, trace_id=trace_id, outcome=outcome
                )

            self._telemetry.set_attributes(
                span,
                verification_summary_attributes(
                    passed=outcome.verification_passed, attempts=outcome.attempts
                ),
            )
            self._telemetry.mark_ok(span)
            return self._result(
                budget=budget,
                span=span,
                trace_id=trace_id,
                success=True,
                output=None if outcome.output is None else str(outcome.output),
                verification_passed=outcome.verification_passed,
                verification_attempts=outcome.attempts,
                verification_feedback=outcome.feedback,
            )

    def _verification_failed_result(
        self,
        *,
        budget: BudgetController,
        span: Span,
        trace_id: str | None,
        outcome: _RunOutcome,
    ) -> RunResult:
        self._telemetry.set_attributes(
            span, verification_summary_attributes(passed=False, attempts=outcome.attempts)
        )
        span.set_attribute(GUARDLOOP_TERMINATED_REASON, "verification_failed")
        if self.verifier_config.raise_on_failure:
            exc = VerificationFailed(
                f"Output failed verification after {outcome.attempts} attempt(s).",
                attempts=outcome.attempts,
                feedback=outcome.feedback,
                verifier_name=outcome.last_verifier_name,
            )
            self._telemetry.record_exception(span, exc)
            return self._result(
                budget=budget,
                span=span,
                trace_id=trace_id,
                success=False,
                terminated_reason="verification_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={"details": _json_safe_details(exc.details)},
                verification_passed=False,
                verification_attempts=outcome.attempts,
                verification_feedback=outcome.feedback,
            )
        return self._result(
            budget=budget,
            span=span,
            trace_id=trace_id,
            success=False,
            output=None if outcome.output is None else str(outcome.output),
            terminated_reason="verification_failed",
            verification_passed=False,
            verification_attempts=outcome.attempts,
            verification_feedback=outcome.feedback,
        )

    async def _run_attempts_with_timeout(
        self,
        agent: AgentCallable,
        ctx: RunContext,
        chain: VerifierChain,
        verifiers_active: bool,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        span: Span,
    ) -> _RunOutcome:
        if self.budget_config.time_limit_seconds is None:
            return await self._run_attempts(agent, ctx, chain, verifiers_active, args, kwargs, span)
        async with asyncio.timeout(self.budget_config.time_limit_seconds):
            return await self._run_attempts(agent, ctx, chain, verifiers_active, args, kwargs, span)

    async def _run_attempts(
        self,
        agent: AgentCallable,
        ctx: RunContext,
        chain: VerifierChain,
        verifiers_active: bool,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        span: Span,
    ) -> _RunOutcome:
        max_attempts = self.verifier_config.max_attempts if verifiers_active else 1
        feedback_log: list[str] = []
        last_output: object | None = None

        for attempt in range(1, max_attempts + 1):
            ctx.attempt = attempt
            last_output = await _call_agent(agent, ctx, *args, **kwargs)
            if not verifiers_active:
                return _RunOutcome(
                    output=last_output, attempts=attempt, verification_passed=None, feedback=[]
                )

            verdict = await chain.run(
                telemetry=self._telemetry,
                output=last_output,
                attempt=attempt,
                max_attempts=max_attempts,
                prior_feedback=tuple(feedback_log),
                run_args=tuple(args),
                run_kwargs=dict(kwargs),
            )
            if verdict.passed:
                return _RunOutcome(
                    output=last_output,
                    attempts=attempt,
                    verification_passed=True,
                    feedback=list(feedback_log),
                )

            message = feedback_for(verdict)
            feedback_log.append(message)
            if self._telemetry.config.enabled:
                span.add_event(
                    EVENT_VERIFICATION_FAILED,
                    verifier_attributes(
                        name=verdict.verifier_name or "verifier",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        passed=False,
                    ),
                )

            if attempt < max_attempts:
                if self.verifier_config.pass_feedback_to_agent:
                    ctx.retry_feedback.append(message)
                if self._telemetry.config.enabled:
                    span.add_event(
                        EVENT_RETRYING,
                        verification_summary_attributes(passed=None, attempts=attempt + 1),
                    )
                continue

            if self._telemetry.config.enabled:
                span.add_event(
                    EVENT_VERIFICATION_EXHAUSTED,
                    verification_summary_attributes(passed=False, attempts=attempt),
                )
            return _RunOutcome(
                output=last_output,
                attempts=attempt,
                verification_passed=False,
                feedback=list(feedback_log),
                last_verifier_name=verdict.verifier_name,
            )

        raise RuntimeError(  # pragma: no cover - the loop above always returns
            "verifier attempt loop exited without producing a result"
        )

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
        verification_passed: bool | None = None,
        verification_attempts: int = 0,
        verification_feedback: list[str] | None = None,
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
            verification_passed=verification_passed,
            verification_attempts=verification_attempts,
            verification_feedback=list(verification_feedback or []),
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
