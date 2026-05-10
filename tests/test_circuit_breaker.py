from __future__ import annotations

from collections.abc import Callable

import pytest

from agentruntime import (
    AgentRuntime,
    BudgetConfig,
    CircuitBreakerConfig,
    CircuitBreakerPolicy,
    CircuitBreakerState,
    RunContext,
)
from agentruntime.budget import BudgetController
from agentruntime.circuit_breaker import CircuitBreakerRegistry
from agentruntime.models import TelemetryConfig as InternalTelemetryConfig
from agentruntime.pricing import PricingCatalog
from agentruntime.telemetry.tracer import Telemetry
from agentruntime.tools import ToolRunner


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def now(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


async def test_closed_breaker_allows_normal_tool_calls() -> None:
    runtime = AgentRuntime()
    calls = 0

    def stable_tool() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    async def agent(ctx: RunContext) -> str:
        return str(await ctx.call_tool("stable", stable_tool))

    result = await runtime.run(agent)
    snapshot = runtime.circuit_breaker_snapshots()["stable"]

    assert result.success is True
    assert result.output == "ok"
    assert calls == 1
    assert snapshot.state == CircuitBreakerState.CLOSED
    assert snapshot.failure_count == 0


async def test_consecutive_failures_open_breaker_and_reject_without_invoking_tool() -> None:
    runtime = AgentRuntime(
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=2, recovery_timeout_seconds=30)
        )
    )
    calls = 0

    def flaky_tool() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("flaky")

    async def agent(ctx: RunContext) -> str:
        return str(await ctx.call_tool("flaky", flaky_tool))

    first = await runtime.run(agent)
    second = await runtime.run(agent)
    blocked = await runtime.run(agent)
    snapshot = runtime.circuit_breaker_snapshots()["flaky"]

    assert first.terminated_reason == "error"
    assert second.terminated_reason == "error"
    assert blocked.terminated_reason == "circuit_breaker_open"
    assert blocked.tool_calls == 0
    assert calls == 2
    assert snapshot.state == CircuitBreakerState.OPEN
    assert snapshot.failure_count == 2


async def test_open_breaker_does_not_increment_tool_call_limit() -> None:
    runtime = AgentRuntime(
        budget=BudgetConfig(tool_call_limit=1),
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=30)
        ),
    )
    calls = 0

    def flaky_tool() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("flaky")

    async def agent(ctx: RunContext) -> str:
        try:
            await ctx.call_tool("flaky", flaky_tool)
        except ValueError:
            pass
        return str(await ctx.call_tool("flaky", flaky_tool))

    result = await runtime.run(agent)

    assert result.terminated_reason == "circuit_breaker_open"
    assert result.tool_calls == 1
    assert calls == 1


async def test_cooldown_expiry_moves_to_half_open_and_success_closes() -> None:
    clock = FakeClock()
    runner, registry = _tool_runner(
        clock=clock.now,
        policy=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=5),
    )

    with pytest.raises(ValueError):
        await runner.call("flaky", _raise_value_error)

    assert registry.snapshots()["flaky"].state == CircuitBreakerState.OPEN

    clock.advance(5.1)
    decision = registry.before_call("flaky")
    assert decision is not None
    assert decision.snapshot.state == CircuitBreakerState.HALF_OPEN

    result = await runner.call("flaky", lambda: "ok")
    snapshot = registry.snapshots()["flaky"]

    assert result == "ok"
    assert snapshot.state == CircuitBreakerState.CLOSED
    assert snapshot.failure_count == 0


async def test_half_open_failure_immediately_reopens() -> None:
    clock = FakeClock()
    runner, registry = _tool_runner(
        clock=clock.now,
        policy=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=5),
    )

    with pytest.raises(ValueError):
        await runner.call("flaky", _raise_value_error)

    clock.advance(5.1)

    with pytest.raises(ValueError):
        await runner.call("flaky", _raise_value_error)

    snapshot = registry.snapshots()["flaky"]

    assert snapshot.state == CircuitBreakerState.OPEN
    assert snapshot.remaining_open_seconds == 5


async def test_per_tool_overrides_are_independent_from_global_default() -> None:
    runner, registry = _tool_runner(
        policy=CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_seconds=30),
        config=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_seconds=30),
            tool_overrides={
                "web_search": CircuitBreakerPolicy(
                    failure_threshold=1,
                    recovery_timeout_seconds=30,
                )
            },
        ),
    )

    with pytest.raises(ValueError):
        await runner.call("web_search", _raise_value_error)
    with pytest.raises(ValueError):
        await runner.call("database", _raise_value_error)

    snapshots = registry.snapshots()

    assert snapshots["web_search"].state == CircuitBreakerState.OPEN
    assert snapshots["database"].state == CircuitBreakerState.CLOSED
    assert snapshots["database"].failure_count == 1


async def test_breaker_state_persists_across_runtime_runs() -> None:
    runtime = AgentRuntime(
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=30)
        )
    )

    async def agent(ctx: RunContext) -> str:
        return str(await ctx.call_tool("flaky", _raise_value_error))

    first = await runtime.run(agent)
    second = await runtime.run(agent)

    assert first.terminated_reason == "error"
    assert second.terminated_reason == "circuit_breaker_open"
    assert runtime.circuit_breaker_snapshots()["flaky"].state == CircuitBreakerState.OPEN


async def test_runtime_reset_helpers_reset_one_or_all_breakers() -> None:
    runtime = AgentRuntime(
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=30)
        )
    )

    async def agent(ctx: RunContext, tool_name: str) -> str:
        return str(await ctx.call_tool(tool_name, _raise_value_error))

    await runtime.run(agent, "one")
    await runtime.run(agent, "two")

    runtime.reset_circuit_breakers("one")
    snapshots = runtime.circuit_breaker_snapshots()

    assert "one" not in snapshots
    assert snapshots["two"].state == CircuitBreakerState.OPEN

    runtime.reset_circuit_breakers()

    assert runtime.circuit_breaker_snapshots() == {}


async def test_circuit_breaker_open_returns_structured_result_metadata() -> None:
    runtime = AgentRuntime(
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=30)
        )
    )

    async def agent(ctx: RunContext) -> str:
        return str(await ctx.call_tool("flaky", _raise_value_error))

    await runtime.run(agent)
    result = await runtime.run(agent)
    details = result.metadata["details"]

    assert result.success is False
    assert result.terminated_reason == "circuit_breaker_open"
    assert details["tool_name"] == "flaky"
    assert details["state"] == "open"
    assert details["failure_count"] == 1
    assert isinstance(details["remaining_open_seconds"], float)


async def test_disabled_breakers_do_not_block_or_track() -> None:
    runtime = AgentRuntime(circuit_breakers=CircuitBreakerConfig(enabled=False))
    calls = 0

    def flaky_tool() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("flaky")

    async def agent(ctx: RunContext) -> str:
        return str(await ctx.call_tool("flaky", flaky_tool))

    await runtime.run(agent)
    await runtime.run(agent)

    assert calls == 2
    assert runtime.circuit_breaker_snapshots() == {}


def _raise_value_error() -> str:
    raise ValueError("flaky")


def _tool_runner(
    *,
    policy: CircuitBreakerPolicy,
    clock: Callable[[], float] | None = None,
    config: CircuitBreakerConfig | None = None,
) -> tuple[ToolRunner, CircuitBreakerRegistry]:
    registry = CircuitBreakerRegistry(
        config or CircuitBreakerConfig(default=policy),
        clock=clock,
    )
    budget = BudgetController(BudgetConfig(tool_call_limit=100), PricingCatalog())
    telemetry = Telemetry(InternalTelemetryConfig(enabled=False))
    return ToolRunner(budget, telemetry, registry), registry
