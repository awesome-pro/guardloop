"""No-key demo: stop retrying a failing tool with a circuit breaker."""

from __future__ import annotations

import asyncio
import json

from guardloop import (
    BudgetConfig,
    CircuitBreakerConfig,
    CircuitBreakerPolicy,
    GuardLoop,
    RunContext,
)


async def main() -> None:
    runtime = GuardLoop(
        budget=BudgetConfig(tool_call_limit=10, time_limit_seconds=10),
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(
                failure_threshold=2,
                recovery_timeout_seconds=30,
            )
        ),
    )

    attempts = 0

    def flaky_search(query: str) -> str:
        nonlocal attempts
        attempts += 1
        print(f"tool attempt {attempts}: searching for {query!r}")
        raise RuntimeError("upstream search API returned HTTP 503")

    async def agent(ctx: RunContext) -> str:
        for step in range(1, 6):
            try:
                await ctx.call_tool("vendor_search", flaky_search, "agent runtime safety")
            except RuntimeError as exc:
                print(f"agent step {step}: tool failed, retrying: {exc}")
                continue
        return "unreachable"

    result = await runtime.run(agent)
    snapshots = {
        name: snapshot.model_dump(mode="json")
        for name, snapshot in runtime.circuit_breaker_snapshots().items()
    }

    print("\nRunResult:")
    print(result.model_dump_json(indent=2))
    print("\nCircuit breaker snapshots:")
    print(json.dumps(snapshots, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
