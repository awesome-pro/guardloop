from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Protocol, cast

from agentruntime import AgentRuntime, BudgetConfig, RunContext
from tests.fakes import FakeOpenAIClient, FakeOpenAIResponses


class HasOutputText(Protocol):
    output_text: str


async def test_successful_agent_run_returns_structured_result() -> None:
    runtime = AgentRuntime(
        budget=BudgetConfig(cost_limit_usd="1.00", token_limit=10_000, time_limit_seconds=10),
        openai_client=FakeOpenAIClient(),
    )

    async def agent(ctx: RunContext, prompt: str) -> str:
        response = await ctx.openai.responses.create(
            model="gpt-5.2",
            input=prompt,
            max_output_tokens=100,
        )
        return cast(HasOutputText, response).output_text

    result = await runtime.run(agent, "hello")

    assert result.success is True
    assert result.output == "ok"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.terminated_reason is None


async def test_runaway_fake_agent_stops_at_cost_cap() -> None:
    fake_responses = FakeOpenAIResponses(input_tokens=600, output_tokens=300)
    runtime = AgentRuntime(
        budget=BudgetConfig(cost_limit_usd="0.02", token_limit=10_000, time_limit_seconds=10),
        openai_client=FakeOpenAIClient(fake_responses),
    )

    async def runaway(ctx: RunContext) -> str:
        while True:
            await ctx.openai.responses.create(
                model="gpt-5.2",
                input="keep going",
                max_output_tokens=500,
            )

    result = await runtime.run(runaway)

    assert result.success is False
    assert result.terminated_reason == "budget_exceeded"
    assert fake_responses.calls > 0
    assert result.cost_usd <= Decimal("0.02")


async def test_timeout_path_returns_timeout_reason() -> None:
    runtime = AgentRuntime(budget=BudgetConfig(time_limit_seconds=0.01))

    async def slow_agent(_: RunContext) -> str:
        await asyncio.sleep(1)
        return "done"

    result = await runtime.run(slow_agent)

    assert result.success is False
    assert result.terminated_reason == "timeout"
    assert result.error_type == "TimeoutError"


async def test_tool_exception_is_captured_as_runtime_error() -> None:
    runtime = AgentRuntime(budget=BudgetConfig(tool_call_limit=5))

    def broken_tool() -> str:
        raise ValueError("tool exploded")

    async def agent(ctx: RunContext) -> str:
        await ctx.call_tool("broken_tool", broken_tool)
        return "unreachable"

    result = await runtime.run(agent)

    assert result.success is False
    assert result.terminated_reason == "error"
    assert result.error_type == "ValueError"
    assert result.tool_calls == 1
