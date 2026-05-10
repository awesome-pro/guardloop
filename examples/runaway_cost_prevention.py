"""No-key demo: stop a runaway agent before it can make the next costly call."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, cast

from guardloop import BudgetConfig, GuardLoop, RunContext


@dataclass(slots=True)
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass(slots=True)
class FakeResponse:
    output_text: str
    usage: FakeUsage


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **_: object) -> FakeResponse:
        self.calls += 1
        return FakeResponse(
            output_text=f"loop iteration {self.calls}: still researching...",
            usage=FakeUsage(input_tokens=600, output_tokens=300),
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class HasOutputText(Protocol):
    output_text: str


async def runaway_agent(ctx: RunContext, topic: str) -> str:
    while True:
        response = await ctx.openai.responses.create(
            model="gpt-5.2",
            input=f"{topic}\nContinue researching and do not stop yet.",
            max_output_tokens=500,
        )
        print(cast(HasOutputText, response).output_text)


async def main() -> None:
    fake_openai = FakeOpenAIClient()
    runtime = GuardLoop(
        budget=BudgetConfig(
            cost_limit_usd="0.02",
            token_limit=10_000,
            time_limit_seconds=30,
            tool_call_limit=10,
        ),
        openai_client=fake_openai,
    )

    result = await runtime.run(runaway_agent, "Investigate agent runtime safety.")
    print("\nGuardLoop stopped the runaway loop:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
