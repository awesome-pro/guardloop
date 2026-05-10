"""Optional live OpenAI smoke test."""

from __future__ import annotations

import asyncio
import os
from typing import Protocol, cast

from guardloop import BudgetConfig, GuardLoop, RunContext


class HasOutputText(Protocol):
    output_text: str


async def agent(ctx: RunContext, prompt: str) -> str:
    response = await ctx.openai.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        input=prompt,
        max_output_tokens=120,
    )
    return cast(HasOutputText, response).output_text


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this live example.")

    runtime = GuardLoop(
        budget=BudgetConfig(
            cost_limit_usd="0.05",
            token_limit=5_000,
            time_limit_seconds=60,
            tool_call_limit=5,
        )
    )
    result = await runtime.run(agent, "Explain agent runtime budget caps in two sentences.")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
