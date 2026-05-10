"""Optional live Anthropic smoke test."""

from __future__ import annotations

import asyncio
import os

from guardloop import BudgetConfig, GuardLoop, RunContext


def _content_text(message: object) -> str:
    blocks = getattr(message, "content", [])
    texts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


async def agent(ctx: RunContext, prompt: str) -> str:
    message = await ctx.anthropic.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return _content_text(message)


async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running this live example.")

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
