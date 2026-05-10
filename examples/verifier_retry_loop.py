"""No-key demo: a verifier rejects a bad answer; the agent self-corrects on retry."""

from __future__ import annotations

import asyncio
import json

from guardloop import (
    BudgetConfig,
    GuardLoop,
    RunContext,
    VerifierConfig,
    VerifierContext,
    VerifierResult,
    is_json_object,
)


def no_todo_placeholder(output: object, ctx: VerifierContext) -> VerifierResult:
    if "TODO" in str(output):
        return VerifierResult(
            passed=False,
            feedback="The answer still contains a 'TODO' placeholder; use a real value.",
        )
    return VerifierResult(passed=True)


async def main() -> None:
    runtime = GuardLoop(
        budget=BudgetConfig(time_limit_seconds=10, tool_call_limit=10),
        verifiers=[no_todo_placeholder, is_json_object(required_keys=["answer"])],
        verifier_config=VerifierConfig(max_retries=2),
    )

    attempt = 0

    async def agent(ctx: RunContext, question: str) -> str:
        nonlocal attempt
        attempt += 1
        print(f"agent attempt {attempt}; feedback so far: {ctx.retry_feedback}")
        if attempt == 1:
            return '{"answer": "TODO"}'  # rejected: placeholder text
        if attempt == 2:
            return "answer = 42"  # rejected: not valid JSON
        return json.dumps({"answer": 42, "question": question})  # accepted

    result = await runtime.run(agent, "what is six times seven?")

    print("\nRunResult:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
