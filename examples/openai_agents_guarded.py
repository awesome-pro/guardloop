# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# The OpenAI Agents SDK's Model interface and Responses-API item types are
# loosely parameterised at this boundary; strict "unknown type" checks are
# relaxed for this demo's fake model wiring.
"""No-key demo: run an OpenAI Agents SDK ``Agent`` under GuardLoop.

The agent uses an in-process fake model (named like a priced model), so the demo
needs no API key. GuardLoop's budget caps and OpenTelemetry spans apply *inside*
``Runner.run`` via ``guarded_runner(...)``; the first run succeeds with cost and
tokens recorded, the second runs under a deliberately tiny token budget and is
stopped before the model call. ``RunConfig(tracing_disabled=True)`` keeps the
SDK's own tracer quiet (it would otherwise warn about the missing API key).

    uv run python examples/openai_agents_guarded.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Agent, Model, ModelResponse, RunConfig
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from guardloop import BudgetConfig, GuardLoop
from guardloop.adapters.openai_agents import guarded_runner


class DemoModel(Model):
    """A canned model that reports its name as ``gpt-5.2`` so default pricing applies."""

    model = "gpt-5.2"

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        message = ResponseOutputMessage(
            id="msg_demo",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text="1) line up the audience, 2) write the changelog, 3) tag and publish",
                    annotations=[],
                )
            ],
        )
        return ModelResponse(
            output=[message],
            usage=Usage(requests=1, input_tokens=64, output_tokens=32, total_tokens=96),
            response_id="resp_demo",
            request_id=None,
        )

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("DemoModel does not support streaming")


def build_agent() -> Agent[Any]:
    return Agent(
        name="release-planner",
        model=DemoModel(),
        instructions="You plan software releases in concise numbered steps.",
    )


async def main() -> None:
    agent = guarded_runner(build_agent(), run_config=RunConfig(tracing_disabled=True))

    print("== run 1: comfortable budget ==")
    generous = GuardLoop(
        budget=BudgetConfig(cost_limit_usd="0.05", token_limit=5_000, tool_call_limit=5)
    )
    ok = await generous.run(agent, "plan a 3-step release")
    print(ok.model_dump_json(indent=2))

    print("\n== run 2: token budget too small for the call ==")
    tight = GuardLoop(budget=BudgetConfig(token_limit=10))
    blocked = await tight.run(agent, "plan a 3-step release")
    print(f"success={blocked.success}")
    print(f"terminated_reason={blocked.terminated_reason}")
    print(f"output={blocked.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
