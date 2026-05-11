# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# LangGraph's StateGraph builder methods carry partially-unparameterised generics in
# their stubs; strict "unknown type" checks are relaxed for this demo's graph wiring.
"""No-key demo: run a LangGraph graph under GuardLoop.

The graph uses an in-process fake chat model (named like a priced model), so the
demo needs no API key. GuardLoop's budget caps and OpenTelemetry spans apply
*inside* the graph via ``guarded_graph(...)``; the first run succeeds, the second
runs under a deliberately tiny token budget and is stopped before the model call.

    uv run python examples/langgraph_guarded.py
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from guardloop import BudgetConfig, GuardLoop
from guardloop.adapters.langgraph import guarded_graph


class DemoChatModel(BaseChatModel):
    """A canned chat model that reports as ``openai`` / ``gpt-5.2`` for pricing."""

    @property
    def _llm_type(self) -> str:
        return "demo-fake-chat"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return LangSmithParams(ls_provider="openai", ls_model_type="chat", ls_model_name="gpt-5.2")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        answer = AIMessage(
            content="1) line up the audience, 2) write the changelog, 3) tag and publish",
            usage_metadata={"input_tokens": 64, "output_tokens": 32, "total_tokens": 96},
        )
        return ChatResult(generations=[ChatGeneration(message=answer)])


class PlannerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph() -> Any:
    def call_model(state: PlannerState) -> dict[str, list[BaseMessage]]:
        return {"messages": [DemoChatModel().invoke(state["messages"])]}

    graph = StateGraph(PlannerState)
    graph.add_node("planner", call_model)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    return graph.compile()


async def main() -> None:
    graph = build_graph()
    agent = guarded_graph(graph, input_key="messages")

    print("== run 1: comfortable budget ==")
    generous = GuardLoop(
        budget=BudgetConfig(cost_limit_usd="0.05", token_limit=5_000, tool_call_limit=5)
    )
    ok = await generous.run(agent, {"messages": [HumanMessage(content="plan a 3-step release")]})
    print(ok.model_dump_json(indent=2))

    print("\n== run 2: token budget too small for the call ==")
    tight = GuardLoop(budget=BudgetConfig(token_limit=10))
    blocked = await tight.run(agent, {"messages": [HumanMessage(content="plan a 3-step release")]})
    print(f"success={blocked.success}")
    print(f"terminated_reason={blocked.terminated_reason}")
    print(f"output={blocked.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
