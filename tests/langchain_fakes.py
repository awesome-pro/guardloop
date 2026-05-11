# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# LangGraph's StateGraph builder methods carry partially-unparameterised generics
# in their stubs; strict "unknown type" checks are relaxed for these test fakes.
"""LangChain fakes for the LangGraph adapter tests.

Imported only by ``tests/test_langgraph_adapter.py`` (which skips when the
``langgraph`` extra is not installed), so the rest of the test suite never
touches ``langchain_core``.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import Field


def fake_ai_message(
    content: str = "ok",
    *,
    input_tokens: int = 5,
    output_tokens: int = 3,
    tool_calls: list[dict[str, Any]] | None = None,
) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=list(tool_calls or []),
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def tool_call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": dict(args), "id": f"call_{name}", "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of ``AIMessage``s (repeating the last once exhausted).

    Reports as provider ``"openai"`` with ``chat_model_name`` so GuardLoop's
    default pricing catalog resolves it without extra configuration.
    """

    chat_model_name: str = "gpt-5.2"
    scripted: list[AIMessage] = Field(default_factory=list)
    call_count: int = 0
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake-chat"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return LangSmithParams(
            ls_provider="openai", ls_model_type="chat", ls_model_name=self.chat_model_name
        )

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen_messages.append(list(messages))
        self.call_count += 1
        if not self.scripted:
            return fake_ai_message()
        return self.scripted[min(self.call_count - 1, len(self.scripted) - 1)]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message(messages))])


class RaisingChatModel(BaseChatModel):
    """A chat model whose generation always raises (drives the ``on_llm_error`` path)."""

    @property
    def _llm_type(self) -> str:
        return "raising-fake-chat"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return LangSmithParams(ls_provider="openai", ls_model_type="chat", ls_model_name="gpt-5.2")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("model exploded")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("model exploded")


@tool
def echo_tool(text: str) -> str:
    """Echo the given text back."""

    return f"echo: {text}"


@tool
def exploding_tool(text: str) -> str:
    """A tool that always raises."""

    raise RuntimeError("upstream tool exploded")


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def single_node_graph(model: BaseChatModel) -> Any:
    """A one-node graph that calls ``model`` once and returns its message."""

    def call_model(state: ChatState) -> dict[str, list[BaseMessage]]:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(ChatState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


def async_single_node_graph(model: BaseChatModel) -> Any:
    """A one-node graph whose node awaits ``model.ainvoke`` (async-node path)."""

    async def call_model(state: ChatState) -> dict[str, list[BaseMessage]]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(ChatState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


def react_loop_graph(
    model: BaseChatModel, tools: list[Any], *, handle_tool_errors: bool = False
) -> Any:
    """A model<->tools loop graph (a minimal ReAct shape) for tool tests."""

    def call_model(state: ChatState) -> dict[str, list[BaseMessage]]:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(ChatState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=handle_tool_errors))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile()
