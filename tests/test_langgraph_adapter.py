"""Tests for the LangGraph adapter (``guardloop.adapters.langgraph``)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from guardloop import (
    BudgetConfig,
    CircuitBreakerConfig,
    CircuitBreakerPolicy,
    GuardLoop,
    TelemetryConfig,
    VerifierConfig,
    VerifierContext,
    VerifierResult,
)
from guardloop.adapters.langgraph import GuardLoopCallbackHandler, guarded_graph
from tests.langchain_fakes import (
    ChatState,
    RaisingChatModel,
    ScriptedChatModel,
    async_single_node_graph,
    echo_tool,
    exploding_tool,
    fake_ai_message,
    react_loop_graph,
    single_node_graph,
    tool_call,
)


def _start(text: str = "hello") -> dict[str, list[Any]]:
    return {"messages": [HumanMessage(content=text)]}


async def test_guarded_graph_happy_path_records_usage_and_output() -> None:
    model = ScriptedChatModel(
        scripted=[fake_ai_message("the plan", input_tokens=12, output_tokens=8)]
    )
    runtime = GuardLoop(
        budget=BudgetConfig(cost_limit_usd="0.10", token_limit=10_000, tool_call_limit=5)
    )
    result = await runtime.run(guarded_graph(single_node_graph(model)), _start())

    assert result.success is True
    assert result.output == "the plan"
    assert result.tokens_used == 20
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.cost_usd > 0
    assert model.call_count == 1


async def test_guarded_graph_works_with_async_nodes() -> None:
    model = ScriptedChatModel(
        scripted=[fake_ai_message("async answer", input_tokens=4, output_tokens=2)]
    )
    runtime = GuardLoop(budget=BudgetConfig(cost_limit_usd="0.10", token_limit=10_000))
    result = await runtime.run(guarded_graph(async_single_node_graph(model)), _start())

    assert result.success is True
    assert result.output == "async answer"
    assert result.tokens_used == 6


async def test_budget_cap_trips_inside_the_graph() -> None:
    # The pre-flight estimate (tiny input + 1024 reserved) is cheap, but the model reports
    # 200k output tokens, so record_llm_call blows the cost cap from inside the graph.
    model = ScriptedChatModel(
        scripted=[fake_ai_message("expensive answer", input_tokens=1_000, output_tokens=200_000)]
    )
    runtime = GuardLoop(budget=BudgetConfig(cost_limit_usd="0.10"))
    result = await runtime.run(guarded_graph(single_node_graph(model)), _start())

    assert result.success is False
    assert result.terminated_reason == "budget_exceeded"
    assert result.error_type == "BudgetExceeded"


async def test_reserved_output_tokens_is_enforced_pre_flight() -> None:
    model = ScriptedChatModel(scripted=[fake_ai_message("never reached")])
    runtime = GuardLoop(budget=BudgetConfig(token_limit=50))
    result = await runtime.run(
        guarded_graph(single_node_graph(model), reserved_output_tokens=200), _start("tiny")
    )

    assert result.success is False
    assert result.terminated_reason == "token_limit_exceeded"
    assert model.call_count == 0


async def test_tool_call_limit_is_enforced_inside_the_graph() -> None:
    model = ScriptedChatModel(
        scripted=[
            fake_ai_message("", tool_calls=[tool_call("echo_tool", text="one")]),
            fake_ai_message("", tool_calls=[tool_call("echo_tool", text="two")]),
            fake_ai_message("done"),
        ]
    )
    graph = react_loop_graph(model, [echo_tool], handle_tool_errors=False)
    runtime = GuardLoop(budget=BudgetConfig(tool_call_limit=1))
    result = await runtime.run(guarded_graph(graph), _start("go"))

    assert result.success is False
    assert result.terminated_reason == "tool_call_limit_exceeded"
    assert result.tool_calls == 1


async def test_circuit_breaker_opens_for_a_failing_tool_in_the_graph() -> None:
    runtime = GuardLoop(
        budget=BudgetConfig(tool_call_limit=10),
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=60)
        ),
    )

    def failing_graph() -> Any:
        model = ScriptedChatModel(
            scripted=[
                fake_ai_message("", tool_calls=[tool_call("exploding_tool", text="x")]),
                fake_ai_message("done"),
            ]
        )
        return react_loop_graph(model, [exploding_tool], handle_tool_errors=False)

    first = await runtime.run(guarded_graph(failing_graph()), _start("go"))
    assert first.success is False
    assert first.error_type == "RuntimeError"
    assert runtime.circuit_breaker_snapshots()["exploding_tool"].state.value == "open"

    second = await runtime.run(guarded_graph(failing_graph()), _start("go"))
    assert second.success is False
    assert second.terminated_reason == "circuit_breaker_open"
    assert second.error_type == "CircuitBreakerOpen"


async def test_verifier_loop_wraps_the_graph_and_injects_feedback() -> None:
    model = ScriptedChatModel(
        scripted=[fake_ai_message("rough draft"), fake_ai_message("GOOD final")]
    )

    def must_say_good(output: object, ctx: VerifierContext) -> VerifierResult:
        if "GOOD" in str(output):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, feedback="include the word GOOD")

    runtime = GuardLoop(verifiers=[must_say_good], verifier_config=VerifierConfig(max_retries=1))
    result = await runtime.run(guarded_graph(single_node_graph(model)), _start("write something"))

    assert result.success is True
    assert result.verification_passed is True
    assert result.verification_attempts == 2
    assert result.output == "GOOD final"
    assert model.call_count == 2
    # the verifier's feedback was injected as a HumanMessage on the second attempt
    second_attempt_inputs = model.seen_messages[1]
    assert any(
        isinstance(message, HumanMessage)
        and "include the word GOOD" in str(getattr(message, "content", ""))
        for message in second_attempt_inputs
    )


async def test_guarded_graph_does_not_mutate_caller_state() -> None:
    model = ScriptedChatModel(scripted=[fake_ai_message("v1"), fake_ai_message("ok done")])

    def needs_done(output: object, ctx: VerifierContext) -> VerifierResult:
        return VerifierResult(passed="done" in str(output), feedback="finish the task")

    runtime = GuardLoop(verifiers=[needs_done], verifier_config=VerifierConfig(max_retries=1))
    state: dict[str, list[Any]] = {"messages": [HumanMessage(content="start")]}
    original_messages = state["messages"]

    result = await runtime.run(guarded_graph(single_node_graph(model)), state)

    assert result.success is True
    assert state["messages"] is original_messages
    assert len(state["messages"]) == 1


async def test_custom_feedback_and_output_hooks_are_used() -> None:
    model = ScriptedChatModel(scripted=[fake_ai_message("first"), fake_ai_message("second")])
    seen_feedback: list[list[str]] = []

    def feedback_to_state(state: Any, feedback: list[str]) -> Any:
        seen_feedback.append(list(feedback))
        new_messages = [*state["messages"], HumanMessage(content="retry: " + " / ".join(feedback))]
        return {**state, "messages": new_messages}

    def reject_first(output: object, ctx: VerifierContext) -> VerifierResult:
        if ctx.attempt == 1:
            return VerifierResult(passed=False, feedback="try harder")
        return VerifierResult(passed=True)

    runtime = GuardLoop(verifiers=[reject_first], verifier_config=VerifierConfig(max_retries=1))
    agent = guarded_graph(
        single_node_graph(model),
        feedback_to_state=feedback_to_state,
        output_from_state=lambda final_state: f"answer={final_state['messages'][-1].content}",
    )
    result = await runtime.run(agent, _start("hello"))

    assert result.success is True
    assert result.output == "answer=second"
    assert seen_feedback == [["try harder"]]


async def test_emits_llm_and_tool_spans_under_agent_run() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("guardloop-langgraph-tests")

    model = ScriptedChatModel(
        scripted=[
            fake_ai_message(
                "", input_tokens=10, output_tokens=4, tool_calls=[tool_call("echo_tool", text="hi")]
            ),
            fake_ai_message("wrapped up", input_tokens=6, output_tokens=3),
        ]
    )
    graph = react_loop_graph(model, [echo_tool], handle_tool_errors=False)
    runtime = GuardLoop(
        budget=BudgetConfig(cost_limit_usd="1.00", tool_call_limit=5),
        telemetry=TelemetryConfig(enabled=True),
        tracer=tracer,
    )
    result = await runtime.run(guarded_graph(graph), _start("go"))

    assert result.success is True
    assert result.tokens_used == 23

    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert "agent_run" in names
    assert "llm_call openai.chat" in names
    assert "tool_call echo_tool" in names

    agent_span = next(span for span in spans if span.name == "agent_run")
    llm_span = next(span for span in spans if span.name == "llm_call openai.chat")
    tool_span = next(span for span in spans if span.name == "tool_call echo_tool")
    agent_context = agent_span.context
    llm_parent = llm_span.parent
    tool_parent = tool_span.parent
    assert agent_context is not None
    assert llm_parent is not None and llm_parent.span_id == agent_context.span_id
    assert tool_parent is not None and tool_parent.span_id == agent_context.span_id

    llm_attributes = llm_span.attributes
    assert llm_attributes is not None
    assert llm_attributes["gen_ai.system"] == "openai"
    assert llm_attributes["gen_ai.request.model"] == "gpt-5.2"

    tool_attributes_seen = tool_span.attributes
    assert tool_attributes_seen is not None
    assert tool_attributes_seen["guardloop.tool.name"] == "echo_tool"


async def test_model_error_propagates_and_records_no_usage() -> None:
    runtime = GuardLoop(budget=BudgetConfig(cost_limit_usd="1.00", token_limit=10_000))
    result = await runtime.run(guarded_graph(single_node_graph(RaisingChatModel())), _start())

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert result.tokens_used == 0
    assert result.cost_usd == 0


async def test_unknown_model_name_raises_a_clear_error() -> None:
    model = ScriptedChatModel(scripted=[fake_ai_message("x")])
    # Blank the model name so it cannot be resolved from callback metadata.
    model.chat_model_name = ""
    runtime = GuardLoop(budget=BudgetConfig(token_limit=10_000))
    result = await runtime.run(guarded_graph(single_node_graph(model)), _start())

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert result.error_message is not None
    assert "model name" in result.error_message


async def test_callback_handler_rejects_non_positive_reserved_output_tokens() -> None:
    runtime = GuardLoop()

    async def agent(ctx: Any) -> object:
        GuardLoopCallbackHandler(ctx, reserved_output_tokens=0)
        return "unreachable"

    result = await runtime.run(agent)
    assert result.success is False
    assert result.error_type == "ValueError"


def test_guarded_graph_rejects_non_positive_reserved_output_tokens() -> None:
    with pytest.raises(ValueError, match="positive"):
        guarded_graph(single_node_graph(ScriptedChatModel()), reserved_output_tokens=0)


def test_chat_state_is_a_typed_dict() -> None:
    # Smoke check that the fakes module exposes the state schema used by the graphs.
    assert "messages" in ChatState.__annotations__


async def test_aimessage_round_trips_through_default_output() -> None:
    # An AIMessage with structured content still yields a usable RunResult.output.
    model = ScriptedChatModel(
        scripted=[
            AIMessage(
                content="structured",
                usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            )
        ]
    )
    runtime = GuardLoop(budget=BudgetConfig(token_limit=10_000))
    result = await runtime.run(guarded_graph(single_node_graph(model)), _start())
    assert result.output == "structured"
