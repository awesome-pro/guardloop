"""Tests for the OpenAI Agents SDK adapter (``guardloop.adapters.openai_agents``)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("agents")

from agents import RunConfig
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
    RunContext,
    TelemetryConfig,
    VerifierConfig,
    VerifierContext,
    VerifierResult,
)
from guardloop.adapters.openai_agents import GuardLoopRunHooks, guarded_runner
from tests.openai_agents_fakes import (
    UNPRICED_MODEL_NAME,
    NoModelNameModel,
    RaisingModel,
    ScriptedModel,
    echo_tool,
    exploding_tool,
    single_agent,
    text_response,
    tool_agent,
    tool_response,
)


def _runner(agent: Any, **kwargs: Any) -> Any:
    # Disable the SDK's own tracing so the no-key test environment stays quiet.
    return guarded_runner(agent, run_config=RunConfig(tracing_disabled=True), **kwargs)


async def test_guarded_runner_happy_path_records_usage_and_output() -> None:
    model = ScriptedModel([text_response("the plan", input_tokens=12, output_tokens=8)])
    runtime = GuardLoop(
        budget=BudgetConfig(cost_limit_usd="0.10", token_limit=10_000, tool_call_limit=5)
    )
    result = await runtime.run(_runner(single_agent(model)), "draft a plan")

    assert result.success is True
    assert result.output == "the plan"
    assert result.tokens_used == 20
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.cost_usd > 0
    assert model.call_count == 1


async def test_budget_cap_trips_inside_the_run() -> None:
    # Pre-flight (tiny input + 1024 reserved) is cheap, but the model reports 200k
    # output tokens, so record_llm_call blows the cost cap from inside on_llm_end.
    model = ScriptedModel([text_response("expensive", input_tokens=1_000, output_tokens=200_000)])
    runtime = GuardLoop(budget=BudgetConfig(cost_limit_usd="0.10"))
    result = await runtime.run(_runner(single_agent(model)), "go")

    assert result.success is False
    assert result.terminated_reason == "budget_exceeded"
    assert result.error_type == "BudgetExceeded"


async def test_reserved_output_tokens_is_enforced_pre_flight() -> None:
    model = ScriptedModel([text_response("never reached")])
    runtime = GuardLoop(budget=BudgetConfig(token_limit=50))
    result = await runtime.run(_runner(single_agent(model), reserved_output_tokens=200), "tiny")

    assert result.success is False
    assert result.terminated_reason == "token_limit_exceeded"
    assert model.call_count == 0


async def test_tool_call_limit_is_enforced_inside_the_run() -> None:
    model = ScriptedModel(
        [
            tool_response(
                "echo_tool", arguments='{"text": "one"}', input_tokens=10, output_tokens=4
            ),
            tool_response(
                "echo_tool", arguments='{"text": "two"}', input_tokens=10, output_tokens=4
            ),
            text_response("done"),
        ]
    )
    runtime = GuardLoop(budget=BudgetConfig(tool_call_limit=1))
    result = await runtime.run(_runner(tool_agent(model, [echo_tool])), "echo things")

    assert result.success is False
    assert result.terminated_reason == "tool_call_limit_exceeded"
    assert result.error_type == "ToolCallLimitExceeded"
    assert result.tool_calls == 1


async def test_open_circuit_breaker_blocks_a_tool_call_in_the_run() -> None:
    # The SDK has no tool-error hook, so the breaker can't open itself from an
    # SDK-managed tool failure; open it explicitly first (the v0.2 mechanism), then
    # confirm the open breaker rejects the SDK tool call via on_tool_start.
    runtime = GuardLoop(
        budget=BudgetConfig(tool_call_limit=10),
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=60)
        ),
    )

    async def open_breaker(ctx: RunContext) -> str:
        ctx.circuit_breakers.record_failure("echo_tool")
        return "opened"

    opened = await runtime.run(open_breaker)
    assert opened.success is True
    assert runtime.circuit_breaker_snapshots()["echo_tool"].state.value == "open"

    model = ScriptedModel(
        [tool_response("echo_tool", arguments='{"text": "x"}'), text_response("done")]
    )
    result = await runtime.run(_runner(tool_agent(model, [echo_tool])), "echo x")

    assert result.success is False
    assert result.terminated_reason == "circuit_breaker_open"
    assert result.error_type == "CircuitBreakerOpen"


async def test_verifier_loop_wraps_the_runner_and_injects_feedback() -> None:
    model = ScriptedModel([text_response("BAD draft"), text_response("GOOD final")])

    def must_say_good(output: object, ctx: VerifierContext) -> VerifierResult:
        if "GOOD" in str(output):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, feedback="include the word GOOD")

    runtime = GuardLoop(verifiers=[must_say_good], verifier_config=VerifierConfig(max_retries=1))
    result = await runtime.run(_runner(single_agent(model)), "write something")

    assert result.success is True
    assert result.verification_passed is True
    assert result.verification_attempts == 2
    assert result.output == "GOOD final"
    assert model.call_count == 2
    # the verifier feedback reached the model on the second attempt
    assert "include the word GOOD" in str(model.seen_inputs[1])


async def test_guarded_runner_does_not_mutate_caller_input() -> None:
    model = ScriptedModel([text_response("v1"), text_response("done now")])

    def needs_done(output: object, ctx: VerifierContext) -> VerifierResult:
        return VerifierResult(passed="done" in str(output), feedback="finish the task")

    runtime = GuardLoop(verifiers=[needs_done], verifier_config=VerifierConfig(max_retries=1))
    user_input: list[dict[str, str]] = [{"role": "user", "content": "start"}]

    result = await runtime.run(_runner(single_agent(model)), user_input)

    assert result.success is True
    assert len(user_input) == 1
    assert user_input == [{"role": "user", "content": "start"}]


async def test_custom_feedback_and_output_hooks_are_used() -> None:
    model = ScriptedModel([text_response("first"), text_response("second")])
    seen_feedback: list[list[str]] = []

    def feedback_to_input(agent_input: Any, feedback: list[str]) -> Any:
        seen_feedback.append(list(feedback))
        return [*agent_input, {"role": "user", "content": "retry: " + " / ".join(feedback)}]

    def output_from_result(result: Any) -> str:
        return f"answer={getattr(result, 'final_output', result)}"

    def reject_first(output: object, ctx: VerifierContext) -> VerifierResult:
        if ctx.attempt == 1:
            return VerifierResult(passed=False, feedback="try harder")
        return VerifierResult(passed=True)

    runtime = GuardLoop(verifiers=[reject_first], verifier_config=VerifierConfig(max_retries=1))
    agent = _runner(
        single_agent(model),
        feedback_to_input=feedback_to_input,
        output_from_result=output_from_result,
    )
    result = await runtime.run(agent, [{"role": "user", "content": "hello"}])

    assert result.success is True
    assert result.output == "answer=second"
    assert seen_feedback == [["try harder"]]


async def test_emits_llm_and_tool_spans_under_agent_run() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("guardloop-openai-agents-tests")

    model = ScriptedModel(
        [
            tool_response(
                "echo_tool", arguments='{"text": "hi"}', input_tokens=10, output_tokens=4
            ),
            text_response("wrapped up", input_tokens=6, output_tokens=3),
        ]
    )
    runtime = GuardLoop(
        budget=BudgetConfig(cost_limit_usd="1.00", tool_call_limit=5),
        telemetry=TelemetryConfig(enabled=True),
        tracer=tracer,
    )
    result = await runtime.run(_runner(tool_agent(model, [echo_tool])), "echo hi")

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


async def test_sdk_tool_failure_does_not_open_the_breaker() -> None:
    # The SDK has no on_tool_error hook and turns a tool exception into an error
    # string fed back to the model, so on_tool_end fires (recording a *success*)
    # and a flaky SDK-managed tool never opens the breaker on its own. This pins
    # that documented limitation.
    model = ScriptedModel(
        [tool_response("exploding_tool", arguments='{"text": "x"}'), text_response("recovered")]
    )
    runtime = GuardLoop(
        budget=BudgetConfig(tool_call_limit=5),
        circuit_breakers=CircuitBreakerConfig(
            default=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=60)
        ),
    )
    result = await runtime.run(_runner(tool_agent(model, [exploding_tool])), "use the tool")

    assert result.success is True
    assert result.output == "recovered"
    assert runtime.circuit_breaker_snapshots()["exploding_tool"].state.value == "closed"


async def test_model_error_propagates_and_records_no_usage() -> None:
    runtime = GuardLoop(budget=BudgetConfig(cost_limit_usd="1.00", token_limit=10_000))
    result = await runtime.run(_runner(single_agent(RaisingModel())), "go")

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert result.tokens_used == 0
    assert result.cost_usd == 0


async def test_unpriced_model_raises_a_clear_pricing_error() -> None:
    model = ScriptedModel([text_response("x")], name=UNPRICED_MODEL_NAME)
    runtime = GuardLoop(budget=BudgetConfig(token_limit=10_000))
    result = await runtime.run(_runner(single_agent(model)), "go")

    assert result.success is False
    assert result.error_type == "ModelPricingMissing"
    assert result.error_message is not None
    assert "pricing" in result.error_message.lower()


async def test_unresolvable_model_name_raises_a_clear_error() -> None:
    runtime = GuardLoop(budget=BudgetConfig(token_limit=10_000))
    result = await runtime.run(_runner(single_agent(NoModelNameModel())), "go")

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert result.error_message is not None
    assert "model name" in result.error_message


async def test_run_hooks_reject_non_positive_reserved_output_tokens() -> None:
    runtime = GuardLoop()

    async def agent(ctx: RunContext) -> object:
        GuardLoopRunHooks(ctx, reserved_output_tokens=0)
        return "unreachable"

    result = await runtime.run(agent)
    assert result.success is False
    assert result.error_type == "ValueError"


def test_guarded_runner_rejects_non_positive_reserved_output_tokens() -> None:
    with pytest.raises(ValueError, match="positive"):
        guarded_runner(single_agent(ScriptedModel()), reserved_output_tokens=0)
