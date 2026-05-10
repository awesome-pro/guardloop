from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from guardloop import (
    BudgetConfig,
    GuardLoop,
    RunContext,
    TelemetryConfig,
    VerifierConfig,
    VerifierContext,
    VerifierResult,
    is_json_object,
    matches_regex,
    non_empty,
)


async def test_verifier_passes_on_first_attempt() -> None:
    calls = 0

    def always_passes(output: object, ctx: VerifierContext) -> VerifierResult:
        return VerifierResult(passed=True)

    runtime = GuardLoop(verifiers=[always_passes])

    async def agent(ctx: RunContext) -> str:
        nonlocal calls
        calls += 1
        return "fine"

    result = await runtime.run(agent)

    assert result.success is True
    assert result.verification_passed is True
    assert result.verification_attempts == 1
    assert result.verification_feedback == []
    assert result.output == "fine"
    assert result.terminated_reason is None
    assert calls == 1


async def test_verifier_fails_then_passes_after_agent_self_corrects() -> None:
    runtime = GuardLoop(
        verifiers=[is_json_object(required_keys=["answer"])],
        verifier_config=VerifierConfig(max_retries=1),
    )
    attempt = 0

    async def agent(ctx: RunContext) -> str:
        nonlocal attempt
        attempt += 1
        if ctx.retry_feedback:
            return '{"answer": 42}'
        return "not json at all"

    result = await runtime.run(agent)

    assert result.success is True
    assert result.verification_passed is True
    assert result.verification_attempts == 2
    assert result.output == '{"answer": 42}'
    assert len(result.verification_feedback) == 1
    assert "JSON" in result.verification_feedback[0]
    assert attempt == 2


async def test_verifier_failing_all_retries_returns_unverified_result() -> None:
    def always_fails(output: object, ctx: VerifierContext) -> VerifierResult:
        return VerifierResult(passed=False, feedback="nope")

    runtime = GuardLoop(verifiers=[always_fails], verifier_config=VerifierConfig(max_retries=1))

    async def agent(ctx: RunContext) -> str:
        return "the answer"

    result = await runtime.run(agent)

    assert result.success is False
    assert result.verification_passed is False
    assert result.terminated_reason == "verification_failed"
    assert result.verification_attempts == 2
    assert result.output == "the answer"
    assert result.verification_feedback == ["nope", "nope"]
    assert result.error_type is None


async def test_verifier_max_retries_zero_runs_agent_once() -> None:
    calls = 0

    def always_fails(output: object, ctx: VerifierContext) -> VerifierResult:
        return VerifierResult(passed=False, feedback="no")

    runtime = GuardLoop(verifiers=[always_fails], verifier_config=VerifierConfig(max_retries=0))

    async def agent(ctx: RunContext) -> str:
        nonlocal calls
        calls += 1
        return "x"

    result = await runtime.run(agent)

    assert calls == 1
    assert result.verification_attempts == 1
    assert result.verification_passed is False
    assert result.terminated_reason == "verification_failed"
    assert result.success is False
    assert result.output == "x"


async def test_verifier_chain_is_fail_fast() -> None:
    second_calls = 0

    def first(output: object, ctx: VerifierContext) -> VerifierResult:
        if ctx.attempt == 1:
            return VerifierResult(passed=False, feedback="retry")
        return VerifierResult(passed=True)

    def second(output: object, ctx: VerifierContext) -> VerifierResult:
        nonlocal second_calls
        second_calls += 1
        return VerifierResult(passed=True)

    runtime = GuardLoop(verifiers=[first, second], verifier_config=VerifierConfig(max_retries=1))

    async def agent(ctx: RunContext) -> str:
        return "out"

    result = await runtime.run(agent)

    assert result.verification_passed is True
    assert result.verification_attempts == 2
    assert second_calls == 1


async def test_verifier_chain_all_pass_runs_agent_once() -> None:
    runtime = GuardLoop(verifiers=[non_empty(), matches_regex(r"\d")])

    async def agent(ctx: RunContext) -> str:
        return "answer 7"

    result = await runtime.run(agent)

    assert result.success is True
    assert result.verification_passed is True
    assert result.verification_attempts == 1


async def test_sync_bool_verifier_and_generated_feedback() -> None:
    def has_no_todo(output: object, ctx: VerifierContext) -> bool:
        return "TODO" not in str(output)

    runtime = GuardLoop(verifiers=[has_no_todo], verifier_config=VerifierConfig(max_retries=1))
    attempt = 0

    async def agent(ctx: RunContext) -> str:
        nonlocal attempt
        attempt += 1
        return "TODO" if attempt == 1 else "done"

    result = await runtime.run(agent)

    assert result.success is True
    assert result.verification_passed is True
    assert result.verification_attempts == 2
    assert result.output == "done"
    assert len(result.verification_feedback) == 1
    assert "no feedback provided" in result.verification_feedback[0]


async def test_async_verifier_is_supported() -> None:
    async def slow_check(output: object, ctx: VerifierContext) -> VerifierResult:
        await asyncio.sleep(0)
        if "ok" in str(output):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, feedback="say ok")

    runtime = GuardLoop(verifiers=[slow_check], verifier_config=VerifierConfig(max_retries=1))
    attempt = 0

    async def agent(ctx: RunContext) -> str:
        nonlocal attempt
        attempt += 1
        return "bad" if attempt == 1 else "ok"

    result = await runtime.run(agent)

    assert result.verification_passed is True
    assert result.verification_attempts == 2


async def test_verifier_exception_terminates_run_without_retry() -> None:
    calls = 0

    def buggy(output: object, ctx: VerifierContext) -> VerifierResult:
        raise ValueError("verifier bug")

    runtime = GuardLoop(verifiers=[buggy], verifier_config=VerifierConfig(max_retries=3))

    async def agent(ctx: RunContext) -> str:
        nonlocal calls
        calls += 1
        return "x"

    result = await runtime.run(agent)

    assert result.success is False
    assert result.terminated_reason == "verifier_error"
    assert result.error_type == "VerifierExecutionError"
    assert calls == 1
    assert result.verification_attempts == 1


async def test_budget_is_shared_across_retry_attempts() -> None:
    def always_fails(output: object, ctx: VerifierContext) -> VerifierResult:
        return VerifierResult(passed=False, feedback="again")

    runtime = GuardLoop(
        budget=BudgetConfig(tool_call_limit=2),
        verifiers=[always_fails],
        verifier_config=VerifierConfig(max_retries=3),
    )

    def noop_tool() -> str:
        return "ok"

    async def agent(ctx: RunContext) -> str:
        await ctx.call_tool("t", noop_tool)
        await ctx.call_tool("t", noop_tool)
        return "done"

    result = await runtime.run(agent)

    assert result.success is False
    assert result.terminated_reason == "tool_call_limit_exceeded"
    assert result.error_type == "ToolCallLimitExceeded"
    assert result.tool_calls == 2
    assert result.verification_attempts == 2
    assert result.verification_feedback == ["again"]


async def test_run_timeout_bounds_the_whole_retry_loop() -> None:
    attempts = 0

    def always_fails(output: object, ctx: VerifierContext) -> VerifierResult:
        return VerifierResult(passed=False, feedback="again")

    runtime = GuardLoop(
        budget=BudgetConfig(time_limit_seconds=0.1),
        verifiers=[always_fails],
        verifier_config=VerifierConfig(max_retries=5),
    )

    async def agent(ctx: RunContext) -> str:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.06)
        return "slow"

    result = await runtime.run(agent)

    assert result.success is False
    assert result.terminated_reason == "timeout"
    assert result.error_type == "TimeoutError"
    assert attempts < 6


async def test_retry_feedback_is_recorded_and_visible_to_agent() -> None:
    seen: list[list[str]] = []

    def fails_twice(output: object, ctx: VerifierContext) -> VerifierResult:
        if ctx.attempt < 3:
            return VerifierResult(passed=False, feedback=f"fix-{ctx.attempt}")
        return VerifierResult(passed=True)

    runtime = GuardLoop(verifiers=[fails_twice], verifier_config=VerifierConfig(max_retries=3))

    async def agent(ctx: RunContext) -> str:
        seen.append(list(ctx.retry_feedback))
        return "out"

    result = await runtime.run(agent)

    assert result.verification_passed is True
    assert result.verification_attempts == 3
    assert seen == [[], ["fix-1"], ["fix-1", "fix-2"]]
    assert result.verification_feedback == ["fix-1", "fix-2"]


async def test_verifier_run_spans_carry_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("guardloop-verifier-tests")

    calls = 0

    def flaky_check(output: object, ctx: VerifierContext) -> VerifierResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return VerifierResult(passed=False, feedback="try again")
        return VerifierResult(passed=True)

    runtime = GuardLoop(
        telemetry=TelemetryConfig(enabled=True),
        verifiers=[flaky_check],
        verifier_config=VerifierConfig(max_retries=1),
        tracer=tracer,
    )

    async def agent(ctx: RunContext) -> str:
        return "out"

    result = await runtime.run(agent)
    spans = exporter.get_finished_spans()

    verifier_spans = [span for span in spans if span.name == "verifier_run flaky_check"]
    assert len(verifier_spans) == 2
    first_attrs = verifier_spans[0].attributes
    assert first_attrs is not None
    assert first_attrs["guardloop.verifier.name"] == "flaky_check"
    assert first_attrs["guardloop.verifier.attempt"] == 1
    assert first_attrs["guardloop.verifier.max_attempts"] == 2
    assert first_attrs["guardloop.verifier.passed"] is False
    second_attrs = verifier_spans[1].attributes
    assert second_attrs is not None
    assert second_attrs["guardloop.verifier.attempt"] == 2
    assert second_attrs["guardloop.verifier.passed"] is True

    agent_span = next(span for span in spans if span.name == "agent_run")
    agent_attrs = agent_span.attributes
    assert agent_attrs is not None
    assert agent_attrs["guardloop.verification.passed"] is True
    assert agent_attrs["guardloop.verification.attempts"] == 2
    event_names = {event.name for event in agent_span.events}
    assert "guardloop.verification.failed" in event_names

    assert result.verification_passed is True


async def test_disabled_verifier_config_skips_verification() -> None:
    calls = 0

    def always_fails(output: object, ctx: VerifierContext) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(passed=False, feedback="no")

    runtime = GuardLoop(verifiers=[always_fails], verifier_config=VerifierConfig(enabled=False))

    async def agent(ctx: RunContext) -> str:
        return "out"

    result = await runtime.run(agent)

    assert calls == 0
    assert result.success is True
    assert result.verification_passed is None
    assert result.verification_attempts == 1
    assert result.terminated_reason is None


async def test_no_verifiers_means_no_verification() -> None:
    runtime = GuardLoop()

    async def agent(ctx: RunContext) -> str:
        return "out"

    result = await runtime.run(agent)

    assert result.success is True
    assert result.verification_passed is None
    assert result.verification_attempts == 1
    assert result.verification_feedback == []


async def test_strict_mode_surfaces_verification_failed() -> None:
    def always_fails(output: object, ctx: VerifierContext) -> VerifierResult:
        return VerifierResult(passed=False, feedback="bad output")

    runtime = GuardLoop(
        verifiers=[always_fails],
        verifier_config=VerifierConfig(max_retries=0, raise_on_failure=True),
    )

    async def agent(ctx: RunContext) -> str:
        return "the answer"

    result = await runtime.run(agent)

    assert result.success is False
    assert result.terminated_reason == "verification_failed"
    assert result.error_type == "VerificationFailed"
    assert result.error_message is not None
    assert "attempt" in result.error_message
    assert result.verification_passed is False
    assert result.verification_attempts == 1
    assert result.output is None
    assert result.verification_feedback == ["bad output"]
    details = result.metadata["details"]
    assert details["attempts"] == 1
    assert details["verifier_name"] == "always_fails"
    assert "bad output" in details["feedback"]


async def test_pass_feedback_to_agent_false_keeps_ctx_feedback_empty() -> None:
    seen: list[list[str]] = []

    def fails_once(output: object, ctx: VerifierContext) -> VerifierResult:
        if ctx.attempt == 1:
            return VerifierResult(passed=False, feedback="hidden")
        return VerifierResult(passed=True)

    runtime = GuardLoop(
        verifiers=[fails_once],
        verifier_config=VerifierConfig(max_retries=1, pass_feedback_to_agent=False),
    )

    async def agent(ctx: RunContext) -> str:
        seen.append(list(ctx.retry_feedback))
        return "out"

    result = await runtime.run(agent)

    assert seen == [[], []]
    assert result.verification_passed is True
    assert result.verification_attempts == 2
    assert result.verification_feedback == ["hidden"]


async def test_verifier_context_carries_run_arguments_and_counts() -> None:
    captured: list[VerifierContext] = []

    def capturing(output: object, ctx: VerifierContext) -> VerifierResult:
        captured.append(ctx)
        return VerifierResult(passed=ctx.attempt == 2)

    runtime = GuardLoop(verifiers=[capturing], verifier_config=VerifierConfig(max_retries=1))

    async def agent(ctx: RunContext, question: str, *, lang: str = "en") -> str:
        return f"{question}:{lang}"

    result = await runtime.run(agent, "hello", lang="fr")

    assert result.verification_passed is True
    assert len(captured) == 2
    first = captured[0]
    assert first.output == "hello:fr"
    assert first.attempt == 1
    assert first.max_attempts == 2
    assert first.retries_remaining == 1
    assert first.prior_feedback == ()
    assert first.run_args == ("hello",)
    assert first.run_kwargs == {"lang": "fr"}
    assert captured[1].attempt == 2
    assert captured[1].retries_remaining == 0


async def test_add_verifier_extends_the_chain() -> None:
    calls = 0

    def spy(output: object, ctx: VerifierContext) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(passed=True)

    runtime = GuardLoop()
    runtime.add_verifier(spy)

    async def agent(ctx: RunContext) -> str:
        return "out"

    result = await runtime.run(agent)

    assert calls == 1
    assert result.verification_passed is True


async def test_builtin_verifiers_reject_then_accept() -> None:
    runtime = GuardLoop(
        verifiers=[non_empty(), matches_regex(r"\d")],
        verifier_config=VerifierConfig(max_retries=2),
    )
    attempt = 0

    async def agent(ctx: RunContext) -> str:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return "   "
        if attempt == 2:
            return "no digits here"
        return "answer 42"

    result = await runtime.run(agent)

    assert result.verification_passed is True
    assert result.verification_attempts == 3
    assert len(result.verification_feedback) == 2
    assert "empty" in result.verification_feedback[0].lower()
    assert "pattern" in result.verification_feedback[1].lower()


async def test_is_json_object_verifier_rejects_until_object_with_key() -> None:
    runtime = GuardLoop(
        verifiers=[is_json_object(required_keys=["answer"])],
        verifier_config=VerifierConfig(max_retries=2),
    )
    attempt = 0

    async def agent(ctx: RunContext) -> str:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return "totally not json"
        if attempt == 2:
            return '{"other": 1}'
        return '{"answer": 42}'

    result = await runtime.run(agent)

    assert result.verification_passed is True
    assert result.verification_attempts == 3
    assert "JSON" in result.verification_feedback[0]
    assert "answer" in result.verification_feedback[1]


def test_verifier_config_rejects_negative_max_retries() -> None:
    with pytest.raises(ValidationError):
        VerifierConfig(max_retries=-1)
