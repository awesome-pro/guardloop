# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# OpenAI Agents SDK hook payloads (``RunContextWrapper`` / ``ModelResponse`` /
# the ``Tool`` union / Responses-API input items) are loosely typed at this
# boundary, so strict "unknown type" checks are relaxed for this module; the
# public surface (``guarded_runner`` / ``GuardLoopRunHooks``) is fully typed.
"""Run an OpenAI Agents SDK ``Agent`` under a GuardLoop ``RunContext``.

``guarded_runner(agent)`` returns a GuardLoop-compatible agent callable you pass
to :meth:`guardloop.GuardLoop.run`. The SDK's ``Agent`` runs via ``Runner.run``,
whose model calls do not flow through GuardLoop's ``ctx.openai`` /
``ctx.anthropic`` wrappers; instead this adapter binds a :class:`RunHooks`
subclass to the ``RunContext`` so budget caps, per-tool circuit breakers, and
OpenTelemetry spans apply *inside* ``Runner.run``. The verifier retry loop wraps
the whole run: on a rejected output the runtime re-invokes the agent, which
re-runs ``Runner.run`` with the verifier feedback injected into a copy of the run
input.

``RunHooks`` methods are natively ``async``, so guardrail exceptions raised inside
them propagate out of ``Runner.run`` (the SDK wraps exceptions raised from the
tool hooks in ``agents.exceptions.UserError``, so :func:`guarded_runner` unwraps a
:class:`~guardloop.GuardLoopError` from the cause chain before re-raising).

Requires the ``openai-agents`` extra: ``pip install "guardloop[openai-agents]"``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from opentelemetry.trace import Span

from guardloop.context import RunContext
from guardloop.exceptions import GuardLoopError
from guardloop.runtime import AgentCallable
from guardloop.telemetry.conventions import (
    llm_request_attributes,
    llm_response_attributes,
    tool_attributes,
)
from guardloop.telemetry.tracer import Telemetry
from guardloop.tokenization import (
    estimate_anthropic_tokens,
    estimate_openai_tokens,
    payload_to_text,
)

try:
    from agents import Agent, ModelResponse, RunContextWrapper, RunHooks, Runner, Tool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'guardloop\'s OpenAI Agents SDK adapter requires the "openai-agents" extra: '
        'pip install "guardloop[openai-agents]"'
    ) from exc

try:  # the default-model resolver moved around between SDK versions; degrade gracefully
    from agents.models import get_default_model as _sdk_get_default_model
except ImportError:  # pragma: no cover

    def _sdk_get_default_model() -> str | None:
        return None


__all__ = ["GuardLoopRunHooks", "guarded_runner"]

DEFAULT_RESERVED_OUTPUT_TOKENS = 1024


@dataclass
class _LLMSpanEntry:
    provider: str
    model: str
    span: Span | None = None


@dataclass
class _ToolSpanEntry:
    tool_name: str
    span: Span | None = None


class GuardLoopRunHooks(RunHooks):  # type: ignore[type-arg]  # RunHooks is RunHooksBase[TContext, Agent]
    """OpenAI Agents SDK ``RunHooks`` that enforce a GuardLoop ``RunContext``.

    Bound to a single :class:`~guardloop.RunContext`, it runs a pre-flight budget
    check before each LLM call (``on_llm_start``), records actual usage afterward
    (``on_llm_end``), and routes tool invocations through the per-tool circuit
    breaker and the tool-call budget (``on_tool_start`` / ``on_tool_end``). Each
    LLM and tool call gets an OpenTelemetry span that is a child of the active
    ``agent_run`` span.

    Pass this to ``Runner.run(agent, input, hooks=GuardLoopRunHooks(ctx))`` from
    inside a GuardLoop agent if you need finer control than :func:`guarded_runner`
    provides; otherwise use :func:`guarded_runner`.

    Note: the SDK has no ``on_tool_error`` lifecycle hook and, by default, turns a
    tool exception into an error *string* fed back to the model, so the breaker
    here records tool *attempts* (``on_tool_start``) and *successes*
    (``on_tool_end``) but not *failures* — a flaky SDK-managed tool will not open
    the breaker on its own (route the tool body through ``ctx.call_tool(...)`` for
    full breaker semantics). The breaker's blocking behaviour (an already-open
    breaker rejects the next tool call) does apply.
    """

    def __init__(
        self, ctx: RunContext, *, reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS
    ) -> None:
        super().__init__()
        if reserved_output_tokens <= 0:
            raise ValueError("reserved_output_tokens must be a positive integer")
        self._ctx = ctx
        self._budget = ctx.budget
        self._telemetry: Telemetry = ctx.telemetry
        self._reserved_output_tokens = reserved_output_tokens
        self._tracing = ctx.telemetry.config.enabled
        # The SDK gives hooks no per-call id. One model call runs at a time within
        # a turn (a verifier retry gets a fresh GuardLoopRunHooks), so a single
        # slot is enough; assert it on entry so a violated assumption fails loudly.
        self._llm_span: _LLMSpanEntry | None = None
        # Tool calls can run in parallel within a turn and the same tool object can
        # appear twice, so bucket spans by id(tool) and pop FIFO within the bucket.
        self._tool_spans: dict[int, list[_ToolSpanEntry]] = {}

    # -- LLM ---------------------------------------------------------------

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[Any] | None,
    ) -> None:
        model = _resolve_model(agent)
        if model is None:
            raise RuntimeError(
                "GuardLoop's OpenAI Agents adapter could not determine the LLM model name from "
                "the agent. Set agent.model to a string model name, or register pricing for the "
                "model via GuardLoop(pricing=[ModelPricing(...)])."
            )
        provider = _resolve_provider(agent, model)
        reserved = _resolve_reserved_output_tokens(agent, self._reserved_output_tokens)
        payload = [system_prompt or "", *(payload_to_text(item) for item in input_items or [])]
        estimated_input = (
            estimate_anthropic_tokens(payload)
            if provider == "anthropic"
            else estimate_openai_tokens(model, payload)
        )
        preflight = self._budget.check_llm_call(
            provider=provider,
            model=model,
            estimated_input_tokens=estimated_input,
            reserved_output_tokens=reserved,
        )
        if self._llm_span is not None:  # pragma: no cover - defensive: see __init__ note
            self._end_span(self._llm_span.span)
        span: Span | None = None
        if self._tracing:
            span = self._telemetry.tracer.start_span(
                f"llm_call {provider}.chat",
                attributes=llm_request_attributes(
                    provider=provider,
                    model=model,
                    estimated_input_tokens=estimated_input,
                    reserved_output_tokens=reserved,
                    estimated_cost_usd=preflight.estimated_cost_usd,
                ),
            )
        self._llm_span = _LLMSpanEntry(provider=provider, model=model, span=span)

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        entry = self._llm_span
        self._llm_span = None
        if entry is None:
            return
        try:
            input_tokens, output_tokens = _extract_usage(response)
            cost = self._budget.record_llm_call(
                provider=entry.provider,
                model=entry.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if entry.span is not None:
                self._telemetry.set_attributes(
                    entry.span,
                    llm_response_attributes(
                        model=entry.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost,
                    ),
                )
                self._telemetry.mark_ok(entry.span)
        except BaseException as exc:
            if entry.span is not None:
                self._telemetry.record_exception(entry.span, exc)
            raise
        finally:
            self._end_span(entry.span)

    # -- Tools -------------------------------------------------------------

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Tool
    ) -> None:
        name = _tool_name(tool)
        span: Span | None = None
        if self._tracing:
            span = self._telemetry.tracer.start_span(
                f"tool_call {name}",
                attributes=tool_attributes(tool_name=name, calls_used=self._budget.tool_calls),
            )
        try:
            decision = self._ctx.circuit_breakers.before_call(name)
            if span is not None and decision is not None:
                self._set_breaker_attributes(span, decision)
            self._budget.record_tool_call_started(name)
            if span is not None:
                self._telemetry.set_attributes(
                    span, tool_attributes(tool_name=name, calls_used=self._budget.tool_calls)
                )
        except BaseException as exc:
            if span is not None:
                self._telemetry.record_exception(span, exc)
                self._end_span(span)
            raise
        self._tool_spans.setdefault(id(tool), []).append(_ToolSpanEntry(tool_name=name, span=span))

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Tool, result: str
    ) -> None:
        bucket = self._tool_spans.get(id(tool))
        entry = bucket.pop(0) if bucket else None
        if entry is None:
            return
        decision = self._ctx.circuit_breakers.record_success(entry.tool_name)
        if entry.span is not None:
            if decision is not None:
                self._set_breaker_attributes(entry.span, decision)
            self._telemetry.mark_ok(entry.span)
            self._end_span(entry.span)

    # -- Handoffs ----------------------------------------------------------

    async def on_handoff(
        self, context: RunContextWrapper[Any], from_agent: Agent[Any], to_agent: Agent[Any]
    ) -> None:
        # One agent_run span can legitimately cover multiple agents/models after a
        # handoff; per-call provider/model resolution in on_llm_start handles that.
        if not self._tracing:
            return
        span = self._telemetry.tracer.start_span(f"handoff {from_agent.name} -> {to_agent.name}")
        self._telemetry.mark_ok(span)
        span.end()

    # -- internals ---------------------------------------------------------

    def _set_breaker_attributes(self, span: Span, decision: Any) -> None:
        snapshot = decision.snapshot
        self._telemetry.set_attributes(
            span,
            tool_attributes(
                tool_name=snapshot.tool_name,
                calls_used=self._budget.tool_calls,
                breaker_state=snapshot.state.value,
                breaker_failure_count=snapshot.failure_count,
                breaker_blocked=False,
                breaker_remaining_open_seconds=snapshot.remaining_open_seconds,
            ),
        )

    def _end_span(self, span: Span | None) -> None:
        if span is not None:
            span.end()

    def reap_unfinished_spans(self) -> None:
        """Close any spans left open when a guard raised mid-run.

        A guardrail exception inside one hook can cancel sibling tool tasks before
        their ``on_tool_end`` fires; :func:`guarded_runner` calls this in a
        ``finally`` so those spans are still ended.
        """

        if self._llm_span is not None:
            self._end_span(self._llm_span.span)
            self._llm_span = None
        for bucket in self._tool_spans.values():
            for entry in bucket:
                self._end_span(entry.span)
        self._tool_spans.clear()


def guarded_runner(
    agent: Agent[Any],
    *,
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
    feedback_to_input: Callable[[Any, list[str]], Any] | None = None,
    output_from_result: Callable[[Any], object] | None = None,
    context: object | None = None,
    run_config: object | None = None,
    max_turns: int | None = None,
) -> AgentCallable:
    """Wrap an OpenAI Agents SDK ``Agent`` as a GuardLoop-compatible agent callable.

    :param agent: an ``agents.Agent`` to run via ``Runner.run``.
    :param reserved_output_tokens: output tokens to reserve in the pre-flight
        budget check when ``agent.model_settings.max_tokens`` is unset (the SDK's
        chat models often leave it ``None``), so the cost/token caps stay
        enforceable.
    :param feedback_to_input: how to inject verifier feedback on a retry —
        ``(input, feedback) -> new_input``. The default appends the feedback to a
        string input, or appends a ``{"role": "user", "content": ...}`` item to a
        list input; the original ``input`` is never mutated.
    :param output_from_result: how to derive the agent's return value from the
        SDK ``RunResult``. The default returns ``result.final_output``.
    :param context: passed straight to ``Runner.run(context=...)`` (the SDK's own
        run-scoped context object, unrelated to GuardLoop's ``RunContext``).
    :param run_config: an ``agents.RunConfig`` passed to ``Runner.run(run_config=...)``.
    :param max_turns: passed to ``Runner.run(max_turns=...)`` when given (the SDK's
        per-turn loop cap, orthogonal to GuardLoop's pre-flight budget — a runaway
        agent stops at whichever fires first).
    """

    if reserved_output_tokens <= 0:
        raise ValueError("reserved_output_tokens must be a positive integer")

    async def agent_fn(ctx: RunContext, agent_input: object) -> object:
        run_input: object = agent_input
        if ctx.retry_feedback:
            feedback = list(ctx.retry_feedback)
            if feedback_to_input is not None:
                run_input = feedback_to_input(agent_input, feedback)
            else:
                run_input = _default_feedback_to_input(agent_input, feedback)
        hooks = GuardLoopRunHooks(ctx, reserved_output_tokens=reserved_output_tokens)
        kwargs: dict[str, Any] = {"hooks": hooks}
        if context is not None:
            kwargs["context"] = context
        if run_config is not None:
            kwargs["run_config"] = run_config
        if max_turns is not None:
            kwargs["max_turns"] = max_turns
        try:
            result = await Runner.run(agent, cast(Any, run_input), **kwargs)
        except BaseException as exc:
            guard_error = _guardloop_error_in_chain(exc)
            if guard_error is not None and guard_error is not exc:
                raise guard_error from exc
            raise
        finally:
            hooks.reap_unfinished_spans()
        if output_from_result is not None:
            return output_from_result(result)
        return getattr(result, "final_output", result)

    return agent_fn


# -- helpers ---------------------------------------------------------------


def _guardloop_error_in_chain(exc: BaseException) -> GuardLoopError | None:
    """Find a GuardLoopError anywhere in ``exc``'s cause/context chain.

    The SDK wraps exceptions raised from the tool lifecycle hooks in
    ``agents.exceptions.UserError`` (with the original as ``__cause__``), so a
    guardrail exception raised from ``on_tool_start`` reaches the caller wrapped.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, GuardLoopError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _default_feedback_to_input(agent_input: object, feedback: list[str]) -> object:
    note = "Revise your previous answer. Issues to fix: " + "; ".join(feedback)
    if isinstance(agent_input, str):
        return agent_input + "\n\n" + note
    if isinstance(agent_input, list):
        items: list[Any] = list(agent_input)
        items.append({"role": "user", "content": note})
        return items
    return agent_input


def _resolve_model(agent: Agent[Any]) -> str | None:
    raw = agent.model
    if raw is None:
        default = _sdk_get_default_model()
        return _bare_model_name(default) if isinstance(default, str) and default.strip() else None
    if isinstance(raw, str):
        return _bare_model_name(raw) if raw.strip() else None
    name = getattr(raw, "model", None)
    if isinstance(name, str) and name.strip():
        return _bare_model_name(name)
    return None


def _bare_model_name(name: str) -> str:
    bare = name.strip()
    if "/" in bare:  # "litellm/anthropic/claude-..." -> "claude-...", "openai/gpt-4o" -> "gpt-4o"
        bare = bare.rsplit("/", 1)[-1].strip()
    return bare


def _resolve_provider(agent: Agent[Any], model_name: str) -> str:
    raw = agent.model
    raw_text = raw if isinstance(raw, str) else type(raw).__name__ if raw is not None else ""
    haystack = f"{model_name} {raw_text}".lower()
    if "anthropic" in haystack or "claude" in haystack:
        return "anthropic"
    return "openai"


def _resolve_reserved_output_tokens(agent: Agent[Any], default: int) -> int:
    settings = getattr(agent, "model_settings", None)
    value = getattr(settings, "max_tokens", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _extract_usage(response: ModelResponse) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    return (
        _non_negative_int(getattr(usage, "input_tokens", 0)),
        _non_negative_int(getattr(usage, "output_tokens", 0)),
    )


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def _tool_name(tool: Tool) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "tool"
