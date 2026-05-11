# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# LangChain callback payloads (``metadata`` / ``invocation_params`` / ``LLMResult``
# internals) and arbitrary graph state are loosely typed, so strict "unknown type"
# checks are relaxed for this boundary module; the public surface is fully typed.
"""Run a LangGraph graph under a GuardLoop ``RunContext``.

``guarded_graph(compiled_graph)`` returns a GuardLoop-compatible agent callable
you pass to :meth:`guardloop.GuardLoop.run`. LangGraph nodes call LangChain chat
models, which do not flow through GuardLoop's ``ctx.openai`` / ``ctx.anthropic``
wrappers; instead this adapter binds a LangChain callback handler to the
``RunContext`` so budget caps, per-tool circuit breakers, and OpenTelemetry spans
apply *inside* the graph. The verifier retry loop wraps the whole graph run: on a
rejected output the runtime re-invokes the agent, which re-runs ``graph.ainvoke``
with the verifier feedback injected into a copy of the input state.

Requires the ``langgraph`` extra: ``pip install "guardloop[langgraph]"``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from opentelemetry.trace import Span

from guardloop.context import RunContext
from guardloop.runtime import AgentCallable
from guardloop.telemetry.conventions import (
    llm_request_attributes,
    llm_response_attributes,
    tool_attributes,
)
from guardloop.telemetry.tracer import Telemetry
from guardloop.tokenization import estimate_anthropic_tokens, estimate_openai_tokens

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.outputs import LLMResult
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'guardloop\'s LangGraph adapter requires the "langgraph" extra: '
        'pip install "guardloop[langgraph]"'
    ) from exc

__all__ = ["GuardLoopCallbackHandler", "guarded_graph"]

DEFAULT_RESERVED_OUTPUT_TOKENS = 1024

_PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "azure_openai": "openai",
    "azure-openai": "openai",
    "azureopenai": "openai",
    "azure": "openai",
    "anthropic": "anthropic",
    "bedrock_anthropic": "anthropic",
    "anthropic_bedrock": "anthropic",
}


@dataclass
class _LLMSpanEntry:
    provider: str
    model: str
    estimated_input_tokens: int
    span: Span | None = None


@dataclass
class _ToolSpanEntry:
    tool_name: str
    span: Span | None = None


class GuardLoopCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that enforces a GuardLoop ``RunContext``.

    Bound to a single :class:`~guardloop.RunContext`, it runs a pre-flight budget
    check before each LLM call (``on_chat_model_start`` / ``on_llm_start``),
    records actual usage afterward (``on_llm_end``), and routes tool invocations
    through the per-tool circuit breaker and the tool-call budget (``on_tool_start``
    / ``on_tool_end`` / ``on_tool_error``). Each LLM and tool call gets an
    OpenTelemetry span that is a child of the active ``agent_run`` span.

    It is a synchronous handler with ``raise_error = True`` so guardrail
    exceptions raised inside callbacks propagate out of ``graph.ainvoke()`` for
    both synchronous and asynchronous graph nodes (LangChain only honours
    ``raise_error`` for synchronous handlers). ``run_inline = True`` keeps it on
    the active event loop / context when nodes run asynchronously.
    """

    raise_error: bool = True
    run_inline: bool = True

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
        self._llm_spans: dict[UUID, _LLMSpanEntry] = {}
        self._tool_spans: dict[UUID, _ToolSpanEntry] = {}

    # -- LLM ---------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        payload = [getattr(message, "content", "") for batch in messages for message in batch]
        self._begin_llm(
            serialized=serialized, metadata=metadata, kwargs=kwargs, payload=payload, run_id=run_id
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._begin_llm(
            serialized=serialized, metadata=metadata, kwargs=kwargs, payload=prompts, run_id=run_id
        )

    def _begin_llm(
        self,
        *,
        serialized: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
        payload: object,
        run_id: UUID,
    ) -> None:
        raw_params = kwargs.get("invocation_params")
        params: Mapping[str, Any] = raw_params if isinstance(raw_params, Mapping) else {}
        meta: Mapping[str, Any] = metadata if isinstance(metadata, Mapping) else {}
        provider = _resolve_provider(meta, params, serialized)
        model = _resolve_model(meta, params, serialized)
        if model is None:
            raise RuntimeError(
                "GuardLoop's LangGraph adapter could not determine the LLM model name from the "
                "LangChain callback metadata. Set a model name on the chat model, or register "
                "pricing for it via GuardLoop(pricing=[...])."
            )
        reserved = _resolve_reserved_output_tokens(meta, params, self._reserved_output_tokens)
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
        self._llm_spans[run_id] = _LLMSpanEntry(
            provider=provider, model=model, estimated_input_tokens=estimated_input, span=span
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._llm_spans.pop(run_id, None)
        if entry is None:
            return
        try:
            input_tokens, output_tokens = _extract_usage(response, entry.estimated_input_tokens)
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
            if entry.span is not None:
                entry.span.end()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._llm_spans.pop(run_id, None)
        if entry is None or entry.span is None:
            return
        self._telemetry.record_exception(entry.span, error)
        entry.span.end()

    # -- Tools -------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = _tool_name(serialized, kwargs)
        span: Span | None = None
        if self._tracing:
            span = self._telemetry.tracer.start_span(
                f"tool_call {name}",
                attributes=tool_attributes(tool_name=name, calls_used=self._budget.tool_calls),
            )
        try:
            decision = self._ctx.circuit_breakers.before_call(name)
            if span is not None and decision is not None:
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
            self._budget.record_tool_call_started(name)
            if span is not None:
                self._telemetry.set_attributes(
                    span, tool_attributes(tool_name=name, calls_used=self._budget.tool_calls)
                )
        except BaseException as exc:
            if span is not None:
                self._telemetry.record_exception(span, exc)
                span.end()
            raise
        self._tool_spans[run_id] = _ToolSpanEntry(tool_name=name, span=span)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._tool_spans.pop(run_id, None)
        if entry is None:
            return
        decision = self._ctx.circuit_breakers.record_success(entry.tool_name)
        if entry.span is not None:
            if decision is not None:
                snapshot = decision.snapshot
                self._telemetry.set_attributes(
                    entry.span,
                    tool_attributes(
                        tool_name=snapshot.tool_name,
                        calls_used=self._budget.tool_calls,
                        breaker_state=snapshot.state.value,
                        breaker_failure_count=snapshot.failure_count,
                        breaker_blocked=False,
                        breaker_remaining_open_seconds=snapshot.remaining_open_seconds,
                    ),
                )
            self._telemetry.mark_ok(entry.span)
            entry.span.end()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._tool_spans.pop(run_id, None)
        if entry is None:
            return
        if not isinstance(error, asyncio.CancelledError | TimeoutError):
            self._ctx.circuit_breakers.record_failure(entry.tool_name)
        if entry.span is not None:
            self._telemetry.record_exception(entry.span, error)
            entry.span.end()


def guarded_graph(
    compiled_graph: Any,
    *,
    input_key: str = "messages",
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
    feedback_to_state: Callable[[Any, list[str]], Any] | None = None,
    output_from_state: Callable[[Any], object] | None = None,
    config: Mapping[str, Any] | None = None,
) -> AgentCallable:
    """Wrap a compiled LangGraph graph as a GuardLoop-compatible agent callable.

    :param compiled_graph: a compiled LangGraph graph (anything with an async
        ``ainvoke(state, config=...)``).
    :param input_key: the state-dict key holding the message list (LangGraph's
        ``"messages"`` convention). Used by the default feedback-injection and
        output-extraction logic.
    :param reserved_output_tokens: output tokens to reserve in the pre-flight
        budget check when the chat model does not declare a ``max_tokens``.
        LangChain models frequently omit it, so this keeps the cost/token caps
        enforceable.
    :param feedback_to_state: how to inject verifier feedback on a retry —
        ``(state, feedback) -> new_state``. The default appends a
        :class:`~langchain_core.messages.HumanMessage` to ``state[input_key]``
        in a shallow copy of the state. The original ``state`` is never mutated.
    :param output_from_state: how to derive the agent's return value from the
        graph's final state. The default returns the last message's ``content``
        (so ``RunResult.output`` is a string and verifiers see a string), falling
        back to ``str(final_state)`` for non-standard state shapes.
    :param config: an extra ``RunnableConfig`` merged into every ``ainvoke``
        call. GuardLoop's callback handler is appended to ``config["callbacks"]``
        (which must be a list if present).
    """

    if reserved_output_tokens <= 0:
        raise ValueError("reserved_output_tokens must be a positive integer")

    async def agent(ctx: RunContext, state: Any) -> object:
        run_state: Any = state
        if ctx.retry_feedback:
            feedback = list(ctx.retry_feedback)
            if feedback_to_state is not None:
                run_state = feedback_to_state(state, feedback)
            else:
                run_state = _default_feedback_to_state(state, feedback, input_key)
        handler = GuardLoopCallbackHandler(ctx, reserved_output_tokens=reserved_output_tokens)
        run_config: dict[str, Any] = dict(config or {})
        run_config["callbacks"] = [*_as_callback_list(run_config.get("callbacks")), handler]
        final_state = await compiled_graph.ainvoke(run_state, config=run_config)
        if output_from_state is not None:
            return output_from_state(final_state)
        return _default_output_from_state(final_state, input_key)

    return agent


# -- helpers ---------------------------------------------------------------


def _as_callback_list(existing: object) -> list[Any]:
    if existing is None:
        return []
    if isinstance(existing, list):
        return list(existing)
    return [existing]


def _default_feedback_to_state(state: Any, feedback: list[str], input_key: str) -> Any:
    message = HumanMessage(
        content="Revise your previous answer. Issues to fix: " + "; ".join(feedback)
    )
    if isinstance(state, Mapping):
        messages = state.get(input_key)
        if isinstance(messages, list):
            new_state = dict(state)
            new_state[input_key] = [*messages, message]
            return new_state
    return state


def _default_output_from_state(final_state: Any, input_key: str) -> object:
    if isinstance(final_state, Mapping):
        messages = final_state.get(input_key)
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content is not None:
                return content
            return last
    return final_state


def _resolve_provider(
    metadata: Mapping[str, Any], params: Mapping[str, Any], serialized: Mapping[str, Any] | None
) -> str:
    raw = metadata.get("ls_provider") or params.get("ls_provider")
    if not raw and isinstance(serialized, Mapping):
        identifier = serialized.get("id")
        if isinstance(identifier, list):
            lowered = [str(part).lower() for part in identifier]
            if "anthropic" in lowered:
                raw = "anthropic"
            elif "openai" in lowered:
                raw = "openai"
    if not raw:
        return "openai"
    key = str(raw).strip().lower()
    return _PROVIDER_ALIASES.get(key, key)


def _resolve_model(
    metadata: Mapping[str, Any], params: Mapping[str, Any], serialized: Mapping[str, Any] | None
) -> str | None:
    for source in (metadata, params):
        for key in ("ls_model_name", "model", "model_name"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(serialized, Mapping):
        nested = serialized.get("kwargs")
        if isinstance(nested, Mapping):
            for key in ("model", "model_name"):
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _resolve_reserved_output_tokens(
    metadata: Mapping[str, Any], params: Mapping[str, Any], default: int
) -> int:
    for source in (params, metadata):
        for key in ("max_tokens", "max_output_tokens", "ls_max_tokens"):
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return default


def _extract_usage(response: LLMResult, fallback_input_tokens: int) -> tuple[int, int]:
    generations = getattr(response, "generations", None)
    if isinstance(generations, list):
        for batch in generations:
            if not isinstance(batch, list):
                continue
            for generation in batch:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                tokens = _usage_pair(usage, "input_tokens", "output_tokens")
                if tokens is not None:
                    return tokens
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, Mapping):
        token_usage = llm_output.get("token_usage")
        if not isinstance(token_usage, Mapping):
            token_usage = llm_output.get("usage")
        if isinstance(token_usage, Mapping):
            tokens = _usage_pair(token_usage, "prompt_tokens", "completion_tokens")
            if tokens is not None:
                return tokens
            tokens = _usage_pair(token_usage, "input_tokens", "output_tokens")
            if tokens is not None:
                return tokens
    return fallback_input_tokens, 0


def _usage_pair(usage: object, input_key: str, output_key: str) -> tuple[int, int] | None:
    if not isinstance(usage, Mapping):
        return None
    input_tokens = usage.get(input_key)
    output_tokens = usage.get(output_key)
    if (
        isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
    ):
        return input_tokens, output_tokens
    return None


def _tool_name(serialized: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> str:
    if isinstance(serialized, Mapping):
        name = serialized.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = kwargs.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "tool"
