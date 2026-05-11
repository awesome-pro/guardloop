# GuardLoop Project Overview

This document explains what GuardLoop is, what has been implemented so far,
how the current system works, and what the next development goals are.

GuardLoop is a Python library for adding production-style runtime guardrails to
AI agents. Its main idea is simple: agent code can still own the reasoning loop,
but GuardLoop owns enforcement around risky operations such as LLM calls and
tool calls.

## Current Status

GuardLoop is currently published as version `0.4.1`.

- GitHub repository: `awesome-pro/guardloop`
- PyPI package: `guardloop`
- Import package: `guardloop`
- Main public class: `GuardLoop`
- Compatibility aliases: `AgentRuntime` and `AgentRuntimeError`
- Python support: `>=3.11`
- Package layout: `src/guardloop`
- Build backend: Hatchling
- Package manager: `uv`

The current portfolio story is:

1. GuardLoop prevents runaway agent cost before an expensive LLM call is sent.
2. GuardLoop prevents repeated calls to failing tools using circuit breakers.
3. GuardLoop re-runs an agent against verifiers until the output passes, bounded by the shared budget.
4. GuardLoop returns structured run results instead of leaving failures hidden.
5. GuardLoop emits OpenTelemetry spans for agent runs, LLM calls, tool calls, and verifier runs.
6. GuardLoop slots *under* an existing LangGraph graph (`guarded_graph`) or OpenAI Agents SDK
   `Agent` (`guarded_runner`) without rewriting it.
7. GuardLoop is typed, tested, packaged, released on GitHub, and published on PyPI.

The four pillars from the original design — resource limits, circuit breakers,
the self-healing verifier loop, and OpenTelemetry-native observability — are all
implemented as of v0.3. v0.4 adds framework adapters (LangGraph in v0.4.0, the
OpenAI Agents SDK in v0.4.1), each of which applies all four pillars inside a
third-party agent.

## Problem Being Solved

AI agents usually run as loops around LLM calls and external tools. When those
loops fail, they can:

- call expensive models repeatedly,
- burn through tokens and cost,
- retry broken tools again and again,
- run too long,
- fail without clear observability.

GuardLoop adds a runtime layer around the agent loop. The user still writes a
normal async Python agent function, but it receives a `RunContext` with protected
LLM clients and tool helpers.

## Implemented Features

### 1. Runtime Execution Wrapper

The main entry point is `GuardLoop`.

```python
from guardloop import GuardLoop, BudgetConfig, RunContext

runtime = GuardLoop(
    budget=BudgetConfig(
        cost_limit_usd="0.10",
        token_limit=10_000,
        time_limit_seconds=60,
        tool_call_limit=20,
    )
)


async def agent(ctx: RunContext, prompt: str) -> str:
    response = await ctx.openai.responses.create(
        model="gpt-5.2",
        input=prompt,
        max_output_tokens=300,
    )
    return str(response.output_text)


result = await runtime.run(agent, "research this topic")
```

`runtime.run()` always returns a `RunResult`. Controlled guardrail stops are
reported as `success=False` with a `terminated_reason`, error metadata, cost,
token counts, duration, and tool-call count.

### 2. Budget Guardrails

GuardLoop supports hard limits for:

- total estimated and actual cost,
- total tokens,
- wall-clock runtime,
- number of tool calls.

Before each LLM call, GuardLoop estimates input tokens, reserves the declared
maximum output tokens, checks pricing, and blocks calls that would exceed the
configured cap.

After each LLM response, GuardLoop reads provider usage metadata and records the
actual input tokens, output tokens, and cost.

Important behavior:

- Cost math uses `Decimal`, not float.
- Time measurement uses `time.monotonic()`.
- Missing output-token limits are blocked before sending provider requests.
- Unknown model pricing is blocked unless custom pricing is supplied.

### 3. OpenAI and Anthropic Wrappers

GuardLoop currently wraps direct provider SDK clients:

- `AsyncOpenAI.responses.create`
- `AsyncAnthropic.messages.create`

The wrappers preserve the normal SDK calling style while adding budget checks,
usage accounting, pricing, and tracing.

Example:

```python
response = await ctx.openai.responses.create(
    model="gpt-5.2",
    input=prompt,
    max_output_tokens=300,
)
```

For Anthropic:

```python
message = await ctx.anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}],
)
```

### 4. Per-Tool Circuit Breakers

v0.2 adds per-tool circuit breakers.

Circuit breaker state lives on the `GuardLoop` instance, so it persists across
multiple `runtime.run()` calls without becoming global process state.

Supported states:

- `closed`: tool calls are allowed.
- `open`: tool calls are rejected immediately.
- `half_open`: one or more trial calls are allowed after cooldown.

Default behavior:

- Circuit breakers are enabled by default.
- Default failure threshold is `3`.
- Default recovery timeout is `30` seconds.
- Default half-open success threshold is `1`.
- Per-tool overrides are supported.

Example:

```python
from guardloop import CircuitBreakerConfig, CircuitBreakerPolicy, GuardLoop

runtime = GuardLoop(
    circuit_breakers=CircuitBreakerConfig(
        default=CircuitBreakerPolicy(
            failure_threshold=3,
            recovery_timeout_seconds=30,
        ),
        tool_overrides={
            "web_search": CircuitBreakerPolicy(
                failure_threshold=2,
                recovery_timeout_seconds=10,
            )
        },
    )
)
```

Tool calls are protected through:

```python
await ctx.call_tool("tool_name", tool_function, *args, **kwargs)
```

or:

```python
protected_tool = ctx.wrap_tool("tool_name", tool_function)
await protected_tool(*args, **kwargs)
```

Open breaker calls are rejected before the tool-call budget is incremented and
before user tool code is invoked.

### 5. OpenTelemetry Tracing

GuardLoop emits spans for:

- agent runs,
- LLM calls,
- tool calls.

Spans include useful attributes such as provider, model, input tokens, output
tokens, estimated cost, actual cost, tool name, circuit breaker state, and
termination reason.

OpenTelemetry export is optional. The core library depends only on
`opentelemetry-api`; exporters are available through the `otel` extra.

```bash
pip install "guardloop[otel]"
```

### 6. Structured Exceptions

GuardLoop has a public exception hierarchy for controlled runtime stops:

- `GuardLoopError`
- `BudgetExceeded`
- `TokenLimitExceeded`
- `ToolCallLimitExceeded`
- `TimeLimitExceeded`
- `ModelPricingMissing`
- `TokenLimitMissing`
- `CircuitBreakerOpen`
- `VerificationFailed` (only in verifier strict mode)
- `VerifierExecutionError` (a verifier callable itself raised)

The runtime catches these controlled exceptions and converts them into
structured `RunResult` objects.

### 7. Verifier Retry Loop

v0.3 adds the self-healing pillar: after an agent returns, GuardLoop runs a
chain of verifiers against the output and, on rejection, feeds feedback back
and retries the agent.

A verifier is any callable — sync or async — with the signature
`(output, VerifierContext) -> VerifierResult | bool | None`:

```python
from guardloop import GuardLoop, RunContext, VerifierConfig, VerifierContext, VerifierResult


def no_todo(output: object, ctx: VerifierContext) -> VerifierResult:
    if "TODO" in str(output):
        return VerifierResult(passed=False, feedback="Replace the TODO placeholder.")
    return VerifierResult(passed=True)


runtime = GuardLoop(verifiers=[no_todo], verifier_config=VerifierConfig(max_retries=2))


async def agent(ctx: RunContext, task: str) -> str:
    # On a retry, ctx.retry_feedback holds the verifier's complaints, oldest first.
    ...
```

Built-in rule-based verifier factories ship in `guardloop`:

- `non_empty(*, allow_whitespace=False)`
- `matches_regex(pattern, *, flags=0)`
- `is_json_object(*, required_keys=())`

Behavior:

- Verifiers are configured per `GuardLoop` instance via `verifiers=[...]` or
  `runtime.add_verifier(fn)`; there is no persistent verifier state across runs.
- `VerifierChain` runs verifiers in order, fail-fast: the first failing verdict
  wins. Anything that isn't a `VerifierResult` is normalized (`True`/`None` →
  passed, `False` → failed with generated feedback).
- `VerifierConfig` controls the loop: `max_retries` (extra agent invocations
  after the first; `0` means no retry), `raise_on_failure` (strict mode),
  `pass_feedback_to_agent`, and `enabled`.
- The retry loop reuses the same `RunContext` and `BudgetController` across
  attempts — cost, tokens, time, and tool calls accumulate, so a verifier loop
  cannot bypass any cap, and the run's single `asyncio.timeout()` bounds the
  whole loop.
- `RunResult` reports `verification_passed: bool | None` (`None` if no verifiers
  ran), `verification_attempts: int`, and `verification_feedback: list[str]`.
- When retries are exhausted: by default `success=False`,
  `terminated_reason="verification_failed"`, `output` still set to the last
  attempt. With `raise_on_failure=True`, the runtime surfaces a
  `VerificationFailed` instead (`output=None`, attempt count and feedback in
  `metadata`). A verifier that itself raises becomes a `VerifierExecutionError`
  (`terminated_reason="verifier_error"`) and is not retried.

No-key demo:

```bash
uv run python examples/verifier_retry_loop.py
```

### 8. Framework Adapters

v0.4 adds framework adapters: thin modules that produce a GuardLoop-compatible
`async def agent(ctx, ...)` callable wrapping a third-party agent, so all four
pillars apply *inside* it without rewriting it. Each adapter lives behind its own
optional extra and is intentionally not re-exported from the top-level `guardloop`
package, so `import guardloop` stays dependency-light.

**LangGraph (`guardloop.adapters.langgraph.guarded_graph`, v0.4.0).** LangGraph
nodes call LangChain chat models, which do not flow through GuardLoop's
`ctx.openai` / `ctx.anthropic` wrappers, so the adapter binds a LangChain callback
handler to the `RunContext` — pre-flight budget check before each LLM call, usage
recorded afterward, tool calls routed through the per-tool circuit breaker and the
tool-call budget. Cost / token / time caps, breakers, and `llm_call` / `tool_call`
OpenTelemetry spans all apply inside a LangGraph run; the verifier retry loop wraps
the whole graph run.

```python
from langchain_core.messages import HumanMessage

from guardloop import GuardLoop, BudgetConfig
from guardloop.adapters.langgraph import guarded_graph

runtime = GuardLoop(budget=BudgetConfig(cost_limit_usd="0.10", token_limit=10_000))
agent = guarded_graph(my_compiled_graph, input_key="messages")
result = await runtime.run(agent, {"messages": [HumanMessage("...")]})
```

`guarded_graph(compiled_graph, *, input_key="messages", reserved_output_tokens=1024,
feedback_to_state=None, output_from_state=None, config=None)`. Notes:
`reserved_output_tokens` is the pre-flight output reservation (LangChain chat models
often omit `max_tokens`); on a verifier retry the feedback is injected into a copy of
the input state (`feedback_to_state` to customise; never mutates the original);
tool-side enforcement is as hard as the graph's own error handling (a `ToolNode` with
the default `handle_tool_errors=True` turns a budget/breaker exception into a
`ToolMessage` — the breaker still records it, LLM-side caps still terminate; use
`handle_tool_errors=False` for hard tool-call enforcement); streaming (`astream` /
`astream_events`) is out of scope. Behind the `langgraph` extra
(`pip install "guardloop[langgraph]"`); exports `guarded_graph` and
`GuardLoopCallbackHandler`. No-key demo: `uv run python examples/langgraph_guarded.py`.

**OpenAI Agents SDK (`guardloop.adapters.openai_agents.guarded_runner`, v0.4.1).**
The SDK's `Agent` runs via `Runner.run`, whose model calls bypass GuardLoop's
provider wrappers too, so the adapter binds a `GuardLoopRunHooks` (a subclass of the
SDK's `RunHooks`) to the `RunContext` — pre-flight budget check on `on_llm_start`,
actual usage recorded on `on_llm_end` (from `response.usage`), tool calls routed
through `before_call` / `record_tool_call_started` / `record_success` on
`on_tool_start` / `on_tool_end`. Same caps, breakers, and `llm_call` / `tool_call`
spans, now inside `Runner.run(...)`; the verifier retry loop wraps the whole run.

```python
from agents import Agent

from guardloop import GuardLoop, BudgetConfig
from guardloop.adapters.openai_agents import guarded_runner

runtime = GuardLoop(budget=BudgetConfig(cost_limit_usd="0.10", token_limit=10_000))
agent = guarded_runner(Agent(name="researcher", model="gpt-5.2", instructions="..."))
result = await runtime.run(agent, "research agent runtime safety")
```

`guarded_runner(agent, *, reserved_output_tokens=1024, feedback_to_input=None,
output_from_result=None, context=None, run_config=None, max_turns=None)`. Notes:
`RunHooks` methods are natively `async` (unlike LangChain's async callbacks, which run
in a detached loop that swallows exceptions — that's why the LangGraph handler had to
be synchronous), so guardrail exceptions propagate; the SDK wraps exceptions raised
from its tool lifecycle hooks in `agents.exceptions.UserError`, so `guarded_runner`
unwraps a `GuardLoopError` from the cause chain before re-raising; `reserved_output_tokens`
is the pre-flight output reservation when `agent.model_settings.max_tokens` is unset;
on a verifier retry the feedback is injected into a copy of the run input
(`feedback_to_input` to customise; `output_from_result` for the answer; default is
`result.final_output`). **Known gap:** the SDK has no `on_tool_error` hook and, by
default, turns a tool exception into an error *string* fed back to the model, so the
breaker records tool *attempts* and *successes* but not *failures* — a flaky
SDK-managed tool won't open the breaker on its own (an already-open breaker still
rejects the next call; route a tool body through `ctx.call_tool(...)` for full breaker
semantics). `max_turns` (the SDK's per-turn loop cap) and GuardLoop's pre-flight
budget compose; the SDK's own tracing is independent of GuardLoop's OpenTelemetry;
streaming (`Runner.run_streamed`) is out of scope for v0.4.1. Behind the
`openai-agents` extra (`pip install "guardloop[openai-agents]"`); exports
`guarded_runner` and `GuardLoopRunHooks`. No-key demo:
`uv run python examples/openai_agents_guarded.py`.

### 9. Demos

No-key demos:

```bash
uv run python examples/runaway_cost_prevention.py
uv run python examples/tool_circuit_breaker.py
uv run python examples/verifier_retry_loop.py
uv run python examples/langgraph_guarded.py
uv run python examples/openai_agents_guarded.py
```

The runaway-cost demo proves that GuardLoop stops an agent before the next LLM
request would exceed the configured budget.

The circuit-breaker demo proves that GuardLoop lets a flaky tool fail up to the
configured threshold, then rejects the next attempt without invoking the tool.

The verifier-retry demo proves that GuardLoop rejects a bad answer, hands the
verifier's feedback back through `ctx.retry_feedback`, and accepts the corrected
answer on a later attempt.

The LangGraph and OpenAI Agents SDK demos prove that the budget / telemetry envelope
applies inside a real graph / `Runner.run`: the first run succeeds with cost and
tokens recorded, and the second run is stopped before the model call by a tiny token
budget.

Optional live provider demos:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

uv run python examples/live_openai_basic.py
uv run python examples/live_anthropic_basic.py
```

## Architecture

```mermaid
flowchart LR
    LG["LangGraph graph"] -. "guarded_graph(...)" .-> A
    OA["OpenAI Agents SDK agent"] -. "guarded_runner(...)" .-> A
    A["User agent function"] --> R["GuardLoop.run()"]
    R --> C["RunContext"]
    R --> B["BudgetController"]
    R --> CB["CircuitBreakerRegistry"]
    R --> V["VerifierChain"]
    R --> T["Telemetry"]
    C --> O["Wrapped OpenAI client"]
    C --> AN["Wrapped Anthropic client"]
    C --> TO["ToolRunner"]
    C --> H["GuardLoopCallbackHandler (LangChain)"]
    C --> RH["GuardLoopRunHooks (OpenAI Agents SDK)"]
    O --> B
    AN --> B
    TO --> B
    TO --> CB
    H --> B
    H --> CB
    RH --> B
    RH --> CB
    O --> T
    AN --> T
    TO --> T
    H --> T
    RH --> T
    V --> T
    V -. "feedback on retry" .-> C
```

Important modules:

- `src/guardloop/runtime.py`: main `GuardLoop` execution wrapper and retry loop.
- `src/guardloop/context.py`: `RunContext` passed into user agents (incl. `circuit_breakers`).
- `src/guardloop/budget.py`: cost, token, time, and tool-call enforcement.
- `src/guardloop/circuit_breaker.py`: per-tool circuit breaker state machines.
- `src/guardloop/verifier.py`: verifier types, the `VerifierChain` runner, and built-in verifiers.
- `src/guardloop/providers/openai.py`: OpenAI Responses wrapper.
- `src/guardloop/providers/anthropic.py`: Anthropic Messages wrapper.
- `src/guardloop/tools.py`: protected sync/async tool execution.
- `src/guardloop/adapters/langgraph.py`: LangGraph adapter — `guarded_graph` and `GuardLoopCallbackHandler` (behind the `langgraph` extra).
- `src/guardloop/adapters/openai_agents.py`: OpenAI Agents SDK adapter — `guarded_runner` and `GuardLoopRunHooks` (behind the `openai-agents` extra).
- `src/guardloop/telemetry/`: OpenTelemetry setup and attribute conventions.
- `src/guardloop/models.py`: Pydantic config/result models.
- `src/guardloop/pricing.py`: model pricing catalog and custom pricing support.
- `src/guardloop/exceptions.py`: public controlled exception hierarchy.

## Public API

Current primary exports:

- `GuardLoop`
- `BudgetConfig`
- `TelemetryConfig`
- `RunContext`
- `RunResult`
- `ModelPricing`
- `CircuitBreakerConfig`
- `CircuitBreakerPolicy`
- `CircuitBreakerState`
- `CircuitBreakerSnapshot`
- `Verifier`
- `VerifierResult`
- `VerifierContext`
- `VerifierConfig`
- `VerifierChain`
- `non_empty`, `matches_regex`, `is_json_object` (built-in verifier factories)
- `GuardLoopError`
- `BudgetExceeded`
- `TokenLimitExceeded`
- `ToolCallLimitExceeded`
- `TimeLimitExceeded`
- `ModelPricingMissing`
- `TokenLimitMissing`
- `CircuitBreakerOpen`
- `VerificationFailed`
- `VerifierExecutionError`

Compatibility aliases:

- `AgentRuntime = GuardLoop`
- `AgentRuntimeError = GuardLoopError`

Framework adapters are *not* part of the top-level `guardloop` namespace (so
`import guardloop` pulls only the base dependencies). Import them from their
submodule, behind the matching extra:

```python
# pip install "guardloop[langgraph]"
from guardloop.adapters.langgraph import guarded_graph, GuardLoopCallbackHandler

# pip install "guardloop[openai-agents]"
from guardloop.adapters.openai_agents import guarded_runner, GuardLoopRunHooks
```

## Testing and Quality

Current test coverage includes:

- cost cap pre-flight blocking,
- token cap blocking,
- tool-call limit blocking,
- Decimal cost accounting,
- OpenAI usage accounting,
- Anthropic usage accounting,
- missing output token limit errors,
- missing model pricing errors,
- successful runtime results,
- runaway fake agent termination,
- timeout handling,
- tool exception handling,
- circuit breaker open/closed/half-open transitions,
- per-tool circuit breaker overrides,
- circuit breaker reset helpers,
- verifier passes / fails-then-passes / exhausts retries,
- `max_retries=0`, fail-fast chains, sync and async verifiers,
- bool-shorthand verdicts and generated feedback,
- verifier exceptions surface as `verifier_error`,
- budget shared across retry attempts (cannot be bypassed),
- run timeout bounds the whole retry loop,
- `ctx.retry_feedback` visibility and `verification_feedback` recording,
- strict mode, disabled verifiers, built-in verifiers,
- OpenTelemetry span attributes (LLM, tool, and `verifier_run` spans),
- LangGraph adapter: happy path (usage + output recorded), sync- and async-node
  graphs, budget cap tripping inside the graph, `reserved_output_tokens` pre-flight,
  tool-call limit inside a `ToolNode` loop, circuit breaker opening across runs,
  verifier loop wrapping the graph with feedback injection, caller state not
  mutated, custom `feedback_to_state` / `output_from_state` hooks, `llm_call` /
  `tool_call` spans parented under `agent_run`, model errors propagating with no
  recorded usage, and an unresolvable model name producing a clear error,
- OpenAI Agents SDK adapter: happy path (usage + output recorded), budget cap
  tripping inside `Runner.run`, `reserved_output_tokens` pre-flight, tool-call
  limit across turns, an open breaker blocking an SDK tool call (and an
  SDK-managed tool failure *not* opening the breaker — the documented gap),
  verifier loop wrapping `Runner.run` with feedback injection, caller input not
  mutated, custom `feedback_to_input` / `output_from_result` hooks, `llm_call` /
  `tool_call` spans parented under `agent_run`, model errors propagating with no
  recorded usage, an unpriced model name producing a `ModelPricingMissing`, an
  unresolvable model object producing a clear `RuntimeError`, and `reserved_output_tokens<=0`
  rejected.

CI (`.github/workflows/ci.yml`) runs the lint / type-check / test gates on every
push to `main` and every pull request, across Python 3.11, 3.12, and 3.13.

Quality gates:

```bash
uv sync --all-extras --group dev
uv run pytest
uv run pytest --cov=guardloop
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
uvx twine check dist/guardloop-0.4.1.tar.gz dist/guardloop-0.4.1-py3-none-any.whl
unzip -l dist/guardloop-0.4.1-py3-none-any.whl | grep adapters   # adapters subpackage shipped in the wheel
```

## Packaging and Release State

GuardLoop is configured and published as a real Python package.

- PyPI project: `guardloop`
- Latest release: `v0.4.1`
- Optional extras: `otel` (OpenTelemetry exporters), `langgraph` (LangGraph adapter), `openai-agents` (OpenAI Agents SDK adapter)
- Build artifacts: wheel and source distribution (both include `guardloop/adapters/`)
- Trusted Publishing: GitHub Actions to PyPI
- GitHub environment: `pypi`
- Changelog: `CHANGELOG.md`

Workflows:

- `.github/workflows/ci.yml` — lint / type-check / test on push and pull request (Python 3.11–3.13).
- `.github/workflows/publish-pypi.yml` — builds distributions with `uv build`,
  uploads artifacts, and publishes to PyPI using `pypa/gh-action-pypi-publish`
  with OIDC Trusted Publishing instead of a long-lived API token.

## Current Limitations

GuardLoop is intentionally focused. It does not yet include:

- LLM-based verifier ergonomics (budget-tracked clients on `VerifierContext`),
- streaming support in the framework adapters (`astream` / `astream_events`, `Runner.run_streamed`),
- circuit-breaker *failure* tracking from SDK-managed tools in the OpenAI Agents
  adapter (the SDK has no tool-error lifecycle hook; attempts and successes are
  tracked, and an already-open breaker still blocks calls),
- persistent circuit breaker state in Redis/database,
- provider-level circuit breakers,
- loop detection (repeated `tool + args`),
- UI dashboard,
- an OpenTelemetry metrics layer and a packaged trace-viewer stack (Jaeger / Phoenix),
- automated docs site,
- semantic versioning automation.

These are good future layers, but v0.4 already implements all four pillars plus
two framework adapters (LangGraph, OpenAI Agents SDK), and is a complete
portfolio-grade library foundation.

## Future Goals

### v0.3: Verifier Retry Loop — delivered

Shipped: deterministic and async verifier callables, a fail-fast `VerifierChain`,
built-in rule-based verifiers, a bounded retry loop that reuses the same budget
and run timeout, `ctx.retry_feedback`, structured `RunResult.verification_*`
fields, an opt-in strict mode, and `verifier_run` OpenTelemetry spans. The one
ergonomic gap deferred to later: budget-tracked LLM clients on `VerifierContext`
for LLM-based verifiers (today an LLM verifier must close over its own client).

### v0.4: Framework Adapters

Goal: integrate GuardLoop with common agent frameworks without changing the core
runtime model or rewriting the user's agent.

- **v0.4.0 — LangGraph (delivered).** `guardloop.adapters.langgraph.guarded_graph`
  returns a GuardLoop-compatible agent; a synchronous LangChain callback handler
  bound to the `RunContext` runs the pre-flight budget check, records usage, and
  routes tool calls through the breaker and tool-call budget — so all four pillars
  apply inside a LangGraph run, and the verifier loop wraps the whole graph run.
  Behind the `langgraph` extra; no-key demo at `examples/langgraph_guarded.py`.
  Also added `RunContext.circuit_breakers` and a CI workflow.
- **v0.4.1 — OpenAI Agents SDK (delivered).** `guardloop.adapters.openai_agents.guarded_runner`
  returns a GuardLoop-compatible agent that calls `Runner.run` under the hood; a
  `GuardLoopRunHooks` (a `RunHooks` subclass) bound to the `RunContext` does the
  pre-flight budget check (`on_llm_start`), usage accounting (`on_llm_end`), and
  breaker + tool-call-budget routing (`on_tool_start` / `on_tool_end`). The
  verifier loop wraps the whole run; feedback is injected into a copy of the run
  input. Behind the `openai-agents` extra; no-key demo at
  `examples/openai_agents_guarded.py`. Known gap: the SDK has no tool-error hook,
  so the breaker tracks tool attempts/successes but not failures from SDK-managed
  tools; streaming (`Runner.run_streamed`) is out of scope.

Portfolio value:

- shows practical ecosystem integration,
- makes GuardLoop easier to demonstrate with real agent workflows,
- proves the core design is framework-agnostic.

### v0.5: Observability Depth

Goal: extend the OpenTelemetry layer past spans and make the whole trace tree
inspectable end to end.

Planned capabilities:

- an OpenTelemetry metrics provider (counters/histograms for cost, tokens, tool
  calls, and verifier attempts, alongside the existing spans),
- per-attempt `agent_attempt` span nesting so a verifier retry loop reads as a
  tree rather than flat sibling LLM calls,
- a one-command `docker-compose` stack (Jaeger + Arize Phoenix) the no-key demos
  export to,
- an architecture write-up plus a recorded walkthrough of the
  `agent_run -> llm_call -> tool_call -> verifier_run` tree as the proof artifact.

Portfolio value:

- demonstrates production observability beyond "we emit traces" — metrics, span
  hierarchy, and a working backend,
- produces one concrete, inspectable artifact (the trace tree) a reviewer can look at.

### v0.6: Persistence and Team Settings

Goal: support longer-lived runtime state and reusable policy configuration.

Possible capabilities:

- pluggable state backends for circuit breakers (Redis/SQL),
- YAML/TOML policy loading and organization/team default policies,
- loop detection (repeated `tool + args` within a run),
- multi-model pricing (Gemini, Groq, Mistral) and a model pricing update workflow,
- LiteLLM integration for one wrapping path across providers.

Portfolio value:

- moves the project closer to production deployment,
- shows system design beyond in-memory library code.

### v1.0: Stable Guardrail Runtime

Goal: define a stable public API and production readiness baseline.

Potential requirements:

- documented API stability policy,
- stronger compatibility tests,
- more provider models and pricing coverage,
- richer examples,
- `CHANGELOG.md` as the source of truth (started at v0.3),
- auto-published docs site,
- release checklist,
- streaming-response accounting and Anthropic token-counting API.

## Interview Talking Points

Use this short explanation:

GuardLoop is a production runtime layer for AI agents. Instead of building
another agent framework, it wraps the risky parts of an agent loop: model calls
and tool calls. It enforces cost, token, time, and tool-call limits; stops
runaway agent loops before sending expensive requests; opens circuit breakers
around flaky tools; re-runs the agent against verifiers — feeding their feedback
back in — until the output passes, all under the same shared budget; and emits
OpenTelemetry traces so failures are observable. It is packaged as a typed
Python library, published on PyPI, tested with fake clients, and — because the
core stayed framework-agnostic — it now slots *under* both LangGraph and the
OpenAI Agents SDK via one-file adapters, with no change to the runtime.

Good questions to be ready for:

- Why use pre-flight token and cost checks before an LLM call?
- Why use `Decimal` for cost?
- Why does circuit breaker state live on `GuardLoop` instead of globally?
- Why direct provider wrappers before framework adapters?
- Why are the adapters converter functions (`guarded_graph` / `guarded_runner`)
  instead of subclasses or framework plugins?
- Why is the LangGraph callback handler synchronous but the OpenAI Agents SDK
  `RunHooks` subclass `async`? (LangChain swallows exceptions from async
  callbacks; the SDK awaits hooks inline.)
- How would you add Redis-backed circuit breaker state?
- How would verifier retries interact with budget limits?
- How would you keep model pricing accurate over time?
- What should happen when provider usage metadata is missing?

## Recommended Next Step

All four pillars plus two framework adapters (LangGraph in v0.4.0, the OpenAI
Agents SDK in v0.4.1) are implemented. The best next engineering milestone is
**v0.5: observability depth** — extending the OpenTelemetry layer past spans.

Build it narrowly:

- add an OpenTelemetry metrics provider (`Counter` / `Histogram` for cost,
  tokens, tool calls, and verifier attempts) alongside the existing spans,
- land the deferred per-attempt `agent_attempt {n}` span nesting so a verifier
  retry loop reads as a tree, not a flat list of sibling LLM calls,
- add a `docker-compose.yml` running Jaeger + Arize Phoenix plus a script that
  runs the no-key demos (including both adapter demos) against it,
- write the architecture write-up and capture a walkthrough of the
  `agent_run -> llm_call -> tool_call -> verifier_run` tree as the proof artifact,
- bump `Development Status` from `3 - Alpha` to `4 - Beta`.

After v0.5: v0.6 (persistence + team policies + loop detection + multi-model
pricing + LiteLLM, with the deferred `failure_error_function`-wrapping option for
the OpenAI Agents adapter's breaker as a candidate) → v1.0 (stable API, docs
site, release checklist, hardened token story). The "wrapper, not framework"
thesis is already proven — the same budget / circuit breaker / verifier /
telemetry envelope works around two different agent frameworks with no change to
the core runtime — so the remaining work is about depth, not new shape.
