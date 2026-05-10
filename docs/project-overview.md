# GuardLoop Project Overview

This document explains what GuardLoop is, what has been implemented so far,
how the current system works, and what the next development goals are.

GuardLoop is a Python library for adding production-style runtime guardrails to
AI agents. Its main idea is simple: agent code can still own the reasoning loop,
but GuardLoop owns enforcement around risky operations such as LLM calls and
tool calls.

## Current Status

GuardLoop is currently published as version `0.2.0`.

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
3. GuardLoop returns structured run results instead of leaving failures hidden.
4. GuardLoop emits OpenTelemetry spans for agent runs, LLM calls, and tool calls.
5. GuardLoop is typed, tested, packaged, released on GitHub, and published on PyPI.

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

The runtime catches these controlled exceptions and converts them into
structured `RunResult` objects.

### 7. Demos

No-key demos:

```bash
uv run python examples/runaway_cost_prevention.py
uv run python examples/tool_circuit_breaker.py
```

The runaway-cost demo proves that GuardLoop stops an agent before the next LLM
request would exceed the configured budget.

The circuit-breaker demo proves that GuardLoop lets a flaky tool fail up to the
configured threshold, then rejects the next attempt without invoking the tool.

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
    A["User agent function"] --> R["GuardLoop.run()"]
    R --> C["RunContext"]
    R --> B["BudgetController"]
    R --> CB["CircuitBreakerRegistry"]
    R --> T["Telemetry"]
    C --> O["Wrapped OpenAI client"]
    C --> AN["Wrapped Anthropic client"]
    C --> TO["ToolRunner"]
    O --> B
    AN --> B
    TO --> B
    TO --> CB
    O --> T
    AN --> T
    TO --> T
```

Important modules:

- `src/guardloop/runtime.py`: main `GuardLoop` execution wrapper.
- `src/guardloop/context.py`: `RunContext` passed into user agents.
- `src/guardloop/budget.py`: cost, token, time, and tool-call enforcement.
- `src/guardloop/circuit_breaker.py`: per-tool circuit breaker state machines.
- `src/guardloop/providers/openai.py`: OpenAI Responses wrapper.
- `src/guardloop/providers/anthropic.py`: Anthropic Messages wrapper.
- `src/guardloop/tools.py`: protected sync/async tool execution.
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
- `GuardLoopError`
- `BudgetExceeded`
- `TokenLimitExceeded`
- `ToolCallLimitExceeded`
- `TimeLimitExceeded`
- `ModelPricingMissing`
- `TokenLimitMissing`
- `CircuitBreakerOpen`

Compatibility aliases:

- `AgentRuntime = GuardLoop`
- `AgentRuntimeError = GuardLoopError`

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
- OpenTelemetry span attributes.

Quality gates:

```bash
uv run pytest
uv run pytest --cov=guardloop
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
uvx twine check dist/guardloop-0.2.0.tar.gz dist/guardloop-0.2.0-py3-none-any.whl
```

## Packaging and Release State

GuardLoop is configured and published as a real Python package.

- PyPI project: `guardloop`
- GitHub release: `v0.2.0`
- Build artifacts: wheel and source distribution
- Trusted Publishing: GitHub Actions to PyPI
- GitHub environment: `pypi`

Publishing workflow:

- `.github/workflows/publish-pypi.yml`
- Builds distributions with `uv build`
- Uploads artifacts
- Publishes to PyPI using `pypa/gh-action-pypi-publish`
- Uses OIDC Trusted Publishing instead of a long-lived API token

## Current Limitations

GuardLoop is intentionally focused. It does not yet include:

- verifier/retry loops for validating final outputs,
- LangGraph adapter,
- OpenAI Agents SDK adapter,
- persistent circuit breaker state in Redis/database,
- provider-level circuit breakers,
- UI dashboard,
- Jaeger/Phoenix trace screenshots,
- automated docs site,
- semantic versioning automation.

These are good future layers, but v0.2 is already a complete portfolio-grade
library foundation.

## Future Goals

### v0.3: Verifier Retry Loop

Goal: let users attach verifiers that can accept or reject an agent result and
optionally trigger a bounded retry.

Planned capabilities:

- deterministic verifier functions,
- optional LLM-based verifier interface,
- retry budget separate from main loop budget,
- structured verifier result model,
- final `RunResult` metadata showing verifier decisions.

Portfolio value:

- demonstrates agent reliability,
- shows controlled self-correction,
- makes the project more than a wrapper around providers.

### v0.4: Framework Adapters

Goal: integrate GuardLoop with common agent frameworks without changing the core
runtime model.

Planned adapters:

- LangGraph adapter,
- OpenAI Agents SDK adapter.

Portfolio value:

- shows practical ecosystem integration,
- makes GuardLoop easier to demonstrate with real agent workflows,
- proves the core design is framework-agnostic.

### v0.5: Observability Polish

Goal: turn the telemetry foundation into strong portfolio artifacts.

Planned capabilities:

- Jaeger trace screenshots,
- Phoenix trace screenshots,
- example trace walkthrough,
- demo video script,
- blog-post style architecture writeup.

Portfolio value:

- gives recruiters/interviewers visual proof,
- makes the project easier to understand quickly,
- highlights production observability skills.

### v0.6: Persistence and Team Settings

Goal: support longer-lived runtime state and reusable policy configuration.

Possible capabilities:

- pluggable state backends for circuit breakers,
- YAML/TOML policy loading,
- organization/team default policies,
- model pricing update workflow.

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
- changelog,
- docs site,
- release checklist.

## Interview Talking Points

Use this short explanation:

GuardLoop is a production runtime layer for AI agents. Instead of building
another agent framework, it wraps the risky parts of an agent loop: model calls
and tool calls. It enforces cost, token, time, and tool-call limits; stops
runaway agent loops before sending expensive requests; opens circuit breakers
around flaky tools; and emits OpenTelemetry traces so failures are observable.
It is packaged as a typed Python library, published on PyPI, tested with fake
clients, and designed to integrate with frameworks later rather than depending
on one framework from the start.

Good questions to be ready for:

- Why use pre-flight token and cost checks before an LLM call?
- Why use `Decimal` for cost?
- Why does circuit breaker state live on `GuardLoop` instead of globally?
- Why direct provider wrappers before framework adapters?
- How would you add Redis-backed circuit breaker state?
- How would verifier retries interact with budget limits?
- How would you keep model pricing accurate over time?
- What should happen when provider usage metadata is missing?

## Recommended Next Step

The best next engineering milestone is v0.3: verifier retry loop.

Build it narrowly:

- add a `Verifier` protocol,
- add deterministic verifier examples first,
- return verifier decisions in `RunResult.metadata`,
- cap retries with the existing budget/time model,
- add a no-key demo where a bad answer is rejected and corrected.

That would make GuardLoop feel like a complete agent reliability toolkit:
budgets prevent runaway cost, circuit breakers protect external tools, and
verifiers protect final answer quality.
