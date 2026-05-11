# Changelog

All notable changes to GuardLoop are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0:
minor releases may include breaking changes).

## [0.4.1] - 2026-05-11

### Added

- **OpenAI Agents SDK adapter (`guardloop.adapters.openai_agents`).**
  `guarded_runner(agent)` returns a GuardLoop-compatible agent callable you pass
  to `GuardLoop.run(...)`; a `GuardLoopRunHooks` (a subclass of the SDK's
  `RunHooks`) bound to the `RunContext` runs the pre-flight budget check before
  each LLM call (`on_llm_start`), records actual usage afterward (`on_llm_end`),
  and routes tool calls through the per-tool circuit breaker and the tool-call
  budget (`on_tool_start` / `on_tool_end`) — so cost / token / time caps,
  breakers, and `llm_call` / `tool_call` OpenTelemetry spans all apply *inside* a
  `Runner.run(...)`. The verifier retry loop wraps the whole run, with verifier
  feedback injected into a copy of the run input (`feedback_to_input` to
  customise; `output_from_result` to customise how the answer is extracted).
  `guarded_runner(..., reserved_output_tokens=N)` sets the output-token
  reservation for the pre-flight check (default `1024`), since the SDK's chat
  models often leave `model_settings.max_tokens` unset. Because the SDK wraps
  exceptions raised from its tool lifecycle hooks in `agents.exceptions.UserError`,
  `guarded_runner` unwraps a `GuardLoopError` from the exception chain before
  re-raising, so a tripped guard still becomes a clean `RunResult` with the right
  `terminated_reason`. Behind the new `openai-agents` optional extra
  (`pip install "guardloop[openai-agents]"`).
- `guardloop.adapters.openai_agents` exports `guarded_runner` and
  `GuardLoopRunHooks`. (Adapters are intentionally not re-exported from the
  top-level `guardloop` package, so `import guardloop` stays dependency-light.)
- No-key demo `examples/openai_agents_guarded.py`.

### Known limitations

- The OpenAI Agents SDK has no `on_tool_error` lifecycle hook and, by default,
  turns a tool exception into an error *string* fed back to the model (so
  `on_tool_end` fires with that string). The adapter therefore records tool
  *attempts* and *successes* but not *failures* — a flaky SDK-managed tool will
  not open the breaker on its own. The breaker's blocking behaviour (an
  already-open breaker rejects the next SDK tool call) does apply; route a tool
  body through `ctx.call_tool(...)` for full breaker semantics. Streaming
  (`Runner.run_streamed`) is out of scope for this release (usage is still
  accounted via `on_llm_end`).

## [0.4.0] - 2026-05-11

### Added

- **LangGraph adapter (`guardloop.adapters.langgraph`).** `guarded_graph(graph)`
  returns a GuardLoop-compatible agent callable you pass to `GuardLoop.run(...)`;
  a `GuardLoopCallbackHandler` (a synchronous LangChain `BaseCallbackHandler`)
  bound to the `RunContext` runs the pre-flight budget check before each LLM call,
  records actual usage afterward, and routes tool calls through the per-tool
  circuit breaker and the tool-call budget — so cost / token / time caps, breakers,
  and `llm_call` / `tool_call` OpenTelemetry spans all apply *inside* a LangGraph
  run. The verifier retry loop wraps the whole graph run, with verifier feedback
  injected into a copy of the input state (`feedback_to_state` to customise).
  `guarded_graph(..., reserved_output_tokens=N)` sets the output-token reservation
  for the pre-flight check (default `1024`), since LangChain chat models often omit
  `max_tokens`. Behind the new `langgraph` optional extra
  (`pip install "guardloop[langgraph]"`).
- `guardloop.adapters` subpackage; `guardloop.adapters.langgraph` exports
  `guarded_graph` and `GuardLoopCallbackHandler`. (Adapters are intentionally not
  re-exported from the top-level `guardloop` package, so `import guardloop` stays
  dependency-light.)
- `RunContext.circuit_breakers` — public read-only access to the per-tool circuit
  breaker registry (used by adapters; also handy for inspecting breaker state).
- No-key demo `examples/langgraph_guarded.py`.
- `.github/workflows/ci.yml` — runs pytest + ruff + pyright on push / pull request
  across Python 3.11–3.13.

### Changed

- `pyproject.toml`: new `langgraph` optional-dependency extra; `langgraph` /
  `langchain-core` added to the dev dependency group; `langgraph` keyword.

## [0.3.0] - 2026-05-10

### Added

- **Verifier retry loop (Pillar 3 / self-healing).** After an agent finishes,
  GuardLoop can run a chain of verifiers against the output; on rejection it
  appends the verifier's feedback to `RunContext.retry_feedback` and re-invokes
  the agent, bounded by `VerifierConfig.max_retries`. All attempts share the
  same budget (cost / tokens / time / tool calls) and the run's single
  `asyncio.timeout`, so a verifier loop cannot bypass any guardrail.
- New module `guardloop.verifier` with public exports: `Verifier` (callable
  type alias — sync or async, returning `VerifierResult`, `bool`, or `None`),
  `VerifierResult`, `VerifierContext`, `VerifierConfig`, and `VerifierChain`.
- Built-in rule-based verifier factories: `non_empty()`, `matches_regex(...)`,
  `is_json_object(required_keys=...)`.
- `GuardLoop(verifiers=[...], verifier_config=VerifierConfig(...))` constructor
  parameters and `GuardLoop.add_verifier(fn)`.
- `RunResult` fields: `verification_passed: bool | None`,
  `verification_attempts: int`, `verification_feedback: list[str]`.
- `RunContext.retry_feedback: list[str]` and `RunContext.attempt: int`.
- New exceptions `VerificationFailed` (`terminated_reason="verification_failed"`,
  raised only in strict mode) and `VerifierExecutionError`
  (`terminated_reason="verifier_error"`, raised when a verifier itself throws).
- OpenTelemetry: `verifier_run <name>` child spans, `agent_run` attributes
  `guardloop.verification.passed` / `guardloop.verification.attempts`, and
  `guardloop.verification.failed` / `.retrying` / `.exhausted` span events.
- No-key demo `examples/verifier_retry_loop.py`.

### Changed

- When verification ultimately fails (retries exhausted), `RunResult.success`
  is `False` with `terminated_reason="verification_failed"`, but `output` still
  holds the last attempt's text — consistent with how budget/timeout stops
  report. Set `VerifierConfig(raise_on_failure=True)` for strict behavior
  (surfaces a `VerificationFailed` with `output=None` and details in
  `metadata`).
- `pyproject.toml`: `Changelog` URL now points at this file.

## [0.2.0] - 2026

### Added

- Per-tool circuit breakers with `closed` / `open` / `half_open` states, a
  global default policy plus per-tool overrides, breaker state that persists on
  the `GuardLoop` instance across runs, and `runtime.circuit_breaker_snapshots()`
  / `runtime.reset_circuit_breakers()`.
- `ctx.call_tool(...)` / `ctx.wrap_tool(...)` route tool calls through the
  breaker before the tool-call budget is incremented.
- `CircuitBreakerOpen` exception and circuit-breaker OpenTelemetry attributes
  on tool spans.
- No-key demo `examples/tool_circuit_breaker.py`.

## [0.1.0] - 2026

### Added

- Async runtime wrapper: `GuardLoop.run(agent, ...)` returns a structured
  `RunResult`; controlled stops become `success=False` with a
  `terminated_reason` instead of raised exceptions.
- Hard budget caps for cost (`Decimal`), tokens, wall-clock time, and tool
  calls, enforced pre-flight before each LLM request.
- Direct wrappers for `AsyncOpenAI.responses.create` and
  `AsyncAnthropic.messages.create` with usage accounting and pricing.
- OpenTelemetry spans for agent runs, LLM calls, and tool calls (core depends
  only on `opentelemetry-api`; exporters via the `otel` extra).
- Public exception hierarchy: `GuardLoopError`, `BudgetExceeded`,
  `TokenLimitExceeded`, `ToolCallLimitExceeded`, `TimeLimitExceeded`,
  `ModelPricingMissing`, `TokenLimitMissing`; `AgentRuntime` / `AgentRuntimeError`
  compatibility aliases.
- No-key demo `examples/runaway_cost_prevention.py`; packaged and published to
  PyPI via GitHub Actions OIDC Trusted Publishing.

[0.4.1]: https://github.com/awesome-pro/guardloop/releases/tag/v0.4.1
[0.4.0]: https://github.com/awesome-pro/guardloop/releases/tag/v0.4.0
[0.3.0]: https://github.com/awesome-pro/guardloop/releases/tag/v0.3.0
[0.2.0]: https://github.com/awesome-pro/guardloop/releases/tag/v0.2.0
[0.1.0]: https://github.com/awesome-pro/guardloop/releases/tag/v0.1.0
