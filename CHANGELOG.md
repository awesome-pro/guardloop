# Changelog

All notable changes to GuardLoop are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0:
minor releases may include breaking changes).

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

[0.3.0]: https://github.com/awesome-pro/guardloop/releases/tag/v0.3.0
[0.2.0]: https://github.com/awesome-pro/guardloop/releases/tag/v0.2.0
[0.1.0]: https://github.com/awesome-pro/guardloop/releases/tag/v0.1.0
