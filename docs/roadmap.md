# GuardLoop Roadmap

GuardLoop is organized around four pillars: resource limits, circuit breakers,
the self-healing verifier loop, and OpenTelemetry-native observability. Releases
ship one runnable, demoable increment at a time.

## v0.1: Runtime Guardrails

The first release proved the most interview-friendly production failure mode:
a runaway agent loop is stopped before the next LLM call would exceed the
configured budget. Hard caps on cost, tokens, wall-clock time, and tool calls;
direct OpenAI and Anthropic wrappers; OpenTelemetry spans for agent runs, LLM
calls, and tool calls.

## v0.2: Circuit Breakers

Per-tool state machines with `closed`, `open`, and `half_open` states. Tool
calls that repeatedly fail are rejected immediately during the open period;
breaker state lives on the `GuardLoop` instance, with a global default policy
plus per-tool overrides.

## v0.3: Verifier Retry Loop

Deterministic and async verifiers that can reject an agent's final output and
feed bounded feedback into a retry, all under the same shared budget and run
timeout. Built-in rule-based verifiers (`non_empty`, `matches_regex`,
`is_json_object`), structured `RunResult.verification_*` fields,
`ctx.retry_feedback`, an opt-in strict mode, and `verifier_run` spans.

## v0.4: Framework Adapters

Slot GuardLoop *under* common agent frameworks without changing the core runtime
model: a LangGraph adapter (consult the budget before each LLM node, inject a
`RunContext`, map the graph's tools onto `ToolRunner`) and an OpenAI Agents SDK
adapter. Optional `[langgraph]` / `[openai-agents]` extras and an example each.

## v0.5: Observability Polish

Turn the OpenTelemetry foundation into portfolio artifacts: a `docker-compose`
stack running Jaeger and Arize Phoenix, captured trace screenshots
(`agent_run -> llm_call -> tool_call -> verifier_run`), an architecture
write-up, and a demo-video script. Optionally: per-attempt `agent_attempt`
span nesting and an OpenTelemetry metrics layer.

## v0.6: Persistence and Team Settings

Pluggable state backends behind a `CircuitBreakerStore` protocol (Redis/SQL so
breaker state survives restarts and is shared across workers); YAML/TOML policy
loading (`GuardLoop.from_config(...)`) and org/team default policies. Natural
home for loop detection (repeated `tool + args` -> `terminated_reason="loop_detected"`),
multi-model pricing (Gemini, Groq, Mistral), and a LiteLLM integration.

## v1.0: Stable Guardrail Runtime

Freeze the public API (stability policy; `__all__` is the contract), graduate
`CHANGELOG.md` to source of truth, stand up an auto-published docs site, broaden
pricing/model coverage with a documented update workflow, add a release
checklist and stronger compatibility tests, and harden the token story
(streaming-response accounting, Anthropic token-counting API).
