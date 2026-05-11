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
model or rewriting your agent.

**v0.4.0 — LangGraph (shipped).** `guardloop.adapters.langgraph.guarded_graph(graph)`
returns a GuardLoop-compatible agent you pass to `runtime.run(...)`. LangGraph nodes
call LangChain chat models (which do not flow through GuardLoop's provider wrappers),
so the adapter binds a LangChain callback handler to the `RunContext`: it runs the
pre-flight budget check before each LLM call, records usage afterward, and routes
tool calls through the per-tool circuit breaker and the tool-call budget — so cost /
token / time caps, breakers, and `llm_call` / `tool_call` OpenTelemetry spans all
apply *inside* the graph. The verifier retry loop wraps the whole graph run, with
the feedback injected into a copy of the input state. Behind the `[langgraph]`
optional extra; no-key demo at `examples/langgraph_guarded.py`. Also adds a public
`RunContext.circuit_breakers` accessor and a CI workflow (pytest + ruff + pyright on
push/PR across Python 3.11–3.13).

**v0.4.1 — OpenAI Agents SDK (shipped).** The same shape with the SDK's run-hooks
instead of LangChain callbacks: `guardloop.adapters.openai_agents.guarded_runner(agent)`
returns a GuardLoop-compatible agent that calls `Runner.run` under the hood, with a
`GuardLoopRunHooks` (a `RunHooks` subclass) bound to the `RunContext` doing the
pre-flight budget check (`on_llm_start`), usage accounting (`on_llm_end`), and
breaker + tool-call-budget routing (`on_tool_start` / `on_tool_end`) — so the caps,
breakers, and `llm_call` / `tool_call` spans apply *inside* `Runner.run(...)`. The
verifier retry loop wraps the whole run (verifiers stay at the `runtime.run` level),
with feedback injected into a copy of the run input. Behind the `[openai-agents]`
optional extra; no-key demo at `examples/openai_agents_guarded.py`. Known gap: the
SDK has no tool-error lifecycle hook and turns a tool exception into an error string
fed back to the model, so the breaker tracks tool *attempts* / *successes* but not
*failures* from SDK-managed tools (an already-open breaker still blocks the next
call; route a tool body through `ctx.call_tool(...)` for full breaker semantics).
Streaming (`Runner.run_streamed`) is out of scope for this release.

## v0.5: Observability Depth

Build out the OpenTelemetry layer past spans: an OpenTelemetry **metrics**
provider (counters/histograms for cost, tokens, tool calls, and verifier
attempts), per-attempt `agent_attempt` span nesting so a verifier retry loop
reads as a tree rather than a flat list of sibling LLM calls, and a one-command
`docker-compose` stack (Jaeger for traces, Arize Phoenix for LLM-trace
inspection) that the no-key demos export to — so the traces and metrics are
inspectable end to end, not just emitted. Documentation artifacts (an
architecture write-up, a recorded walkthrough of the `agent_run -> llm_call ->
tool_call -> verifier_run` tree) fall out of having that stack.

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
