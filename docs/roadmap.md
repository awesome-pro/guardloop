# GuardLoop Roadmap

## v0.1: Runtime Guardrails

The first release proves the most interview-friendly production failure mode:
a runaway agent loop is stopped before the next LLM call would exceed the
configured budget.

## v0.2: Circuit Breakers

Implemented per-tool state machines with closed, open, and half-open states.
Tool calls that repeatedly fail are rejected immediately during the open period.

## v0.3: Verifier Retry Loop

Next, add deterministic and LLM-based verifiers that can reject final outputs
and send bounded feedback into a retry attempt.

## v0.4: Framework Adapters

Add adapters for LangGraph and OpenAI Agents SDK without changing the core
runtime model.

## v0.5: Portfolio Polish

Add Jaeger/Phoenix trace screenshots, a demo video, a blog post, and release
packaging for GitHub and PyPI.
