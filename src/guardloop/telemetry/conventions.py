"""OpenTelemetry attribute names behind one compatibility layer.

The GenAI semantic conventions are still evolving, so provider wrappers should
use these constants instead of scattering raw strings through the codebase.
"""

from __future__ import annotations

from decimal import Decimal

AttributeValue = str | bool | int | float
Attributes = dict[str, AttributeValue]

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

GUARDLOOP_COST_USD = "guardloop.budget.cost_usd"
GUARDLOOP_ESTIMATED_COST_USD = "guardloop.budget.estimated_cost_usd"
GUARDLOOP_TOOL_NAME = "guardloop.tool.name"
GUARDLOOP_TOOL_CALLS_USED = "guardloop.tool.calls_used"
GUARDLOOP_CIRCUIT_BREAKER_STATE = "guardloop.circuit_breaker.state"
GUARDLOOP_CIRCUIT_BREAKER_FAILURE_COUNT = "guardloop.circuit_breaker.failure_count"
GUARDLOOP_CIRCUIT_BREAKER_BLOCKED = "guardloop.circuit_breaker.blocked"
GUARDLOOP_CIRCUIT_BREAKER_REMAINING_OPEN_SECONDS = (
    "guardloop.circuit_breaker.remaining_open_seconds"
)
GUARDLOOP_VERIFIER_NAME = "guardloop.verifier.name"
GUARDLOOP_VERIFIER_PASSED = "guardloop.verifier.passed"
GUARDLOOP_VERIFIER_ATTEMPT = "guardloop.verifier.attempt"
GUARDLOOP_VERIFIER_MAX_ATTEMPTS = "guardloop.verifier.max_attempts"
GUARDLOOP_VERIFICATION_PASSED = "guardloop.verification.passed"
GUARDLOOP_VERIFICATION_ATTEMPTS = "guardloop.verification.attempts"
GUARDLOOP_TERMINATED_REASON = "guardloop.terminated_reason"


def decimal_attr(value: Decimal) -> float:
    return float(value)


def run_attributes() -> Attributes:
    return {"guardloop.operation": "agent.run"}


def llm_request_attributes(
    *,
    provider: str,
    model: str,
    estimated_input_tokens: int,
    reserved_output_tokens: int,
    estimated_cost_usd: Decimal,
) -> Attributes:
    return {
        GEN_AI_SYSTEM: provider,
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        "guardloop.estimated_input_tokens": estimated_input_tokens,
        "guardloop.reserved_output_tokens": reserved_output_tokens,
        GUARDLOOP_ESTIMATED_COST_USD: decimal_attr(estimated_cost_usd),
    }


def llm_response_attributes(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: Decimal,
) -> Attributes:
    return {
        GEN_AI_RESPONSE_MODEL: model,
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        GUARDLOOP_COST_USD: decimal_attr(cost_usd),
    }


def tool_attributes(
    *,
    tool_name: str,
    calls_used: int,
    breaker_state: str | None = None,
    breaker_failure_count: int | None = None,
    breaker_blocked: bool | None = None,
    breaker_remaining_open_seconds: float | None = None,
) -> Attributes:
    attributes: Attributes = {
        GUARDLOOP_TOOL_NAME: tool_name,
        GUARDLOOP_TOOL_CALLS_USED: calls_used,
    }
    if breaker_state is not None:
        attributes[GUARDLOOP_CIRCUIT_BREAKER_STATE] = breaker_state
    if breaker_failure_count is not None:
        attributes[GUARDLOOP_CIRCUIT_BREAKER_FAILURE_COUNT] = breaker_failure_count
    if breaker_blocked is not None:
        attributes[GUARDLOOP_CIRCUIT_BREAKER_BLOCKED] = breaker_blocked
    if breaker_remaining_open_seconds is not None:
        attributes[GUARDLOOP_CIRCUIT_BREAKER_REMAINING_OPEN_SECONDS] = (
            breaker_remaining_open_seconds
        )
    return attributes


def verifier_attributes(
    *,
    name: str,
    attempt: int,
    max_attempts: int,
    passed: bool | None = None,
) -> Attributes:
    attributes: Attributes = {
        GUARDLOOP_VERIFIER_NAME: name,
        GUARDLOOP_VERIFIER_ATTEMPT: attempt,
        GUARDLOOP_VERIFIER_MAX_ATTEMPTS: max_attempts,
    }
    if passed is not None:
        attributes[GUARDLOOP_VERIFIER_PASSED] = passed
    return attributes


def verification_summary_attributes(*, passed: bool | None, attempts: int) -> Attributes:
    attributes: Attributes = {GUARDLOOP_VERIFICATION_ATTEMPTS: attempts}
    if passed is not None:
        attributes[GUARDLOOP_VERIFICATION_PASSED] = passed
    return attributes
