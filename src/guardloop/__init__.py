"""GuardLoop public API."""

from guardloop.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerPolicy,
    CircuitBreakerSnapshot,
    CircuitBreakerState,
)
from guardloop.context import RunContext
from guardloop.exceptions import (
    BudgetExceeded,
    CircuitBreakerOpen,
    GuardLoopError,
    ModelPricingMissing,
    TimeLimitExceeded,
    TokenLimitExceeded,
    TokenLimitMissing,
    ToolCallLimitExceeded,
    VerificationFailed,
    VerifierExecutionError,
)
from guardloop.models import BudgetConfig, RunResult, TelemetryConfig
from guardloop.pricing import ModelPricing
from guardloop.runtime import GuardLoop
from guardloop.verifier import (
    Verifier,
    VerifierChain,
    VerifierConfig,
    VerifierContext,
    VerifierResult,
    is_json_object,
    matches_regex,
    non_empty,
)

AgentRuntime = GuardLoop
AgentRuntimeError = GuardLoopError

__all__ = [
    "AgentRuntime",
    "AgentRuntimeError",
    "BudgetConfig",
    "BudgetExceeded",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitBreakerPolicy",
    "CircuitBreakerSnapshot",
    "CircuitBreakerState",
    "GuardLoop",
    "GuardLoopError",
    "ModelPricing",
    "ModelPricingMissing",
    "RunContext",
    "RunResult",
    "TelemetryConfig",
    "TimeLimitExceeded",
    "TokenLimitExceeded",
    "TokenLimitMissing",
    "ToolCallLimitExceeded",
    "VerificationFailed",
    "Verifier",
    "VerifierChain",
    "VerifierConfig",
    "VerifierContext",
    "VerifierExecutionError",
    "VerifierResult",
    "is_json_object",
    "matches_regex",
    "non_empty",
]
