"""Public exception hierarchy for controlled runtime stops."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


class GuardLoopError(Exception):
    """Base class for all controlled GuardLoop exceptions."""

    terminated_reason = "runtime_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class BudgetExceeded(GuardLoopError):
    """Raised when a call would exceed the configured cost cap."""

    terminated_reason = "budget_exceeded"

    def __init__(
        self,
        message: str,
        *,
        limit: Decimal | None = None,
        current: Decimal | None = None,
        projected: Decimal | None = None,
    ) -> None:
        details = {"limit": limit, "current": current, "projected": projected}
        super().__init__(message, details=details)


class TokenLimitExceeded(GuardLoopError):
    """Raised when a call would exceed the configured token cap."""

    terminated_reason = "token_limit_exceeded"


class ToolCallLimitExceeded(GuardLoopError):
    """Raised when a tool call would exceed the configured tool-call cap."""

    terminated_reason = "tool_call_limit_exceeded"


class CircuitBreakerOpen(GuardLoopError):
    """Raised when a tool circuit breaker rejects a call."""

    terminated_reason = "circuit_breaker_open"

    def __init__(
        self,
        *,
        tool_name: str,
        state: str,
        failure_count: int,
        remaining_open_seconds: float,
    ) -> None:
        details = {
            "tool_name": tool_name,
            "state": state,
            "failure_count": failure_count,
            "remaining_open_seconds": remaining_open_seconds,
        }
        super().__init__(
            f"Circuit breaker for tool '{tool_name}' is open for another "
            f"{remaining_open_seconds:.3f}s.",
            details=details,
        )


class TimeLimitExceeded(GuardLoopError):
    """Raised when the run exceeds the configured wall-clock cap."""

    terminated_reason = "timeout"


class ModelPricingMissing(GuardLoopError):
    """Raised when no pricing entry exists for a provider/model pair."""

    terminated_reason = "model_pricing_missing"


class TokenLimitMissing(GuardLoopError):
    """Raised when the runtime cannot reserve output tokens before an LLM call."""

    terminated_reason = "token_limit_missing"


AgentRuntimeError = GuardLoopError
