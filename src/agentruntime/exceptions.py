"""Public exception hierarchy for controlled runtime stops."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


class AgentRuntimeError(Exception):
    """Base class for all controlled AgentRuntime exceptions."""

    terminated_reason = "runtime_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class BudgetExceeded(AgentRuntimeError):
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


class TokenLimitExceeded(AgentRuntimeError):
    """Raised when a call would exceed the configured token cap."""

    terminated_reason = "token_limit_exceeded"


class ToolCallLimitExceeded(AgentRuntimeError):
    """Raised when a tool call would exceed the configured tool-call cap."""

    terminated_reason = "tool_call_limit_exceeded"


class TimeLimitExceeded(AgentRuntimeError):
    """Raised when the run exceeds the configured wall-clock cap."""

    terminated_reason = "timeout"


class ModelPricingMissing(AgentRuntimeError):
    """Raised when no pricing entry exists for a provider/model pair."""

    terminated_reason = "model_pricing_missing"


class TokenLimitMissing(AgentRuntimeError):
    """Raised when the runtime cannot reserve output tokens before an LLM call."""

    terminated_reason = "token_limit_missing"
