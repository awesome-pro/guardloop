"""Pydantic models for runtime configuration and results."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

DecimalInput = Decimal | str | int | float


class BudgetConfig(BaseModel):
    """Hard resource limits for a single agent run."""

    model_config = ConfigDict(frozen=True)

    cost_limit_usd: DecimalInput | None = Field(default=None)
    token_limit: int | None = Field(default=None)
    time_limit_seconds: float | None = Field(default=None)
    tool_call_limit: int | None = Field(default=None)

    @field_validator("cost_limit_usd", mode="before")
    @classmethod
    def _parse_decimal(cls, value: object) -> object:
        if value is None or isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    @field_validator("cost_limit_usd")
    @classmethod
    def _validate_cost_limit(cls, value: DecimalInput | None) -> DecimalInput | None:
        decimal_value = Decimal(str(value)) if value is not None else None
        if decimal_value is not None and decimal_value < 0:
            raise ValueError("cost_limit_usd must be non-negative")
        return value

    @property
    def cost_limit(self) -> Decimal | None:
        if self.cost_limit_usd is None:
            return None
        if isinstance(self.cost_limit_usd, Decimal):
            return self.cost_limit_usd
        return Decimal(str(self.cost_limit_usd))

    @field_validator("token_limit", "tool_call_limit")
    @classmethod
    def _validate_optional_non_negative_int(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("limits must be non-negative")
        return value

    @field_validator("time_limit_seconds")
    @classmethod
    def _validate_time_limit(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("time_limit_seconds must be greater than zero")
        return value


class TelemetryConfig(BaseModel):
    """OpenTelemetry behavior for runtime spans."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    service_name: str = "guardloop"
    otlp_endpoint: str | None = None
    console_exporter: bool = False


class RunResult(BaseModel):
    """Structured result returned from every runtime execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: str | None = None
    success: bool
    cost_usd: Decimal = Decimal("0")
    estimated_cost_usd: Decimal = Decimal("0")
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    tool_calls: int = 0
    verification_passed: bool | None = None
    verification_attempts: int = 0
    verification_feedback: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    terminated_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("cost_usd", "estimated_cost_usd")
    def _serialize_decimal(self, value: Decimal) -> str:
        return str(value)
