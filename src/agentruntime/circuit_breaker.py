"""Per-tool circuit breaker state machines."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentruntime.exceptions import CircuitBreakerOpen


class CircuitBreakerState(StrEnum):
    """Public circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerPolicy(BaseModel):
    """Failure policy for one tool circuit breaker."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0
    half_open_success_threshold: int = 1

    @field_validator("failure_threshold", "half_open_success_threshold")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("circuit breaker thresholds must be at least 1")
        return value

    @field_validator("recovery_timeout_seconds")
    @classmethod
    def _validate_recovery_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("recovery_timeout_seconds must be greater than zero")
        return value


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for runtime tools."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    default: CircuitBreakerPolicy = Field(default_factory=CircuitBreakerPolicy)
    tool_overrides: dict[str, CircuitBreakerPolicy] = Field(default_factory=dict)


class CircuitBreakerSnapshot(BaseModel):
    """Point-in-time circuit breaker state for inspection and metadata."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    state: CircuitBreakerState
    failure_count: int = 0
    consecutive_successes: int = 0
    opened_at: float | None = None
    remaining_open_seconds: float = 0.0


@dataclass
class CircuitBreakerDecision:
    """Internal decision returned after a breaker state check or update."""

    snapshot: CircuitBreakerSnapshot
    events: tuple[str, ...] = ()


@dataclass
class _CircuitBreakerRecord:
    policy: CircuitBreakerPolicy
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    consecutive_successes: int = 0
    opened_at: float | None = None


Clock = Callable[[], float]

EVENT_OPENED = "agentruntime.circuit_breaker.opened"
EVENT_REOPENED = "agentruntime.circuit_breaker.reopened"
EVENT_HALF_OPENED = "agentruntime.circuit_breaker.half_opened"
EVENT_CLOSED = "agentruntime.circuit_breaker.closed"


class CircuitBreakerRegistry:
    """Thread-safe in-memory registry of per-tool circuit breakers."""

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or CircuitBreakerConfig()
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._breakers: dict[str, _CircuitBreakerRecord] = {}

    def before_call(self, tool_name: str) -> CircuitBreakerDecision | None:
        policy = self._policy_for(tool_name)
        if policy is None:
            return None

        with self._lock:
            now = self._clock()
            breaker = self._breaker_for(tool_name, policy)
            if breaker.state != CircuitBreakerState.OPEN:
                return CircuitBreakerDecision(snapshot=self._snapshot(tool_name, breaker, now))

            remaining = self._remaining_open_seconds(breaker, now)
            if remaining > 0:
                snapshot = self._snapshot(tool_name, breaker, now)
                raise CircuitBreakerOpen(
                    tool_name=tool_name,
                    state=snapshot.state.value,
                    failure_count=snapshot.failure_count,
                    remaining_open_seconds=snapshot.remaining_open_seconds,
                )

            breaker.state = CircuitBreakerState.HALF_OPEN
            breaker.consecutive_successes = 0
            return CircuitBreakerDecision(
                snapshot=self._snapshot(tool_name, breaker, now),
                events=(EVENT_HALF_OPENED,),
            )

    def record_success(self, tool_name: str) -> CircuitBreakerDecision | None:
        policy = self._policy_for(tool_name)
        if policy is None:
            return None

        with self._lock:
            now = self._clock()
            breaker = self._breaker_for(tool_name, policy)
            events: tuple[str, ...] = ()

            if breaker.state == CircuitBreakerState.HALF_OPEN:
                breaker.consecutive_successes += 1
                if breaker.consecutive_successes >= breaker.policy.half_open_success_threshold:
                    breaker.state = CircuitBreakerState.CLOSED
                    breaker.failure_count = 0
                    breaker.consecutive_successes = 0
                    breaker.opened_at = None
                    events = (EVENT_CLOSED,)
            elif breaker.state == CircuitBreakerState.CLOSED:
                breaker.failure_count = 0
                breaker.consecutive_successes = 0

            return CircuitBreakerDecision(
                snapshot=self._snapshot(tool_name, breaker, now), events=events
            )

    def record_failure(self, tool_name: str) -> CircuitBreakerDecision | None:
        policy = self._policy_for(tool_name)
        if policy is None:
            return None

        with self._lock:
            now = self._clock()
            breaker = self._breaker_for(tool_name, policy)
            events: tuple[str, ...] = ()

            if breaker.state == CircuitBreakerState.HALF_OPEN:
                breaker.state = CircuitBreakerState.OPEN
                breaker.failure_count = max(1, breaker.failure_count)
                breaker.consecutive_successes = 0
                breaker.opened_at = now
                events = (EVENT_REOPENED,)
            else:
                breaker.failure_count += 1
                breaker.consecutive_successes = 0
                if (
                    breaker.state == CircuitBreakerState.CLOSED
                    and breaker.failure_count >= breaker.policy.failure_threshold
                ):
                    breaker.state = CircuitBreakerState.OPEN
                    breaker.opened_at = now
                    events = (EVENT_OPENED,)

            return CircuitBreakerDecision(
                snapshot=self._snapshot(tool_name, breaker, now), events=events
            )

    def snapshots(self) -> dict[str, CircuitBreakerSnapshot]:
        with self._lock:
            now = self._clock()
            return {
                tool_name: self._snapshot(tool_name, breaker, now)
                for tool_name, breaker in sorted(self._breakers.items())
            }

    def reset(self, tool_name: str | None = None) -> None:
        with self._lock:
            if tool_name is None:
                self._breakers.clear()
                return
            self._breakers.pop(tool_name, None)

    def _policy_for(self, tool_name: str) -> CircuitBreakerPolicy | None:
        if not self._config.enabled:
            return None

        policy = self._config.tool_overrides.get(tool_name, self._config.default)
        if not policy.enabled:
            return None
        return policy

    def _breaker_for(self, tool_name: str, policy: CircuitBreakerPolicy) -> _CircuitBreakerRecord:
        breaker = self._breakers.get(tool_name)
        if breaker is None:
            breaker = _CircuitBreakerRecord(policy=policy)
            self._breakers[tool_name] = breaker
        return breaker

    def _remaining_open_seconds(self, breaker: _CircuitBreakerRecord, now: float) -> float:
        if breaker.state != CircuitBreakerState.OPEN or breaker.opened_at is None:
            return 0.0
        opened_until = breaker.opened_at + breaker.policy.recovery_timeout_seconds
        return max(0.0, opened_until - now)

    def _snapshot(
        self, tool_name: str, breaker: _CircuitBreakerRecord, now: float
    ) -> CircuitBreakerSnapshot:
        return CircuitBreakerSnapshot(
            tool_name=tool_name,
            state=breaker.state,
            failure_count=breaker.failure_count,
            consecutive_successes=breaker.consecutive_successes,
            opened_at=breaker.opened_at,
            remaining_open_seconds=self._remaining_open_seconds(breaker, now),
        )
