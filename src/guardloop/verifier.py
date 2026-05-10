"""Output verifiers and the verify-fix-retry loop building blocks.

A verifier runs after an agent finishes and decides whether the agent's output
is acceptable. When a verifier rejects the output, the runtime feeds the
verifier's feedback back to the agent (via :attr:`RunContext.retry_feedback`)
and re-invokes it, bounded by :attr:`VerifierConfig.max_retries` and the run's
shared budget. Verifiers are plain callables -- sync or async -- that take the
agent's return value plus a :class:`VerifierContext` and return a
:class:`VerifierResult` (or a bare ``bool`` / ``None`` shorthand).
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from guardloop.exceptions import VerifierExecutionError
from guardloop.telemetry.conventions import verifier_attributes
from guardloop.telemetry.tracer import Telemetry

EVENT_VERIFICATION_FAILED = "guardloop.verification.failed"
EVENT_RETRYING = "guardloop.verification.retrying"
EVENT_VERIFICATION_EXHAUSTED = "guardloop.verification.exhausted"

_NO_FEEDBACK = "Output rejected by verifier {name!r} (no feedback provided)."


class VerifierResult(BaseModel):
    """Verdict returned by a verifier for one attempt's output."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    feedback: str | None = None
    verifier_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("feedback")
    @classmethod
    def _strip_feedback(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


@dataclass(frozen=True, slots=True)
class VerifierContext:
    """Read-only context handed to a verifier for one attempt's output."""

    output: object
    attempt: int
    max_attempts: int
    retries_remaining: int
    prior_feedback: tuple[str, ...]
    run_args: tuple[Any, ...]
    run_kwargs: dict[str, Any]


VerifierReturn = VerifierResult | bool | None
Verifier = Callable[[object, VerifierContext], VerifierReturn | Awaitable[VerifierReturn]]


class VerifierConfig(BaseModel):
    """Configuration for the verify-fix-retry loop."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_retries: int = 1
    raise_on_failure: bool = False
    pass_feedback_to_agent: bool = True

    @field_validator("max_retries")
    @classmethod
    def _validate_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_retries must be non-negative")
        return value

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1


def feedback_for(result: VerifierResult) -> str:
    """Return ``result.feedback``, or a generic message if a rejection gave none."""

    if result.feedback:
        return result.feedback
    return _NO_FEEDBACK.format(name=result.verifier_name or "verifier")


def _verifier_name(verifier: Verifier, index: int) -> str:
    name = getattr(verifier, "__name__", None)
    if isinstance(name, str) and name and name != "<lambda>":
        return name
    return f"verifier_{index}"


def _normalize(raw: object, *, default_name: str) -> VerifierResult:
    if raw is None or raw is True:
        return VerifierResult(passed=True, verifier_name=default_name)
    if raw is False:
        return VerifierResult(passed=False, verifier_name=default_name)
    if isinstance(raw, VerifierResult):
        if raw.verifier_name is not None:
            return raw
        return raw.model_copy(update={"verifier_name": default_name})
    raise TypeError(
        f"Verifier {default_name!r} returned {type(raw).__name__}; "
        "expected VerifierResult, bool, or None."
    )


@dataclass(frozen=True)
class VerifierChain:
    """An ordered group of verifiers, run fail-fast against an agent's output."""

    verifiers: tuple[Verifier, ...] = ()

    @classmethod
    def from_iterable(cls, verifiers: Iterable[Verifier] | None) -> VerifierChain:
        return cls(tuple(verifiers or ()))

    def __bool__(self) -> bool:
        return bool(self.verifiers)

    def __len__(self) -> int:
        return len(self.verifiers)

    async def run(
        self,
        *,
        telemetry: Telemetry,
        output: object,
        attempt: int,
        max_attempts: int,
        prior_feedback: tuple[str, ...] = (),
        run_args: tuple[Any, ...] = (),
        run_kwargs: dict[str, Any] | None = None,
    ) -> VerifierResult:
        """Run every verifier in order; return the first failure, else a pass."""

        context = VerifierContext(
            output=output,
            attempt=attempt,
            max_attempts=max_attempts,
            retries_remaining=max(0, max_attempts - attempt),
            prior_feedback=prior_feedback,
            run_args=run_args,
            run_kwargs=dict(run_kwargs or {}),
        )
        for index, verifier in enumerate(self.verifiers):
            name = _verifier_name(verifier, index)
            with telemetry.start_span(
                f"verifier_run {name}",
                verifier_attributes(name=name, attempt=attempt, max_attempts=max_attempts),
            ) as span:
                try:
                    raw = verifier(output, context)
                    if inspect.isawaitable(raw):
                        raw = await raw
                    result = _normalize(raw, default_name=name)
                except Exception as exc:
                    telemetry.record_exception(span, exc)
                    raise VerifierExecutionError(name=name, attempt=attempt) from exc
                telemetry.set_attributes(
                    span,
                    verifier_attributes(
                        name=name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        passed=result.passed,
                    ),
                )
                telemetry.mark_ok(span)
                if not result.passed:
                    return result
        return VerifierResult(passed=True)


def non_empty(*, allow_whitespace: bool = False) -> Verifier:
    """Build a verifier that rejects empty output (whitespace-only too, by default)."""

    def non_empty(output: object, ctx: VerifierContext) -> VerifierResult:
        text = "" if output is None else str(output)
        candidate = text if allow_whitespace else text.strip()
        if candidate:
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, feedback="Output is empty; produce a non-empty answer.")

    return non_empty


def matches_regex(pattern: str | re.Pattern[str], *, flags: int = 0) -> Verifier:
    """Build a verifier that rejects output without a match for ``pattern`` (re.search)."""

    compiled = re.compile(pattern, flags) if isinstance(pattern, str) else pattern

    def matches_regex(output: object, ctx: VerifierContext) -> VerifierResult:
        text = "" if output is None else str(output)
        if compiled.search(text) is not None:
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            feedback=f"Output does not match the required pattern /{compiled.pattern}/.",
        )

    return matches_regex


def is_json_object(*, required_keys: Iterable[str] = ()) -> Verifier:
    """Build a verifier that rejects output that is not a JSON object with ``required_keys``."""

    keys = tuple(required_keys)

    def is_json_object(output: object, ctx: VerifierContext) -> VerifierResult:
        text = "" if output is None else str(output)
        try:
            parsed: object = json.loads(text)
        except json.JSONDecodeError as exc:
            return VerifierResult(
                passed=False,
                feedback=f"Output is not valid JSON ({exc.msg}); return a single JSON object.",
            )
        if not isinstance(parsed, dict):
            return VerifierResult(
                passed=False,
                feedback=(
                    f"Output must be a JSON object, got {type(parsed).__name__}; "
                    "return a single JSON object."
                ),
            )
        missing = [key for key in keys if key not in parsed]
        if missing:
            return VerifierResult(
                passed=False,
                feedback=f"JSON object is missing required key(s): {', '.join(missing)}.",
            )
        return VerifierResult(passed=True)

    return is_json_object
