"""Runtime-aware tool wrappers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from agentruntime.budget import BudgetController
from agentruntime.telemetry.conventions import tool_attributes
from agentruntime.telemetry.tracer import Telemetry


class ToolRunner:
    """Wrap arbitrary sync or async Python callables with runtime checks."""

    def __init__(self, budget: BudgetController, telemetry: Telemetry) -> None:
        self._budget = budget
        self._telemetry = telemetry

    def wrap(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            return await self.call(name, func, *args, **kwargs)

        return wrapped

    async def call(self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._budget.record_tool_call_started(name)
        with self._telemetry.start_span(
            f"tool_call {name}",
            tool_attributes(tool_name=name, calls_used=self._budget.tool_calls),
        ) as span:
            try:
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                self._telemetry.mark_ok(span)
                return result
            except Exception as exc:
                self._telemetry.record_exception(span, exc)
                raise
