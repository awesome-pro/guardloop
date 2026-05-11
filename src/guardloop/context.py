"""RunContext passed to user agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from guardloop.budget import BudgetController
from guardloop.circuit_breaker import CircuitBreakerRegistry
from guardloop.providers.anthropic import WrappedAnthropicClient
from guardloop.providers.openai import WrappedOpenAIClient
from guardloop.telemetry.tracer import Telemetry
from guardloop.tools import ToolRunner


class RunContext:
    """Runtime services available to an agent during one execution.

    After a verifier rejects an attempt's output and retries remain, the runtime
    appends the verifier's feedback to :attr:`retry_feedback` and re-invokes the
    agent with the same ``*args`` and ``**kwargs``. Agents that want to
    self-correct should read :attr:`retry_feedback` (for example, inject it into
    the next prompt). Mutating :attr:`retry_feedback` or :attr:`attempt` has no
    effect on the runtime.
    """

    def __init__(
        self,
        *,
        budget: BudgetController,
        telemetry: Telemetry,
        circuit_breakers: CircuitBreakerRegistry,
        openai_client: Any | None = None,
        anthropic_client: Any | None = None,
    ) -> None:
        self.budget = budget
        self.telemetry = telemetry
        self.retry_feedback: list[str] = []
        self.attempt: int = 1
        self._circuit_breakers = circuit_breakers
        self._raw_openai_client = openai_client
        self._raw_anthropic_client = anthropic_client
        self._openai: WrappedOpenAIClient | None = None
        self._anthropic: WrappedAnthropicClient | None = None
        self._tools = ToolRunner(budget, telemetry, circuit_breakers)

    @property
    def circuit_breakers(self) -> CircuitBreakerRegistry:
        """The per-tool circuit breaker registry shared by this run.

        Exposed so framework adapters can route tool calls through the breaker
        (``before_call`` / ``record_success`` / ``record_failure``) and so agents
        can inspect breaker state. The registry persists on the ``GuardLoop``
        instance across runs.
        """

        return self._circuit_breakers

    @property
    def openai(self) -> WrappedOpenAIClient:
        if self._openai is None:
            client = self._raw_openai_client
            if client is None:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
            self._openai = WrappedOpenAIClient(client, self.budget, self.telemetry)
        return self._openai

    @property
    def anthropic(self) -> WrappedAnthropicClient:
        if self._anthropic is None:
            client = self._raw_anthropic_client
            if client is None:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic()
            self._anthropic = WrappedAnthropicClient(client, self.budget, self.telemetry)
        return self._anthropic

    def wrap_tool(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        return self._tools.wrap(name, func)

    async def call_tool(
        self,
        name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return await self._tools.call(name, func, *args, **kwargs)
