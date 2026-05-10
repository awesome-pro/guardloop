"""Resource accounting and hard pre-flight budget checks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal

from guardloop.exceptions import (
    BudgetExceeded,
    TimeLimitExceeded,
    TokenLimitExceeded,
    TokenLimitMissing,
    ToolCallLimitExceeded,
)
from guardloop.models import BudgetConfig
from guardloop.pricing import ModelPricing, PricingCatalog


@dataclass(frozen=True, slots=True)
class LLMPreflight:
    provider: str
    model: str
    estimated_input_tokens: int
    reserved_output_tokens: int
    estimated_cost_usd: Decimal
    pricing: ModelPricing


class BudgetController:
    """Single source of truth for one runtime execution's resource usage."""

    def __init__(self, config: BudgetConfig, pricing_catalog: PricingCatalog) -> None:
        self.config = config
        self.pricing_catalog = pricing_catalog
        self._started_at = time.monotonic()
        self._cost_usd = Decimal("0")
        self._estimated_cost_usd = Decimal("0")
        self._input_tokens = 0
        self._output_tokens = 0
        self._tool_calls = 0

    @property
    def cost_usd(self) -> Decimal:
        return self._cost_usd

    @property
    def estimated_cost_usd(self) -> Decimal:
        return self._estimated_cost_usd

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def tokens_used(self) -> int:
        return self._input_tokens + self._output_tokens

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def duration_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def check_time(self) -> None:
        if (
            self.config.time_limit_seconds is not None
            and self.duration_seconds > self.config.time_limit_seconds
        ):
            raise TimeLimitExceeded(
                f"Run exceeded time limit of {self.config.time_limit_seconds:.3f}s.",
                details={
                    "limit_seconds": self.config.time_limit_seconds,
                    "duration_seconds": self.duration_seconds,
                },
            )

    def check_llm_call(
        self,
        *,
        provider: str,
        model: str,
        estimated_input_tokens: int,
        reserved_output_tokens: int | None,
    ) -> LLMPreflight:
        self.check_time()
        if reserved_output_tokens is None or reserved_output_tokens <= 0:
            raise TokenLimitMissing(
                "LLM calls must include a positive max output token limit so the runtime can "
                "reserve worst-case spend before the request."
            )

        pricing = self.pricing_catalog.get(provider, model)
        projected_tokens = self.tokens_used + estimated_input_tokens + reserved_output_tokens
        if self.config.token_limit is not None and projected_tokens > self.config.token_limit:
            raise TokenLimitExceeded(
                "LLM call would exceed token_limit before the request is sent.",
                details={
                    "limit": self.config.token_limit,
                    "current_tokens": self.tokens_used,
                    "estimated_input_tokens": estimated_input_tokens,
                    "reserved_output_tokens": reserved_output_tokens,
                    "projected_tokens": projected_tokens,
                },
            )

        projected_call_cost = pricing.estimate_cost(
            input_tokens=estimated_input_tokens,
            output_tokens=reserved_output_tokens,
        )
        projected_cost = self.cost_usd + projected_call_cost
        cost_limit = self.config.cost_limit
        if cost_limit is not None and projected_cost > cost_limit:
            raise BudgetExceeded(
                "LLM call would exceed cost_limit_usd before the request is sent.",
                limit=cost_limit,
                current=self.cost_usd,
                projected=projected_cost,
            )

        self._estimated_cost_usd += projected_call_cost
        return LLMPreflight(
            provider=provider,
            model=model,
            estimated_input_tokens=estimated_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            estimated_cost_usd=projected_call_cost,
            pricing=pricing,
        )

    def record_llm_call(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        self.check_time()
        pricing = self.pricing_catalog.get(provider, model)
        actual_cost = pricing.estimate_cost(input_tokens=input_tokens, output_tokens=output_tokens)
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cost_usd += actual_cost

        if self.config.token_limit is not None and self.tokens_used > self.config.token_limit:
            raise TokenLimitExceeded(
                "Actual provider usage exceeded token_limit after the request completed.",
                details={"limit": self.config.token_limit, "tokens_used": self.tokens_used},
            )
        cost_limit = self.config.cost_limit
        if cost_limit is not None and self.cost_usd > cost_limit:
            raise BudgetExceeded(
                "Actual provider usage exceeded cost_limit_usd after the request completed.",
                limit=cost_limit,
                current=self.cost_usd,
                projected=self.cost_usd,
            )
        return actual_cost

    def record_tool_call_started(self, tool_name: str) -> None:
        self.check_time()
        projected = self._tool_calls + 1
        if self.config.tool_call_limit is not None and projected > self.config.tool_call_limit:
            raise ToolCallLimitExceeded(
                "Tool call would exceed tool_call_limit before the tool is invoked.",
                details={
                    "tool": tool_name,
                    "limit": self.config.tool_call_limit,
                    "current_tool_calls": self._tool_calls,
                    "projected_tool_calls": projected,
                },
            )
        self._tool_calls = projected
