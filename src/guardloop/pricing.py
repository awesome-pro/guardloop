"""Provider/model pricing catalog.

Prices are USD per one million tokens and are intentionally overrideable because
provider pricing changes over time.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator

from guardloop.exceptions import ModelPricingMissing

MILLION = Decimal("1000000")


class ModelPricing(BaseModel):
    """Token pricing for one provider/model pair."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    input_cost_per_million_tokens: Decimal
    output_cost_per_million_tokens: Decimal

    @field_validator(
        "input_cost_per_million_tokens",
        "output_cost_per_million_tokens",
        mode="before",
    )
    @classmethod
    def _parse_decimal(cls, value: object) -> object:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    @field_validator("provider", "model")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return value.strip().lower()

    def estimate_cost(self, *, input_tokens: int, output_tokens: int) -> Decimal:
        input_cost = Decimal(input_tokens) * self.input_cost_per_million_tokens / MILLION
        output_cost = Decimal(output_tokens) * self.output_cost_per_million_tokens / MILLION
        return input_cost + output_cost


def _price(provider: str, model: str, input_price: str, output_price: str) -> ModelPricing:
    return ModelPricing(
        provider=provider,
        model=model,
        input_cost_per_million_tokens=Decimal(input_price),
        output_cost_per_million_tokens=Decimal(output_price),
    )


DEFAULT_MODEL_PRICES: tuple[ModelPricing, ...] = (
    # OpenAI API pricing checked May 3, 2026.
    _price("openai", "gpt-5.5", "5.00", "30.00"),
    _price("openai", "gpt-5.4", "2.50", "15.00"),
    _price("openai", "gpt-5.4-mini", "0.75", "4.50"),
    _price("openai", "gpt-5.2", "1.75", "14.00"),
    _price("openai", "gpt-5.2-2025-12-11", "1.75", "14.00"),
    _price("openai", "gpt-5.2-chat-latest", "1.75", "14.00"),
    _price("openai", "gpt-5.2-codex", "1.75", "14.00"),
    _price("openai", "gpt-5.2-pro", "21.00", "168.00"),
    _price("openai", "gpt-5.1", "1.25", "10.00"),
    _price("openai", "gpt-5", "1.25", "10.00"),
    _price("openai", "gpt-5-mini", "0.25", "2.00"),
    _price("openai", "gpt-5-nano", "0.05", "0.40"),
    _price("openai", "gpt-4.1", "2.00", "8.00"),
    _price("openai", "gpt-4.1-mini", "0.40", "1.60"),
    _price("openai", "gpt-4.1-nano", "0.10", "0.40"),
    _price("openai", "gpt-4o", "2.50", "10.00"),
    _price("openai", "gpt-4o-mini", "0.15", "0.60"),
    # Anthropic Claude pricing checked May 3, 2026.
    _price("anthropic", "claude-opus-4-1-20250805", "15.00", "75.00"),
    _price("anthropic", "claude-opus-4-1", "15.00", "75.00"),
    _price("anthropic", "claude-opus-4-20250514", "15.00", "75.00"),
    _price("anthropic", "claude-opus-4-0", "15.00", "75.00"),
    _price("anthropic", "claude-sonnet-4-20250514", "3.00", "15.00"),
    _price("anthropic", "claude-sonnet-4-0", "3.00", "15.00"),
    _price("anthropic", "claude-3-7-sonnet-20250219", "3.00", "15.00"),
    _price("anthropic", "claude-3-7-sonnet-latest", "3.00", "15.00"),
    _price("anthropic", "claude-3-5-haiku-20241022", "0.80", "4.00"),
    _price("anthropic", "claude-3-5-haiku-latest", "0.80", "4.00"),
    _price("anthropic", "claude-3-haiku-20240307", "0.25", "1.25"),
)


class PricingCatalog:
    """Lookup table for model pricing with user-provided overrides."""

    def __init__(
        self,
        prices: Iterable[ModelPricing] | None = None,
        *,
        include_defaults: bool = True,
    ) -> None:
        entries = list(DEFAULT_MODEL_PRICES if include_defaults else ())
        if prices is not None:
            entries.extend(prices)
        self._prices = {(entry.provider, entry.model): entry for entry in entries}

    def get(self, provider: str, model: str) -> ModelPricing:
        key = (provider.strip().lower(), model.strip().lower())
        if key not in self._prices:
            raise ModelPricingMissing(
                f"No pricing is configured for provider={provider!r}, model={model!r}. "
                "Pass a custom ModelPricing entry to GuardLoop(pricing=[...]).",
                details={"provider": provider, "model": model},
            )
        return self._prices[key]
