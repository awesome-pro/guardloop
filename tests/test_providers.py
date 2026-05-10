from __future__ import annotations

import pytest

from agentruntime.budget import BudgetController
from agentruntime.exceptions import ModelPricingMissing, TokenLimitMissing
from agentruntime.models import BudgetConfig, TelemetryConfig
from agentruntime.pricing import PricingCatalog
from agentruntime.providers.anthropic import WrappedAnthropicClient
from agentruntime.providers.openai import WrappedOpenAIClient
from agentruntime.telemetry.tracer import Telemetry
from tests.fakes import FakeAnthropicClient, FakeOpenAIClient


def _budget() -> BudgetController:
    return BudgetController(BudgetConfig(cost_limit_usd="1.00"), PricingCatalog())


def _telemetry() -> Telemetry:
    return Telemetry(TelemetryConfig(enabled=False))


async def test_openai_wrapper_records_sdk_usage_tokens() -> None:
    budget = _budget()
    client = WrappedOpenAIClient(FakeOpenAIClient(), budget, _telemetry())

    await client.responses.create(model="gpt-5.2", input="hello", max_output_tokens=100)

    assert budget.input_tokens == 100
    assert budget.output_tokens == 50
    assert budget.cost_usd > 0


async def test_anthropic_wrapper_records_sdk_usage_tokens() -> None:
    budget = _budget()
    client = WrappedAnthropicClient(FakeAnthropicClient(), budget, _telemetry())

    await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert budget.input_tokens == 100
    assert budget.output_tokens == 50
    assert budget.cost_usd > 0


async def test_openai_wrapper_requires_max_output_tokens() -> None:
    client = WrappedOpenAIClient(FakeOpenAIClient(), _budget(), _telemetry())

    with pytest.raises(TokenLimitMissing):
        await client.responses.create(model="gpt-5.2", input="hello")


async def test_anthropic_wrapper_requires_max_tokens() -> None:
    client = WrappedAnthropicClient(FakeAnthropicClient(), _budget(), _telemetry())

    with pytest.raises(TokenLimitMissing):
        await client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
        )


async def test_unknown_model_pricing_is_rejected() -> None:
    client = WrappedOpenAIClient(FakeOpenAIClient(), _budget(), _telemetry())

    with pytest.raises(ModelPricingMissing):
        await client.responses.create(
            model="unknown-model",
            input="hello",
            max_output_tokens=100,
        )
