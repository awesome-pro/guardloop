from __future__ import annotations

from decimal import Decimal

import pytest

from guardloop.budget import BudgetController
from guardloop.exceptions import BudgetExceeded, TokenLimitExceeded, ToolCallLimitExceeded
from guardloop.models import BudgetConfig
from guardloop.pricing import ModelPricing, PricingCatalog


def _catalog() -> PricingCatalog:
    return PricingCatalog(
        [
            ModelPricing(
                provider="test",
                model="expensive",
                input_cost_per_million_tokens=Decimal("100.00"),
                output_cost_per_million_tokens=Decimal("100.00"),
            ),
            ModelPricing(
                provider="test",
                model="precise",
                input_cost_per_million_tokens=Decimal("0.10"),
                output_cost_per_million_tokens=Decimal("0.20"),
            ),
        ],
        include_defaults=False,
    )


def test_blocks_llm_call_before_cost_cap_is_exceeded() -> None:
    budget = BudgetController(BudgetConfig(cost_limit_usd="0.01"), _catalog())

    with pytest.raises(BudgetExceeded):
        budget.check_llm_call(
            provider="test",
            model="expensive",
            estimated_input_tokens=100,
            reserved_output_tokens=100,
        )


def test_blocks_llm_call_before_token_cap_is_exceeded() -> None:
    budget = BudgetController(BudgetConfig(token_limit=99), _catalog())

    with pytest.raises(TokenLimitExceeded):
        budget.check_llm_call(
            provider="test",
            model="expensive",
            estimated_input_tokens=50,
            reserved_output_tokens=50,
        )


def test_blocks_tool_call_before_tool_limit_is_exceeded() -> None:
    budget = BudgetController(BudgetConfig(tool_call_limit=1), _catalog())

    budget.record_tool_call_started("search")
    with pytest.raises(ToolCallLimitExceeded):
        budget.record_tool_call_started("search")


def test_decimal_cost_accounting_has_no_float_drift() -> None:
    budget = BudgetController(BudgetConfig(cost_limit_usd="1.00"), _catalog())

    budget.record_llm_call(
        provider="test",
        model="precise",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert budget.cost_usd == Decimal("0.30")
    assert isinstance(budget.cost_usd, Decimal)
