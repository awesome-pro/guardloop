"""Fake provider clients for tests and examples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FakeUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(slots=True)
class FakeResponse:
    output_text: str
    usage: FakeUsage


class FakeOpenAIResponses:
    def __init__(
        self,
        *,
        input_tokens: int = 100,
        output_tokens: int = 50,
        output_text: str = "ok",
    ) -> None:
        self.calls = 0
        self.kwargs_seen: list[dict[str, object]] = []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.output_text = output_text

    async def create(self, **kwargs: object) -> FakeResponse:
        self.calls += 1
        self.kwargs_seen.append(kwargs)
        return FakeResponse(
            output_text=self.output_text,
            usage=FakeUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )


class FakeOpenAIClient:
    def __init__(self, responses: FakeOpenAIResponses | None = None) -> None:
        self.responses = responses or FakeOpenAIResponses()


@dataclass(slots=True)
class FakeContentBlock:
    text: str


@dataclass(slots=True)
class FakeAnthropicResponse:
    content: list[FakeContentBlock]
    usage: FakeUsage


class FakeAnthropicMessages:
    def __init__(
        self,
        *,
        input_tokens: int = 100,
        output_tokens: int = 50,
        output_text: str = "ok",
    ) -> None:
        self.calls = 0
        self.kwargs_seen: list[dict[str, object]] = []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.output_text = output_text

    async def create(self, **kwargs: object) -> FakeAnthropicResponse:
        self.calls += 1
        self.kwargs_seen.append(kwargs)
        return FakeAnthropicResponse(
            content=[FakeContentBlock(self.output_text)],
            usage=FakeUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )


class FakeAnthropicClient:
    def __init__(self, messages: FakeAnthropicMessages | None = None) -> None:
        self.messages = messages or FakeAnthropicMessages()
