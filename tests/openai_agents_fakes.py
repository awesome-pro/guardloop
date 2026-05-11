# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# The OpenAI Agents SDK's Model interface and Responses-API item types are
# loosely parameterised at this boundary; strict "unknown type" checks are
# relaxed for these test fakes.
"""OpenAI Agents SDK fakes for the ``guardloop.adapters.openai_agents`` tests.

Imported only by ``tests/test_openai_agents_adapter.py`` (which skips when the
``openai-agents`` extra is not installed), so the rest of the test suite never
touches ``agents``.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, Model, ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

try:  # function_tool lives at the package root in current SDK versions
    from agents import function_tool
except ImportError:  # pragma: no cover - older layout
    from agents.tool import function_tool

DEFAULT_MODEL_NAME = "gpt-5.2"  # priced in GuardLoop's default catalog
UNPRICED_MODEL_NAME = "guardloop-test-unpriced-model"


def text_message(text: str, *, message_id: str = "msg_fake") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=message_id,
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def function_call(
    name: str, *, arguments: str = "{}", call_id: str | None = None
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        name=name,
        arguments=arguments,
        call_id=call_id or f"call_{name}",
        id=f"fc_{name}",
    )


def model_response(
    *output_items: Any,
    input_tokens: int = 5,
    output_tokens: int = 3,
) -> ModelResponse:
    return ModelResponse(
        output=list(output_items),
        usage=Usage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        response_id="resp_fake",
        request_id=None,
    )


def text_response(text: str, *, input_tokens: int = 5, output_tokens: int = 3) -> ModelResponse:
    return model_response(
        text_message(text), input_tokens=input_tokens, output_tokens=output_tokens
    )


def tool_response(
    name: str, *, arguments: str = "{}", input_tokens: int = 5, output_tokens: int = 3
) -> ModelResponse:
    return model_response(
        function_call(name, arguments=arguments),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class ScriptedModel(Model):
    """Replays a fixed list of ``ModelResponse``s (repeating the last once exhausted).

    Exposes a ``model`` attribute (default a priced name) so the adapter resolves
    pricing without extra configuration, mirroring how real SDK model objects
    (``OpenAIResponsesModel``) carry their model name.
    """

    def __init__(
        self, scripted: list[ModelResponse] | None = None, *, name: str = DEFAULT_MODEL_NAME
    ) -> None:
        self.model = name
        self.scripted: list[ModelResponse] = list(scripted or [])
        self.call_count = 0
        self.seen_inputs: list[Any] = []
        self.seen_system_prompts: list[str | None] = []

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        self.seen_system_prompts.append(system_instructions)
        self.seen_inputs.append(input)
        index = min(self.call_count, len(self.scripted) - 1) if self.scripted else 0
        self.call_count += 1
        if not self.scripted:
            return text_response("ok")
        return self.scripted[index]

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ScriptedModel does not support streaming")


class RaisingModel(Model):
    """A model whose ``get_response`` always raises (drives the model-error path)."""

    model = DEFAULT_MODEL_NAME

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise RuntimeError("model exploded")

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class NoModelNameModel(Model):
    """A custom model object that does not expose its name (drives the clear-error path)."""

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return text_response("never reached")

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@function_tool
def echo_tool(text: str) -> str:
    """Echo the given text back."""

    return f"echo: {text}"


@function_tool
def exploding_tool(text: str) -> str:
    """A tool that always raises (the SDK turns this into an error string by default)."""

    raise RuntimeError("upstream tool exploded")


def single_agent(
    model: Model, *, name: str = "scripted-agent", instructions: str = "Be helpful."
) -> Agent[Any]:
    return Agent(name=name, model=model, instructions=instructions)


def tool_agent(
    model: Model,
    tools: list[Any],
    *,
    name: str = "tool-agent",
    instructions: str = "Use the tools to help the user.",
) -> Agent[Any]:
    return Agent(name=name, model=model, instructions=instructions, tools=tools)
