from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentruntime import AgentRuntime, BudgetConfig, RunContext, TelemetryConfig
from tests.fakes import FakeOpenAIClient


async def test_runtime_emits_agent_llm_and_tool_spans() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("agentruntime-tests")

    runtime = AgentRuntime(
        budget=BudgetConfig(cost_limit_usd="1.00", tool_call_limit=5),
        telemetry=TelemetryConfig(enabled=True),
        openai_client=FakeOpenAIClient(),
        tracer=tracer,
    )

    async def agent(ctx: RunContext) -> str:
        await ctx.openai.responses.create(
            model="gpt-5.2",
            input="hello",
            max_output_tokens=100,
        )
        return str(await ctx.call_tool("formatter", lambda: "formatted"))

    result = await runtime.run(agent)
    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}

    assert result.success is True
    assert "agent_run" in names
    assert "llm_call openai.responses.create" in names
    assert "tool_call formatter" in names

    llm_span = next(span for span in spans if span.name == "llm_call openai.responses.create")
    llm_attributes = llm_span.attributes
    assert llm_attributes is not None
    assert llm_attributes["gen_ai.system"] == "openai"
    assert llm_attributes["gen_ai.request.model"] == "gpt-5.2"
    assert llm_attributes["gen_ai.usage.input_tokens"] == 100
    assert llm_attributes["gen_ai.usage.output_tokens"] == 50

    tool_span = next(span for span in spans if span.name == "tool_call formatter")
    tool_attributes = tool_span.attributes
    assert tool_attributes is not None
    assert tool_attributes["agentruntime.tool.name"] == "formatter"
    assert tool_attributes["agentruntime.tool.calls_used"] == 1
