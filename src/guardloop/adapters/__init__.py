"""Framework adapters for running third-party agents under GuardLoop.

Each adapter lives in its own submodule and requires its own optional extra;
importing a submodule is what pulls the framework dependency in. The core
``guardloop`` package never imports an adapter, so ``pip install guardloop``
stays dependency-light.

LangGraph::

    pip install "guardloop[langgraph]"
    from guardloop.adapters.langgraph import guarded_graph

OpenAI Agents SDK::

    pip install "guardloop[openai-agents]"
    from guardloop.adapters.openai_agents import guarded_runner
"""

from __future__ import annotations
