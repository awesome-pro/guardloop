"""Framework adapters for running third-party agents under GuardLoop.

Each adapter lives in its own submodule and requires its own optional extra;
importing a submodule is what pulls the framework dependency in. The core
``guardloop`` package never imports an adapter, so ``pip install guardloop``
stays dependency-light.

LangGraph::

    pip install "guardloop[langgraph]"
    from guardloop.adapters.langgraph import guarded_graph
"""

from __future__ import annotations
