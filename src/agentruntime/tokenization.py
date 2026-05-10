"""Token estimation helpers used before provider calls."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import cast

import tiktoken


def payload_to_text(payload: object) -> str:
    """Flatten common LLM input payloads into text for token estimation."""

    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        values = cast(Mapping[object, object], payload).values()
        return " ".join(payload_to_text(value) for value in values)
    if isinstance(payload, Sequence) and not isinstance(payload, bytes | bytearray | str):
        items = cast(Sequence[object], payload)
        return " ".join(payload_to_text(item) for item in items)
    try:
        return json.dumps(payload, default=str, sort_keys=True)
    except TypeError:
        return str(payload)


def estimate_openai_tokens(model: str, payload: object) -> int:
    text = payload_to_text(payload)
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def estimate_anthropic_tokens(payload: object) -> int:
    """Conservative local estimate when Anthropic token counting is not used."""

    text = payload_to_text(payload)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
