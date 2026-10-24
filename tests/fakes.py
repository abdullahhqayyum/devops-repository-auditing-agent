"""Deterministic stand-in for an LLM, used to test the controller without network access."""

from __future__ import annotations

import itertools
from collections.abc import Callable

from llm_client import LLMResponse, ToolCall

_ids = itertools.count(1)


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(f"call_{next(_ids)}", name, arguments)


class ScriptedLLM:
    """Replays a list of turns. A turn is a list of ToolCalls, or a callable that receives
    the message history and returns a list of ToolCalls (to react to tool results)."""

    model = "scripted-test-model"

    def __init__(self, turns: list[list[ToolCall] | Callable[[list[dict]], list[ToolCall]]]):
        self.turns = list(turns)
        self.seen_messages: list[list[dict]] = []

    def chat(self, system, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        if not self.turns:
            return LLMResponse("nothing more to do", [])
        turn = self.turns.pop(0)
        calls = turn(messages) if callable(turn) else turn
        return LLMResponse("reasoning about next step", calls)
