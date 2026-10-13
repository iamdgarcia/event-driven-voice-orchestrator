"""Scripted fake streaming model — no API keys, deterministic behavior.

Decisions are keyed by (scenario script) + (what the system prompt says about
pending tools). Good for demos, tests, and benchmarks of the runtime itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from eventorchestrator.domain.models import AssistantDecision, ToolCall
from eventorchestrator.domain.ports import (
    DecisionCompleted,
    ModelPort,
    ModelStreamEvent,
    TextDelta,
)

CHUNK_PACE_S = 0.015  # 15 ms between streamed chunks (audible-ish pacing)


class ScriptedModel(ModelPort):
    """Plays a scripted sequence of decisions, streaming their text.

    Each script entry is either a str (assistant text, streamed) or an
    AssistantDecision (tool calls => dispatched in background; content is
    streamed first). The script loops on its last entry if the conversation
    continues past it.
    """

    def __init__(self, script: Sequence[str | AssistantDecision]) -> None:
        self._script = list(script)
        self._cursor = 0
        self.invocations = 0

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: Sequence,
        tools: Sequence,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.invocations += 1
        step = self._script[min(self._cursor, len(self._script) - 1)]
        self._cursor += 1

        if isinstance(step, AssistantDecision):
            if step.content:
                async for delta in self._stream_text(step.content):
                    yield delta
            yield DecisionCompleted(decision=step)
            return

        async for delta in self._stream_text(step):
            yield delta
        yield DecisionCompleted(decision=AssistantDecision(content=step))

    @staticmethod
    async def _stream_text(text: str, words_per_chunk: int = 2) -> AsyncIterator[ModelStreamEvent]:
        words = text.split(" ")
        for i in range(0, len(words), words_per_chunk):
            yield TextDelta(
                " ".join(words[i : i + words_per_chunk])
                + (" " if i + words_per_chunk < len(words) else "")
            )
            await asyncio.sleep(CHUNK_PACE_S)


def tool_call(call_id: str, name: str, arguments: dict | None = None) -> AssistantDecision:
    """Shorthand: one decision that requests tool calls."""
    return AssistantDecision(
        content="", tool_calls=(ToolCall(call_id=call_id, name=name, arguments=arguments or {}),)
    )
