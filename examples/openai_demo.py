"""Demo with a REAL model via the OpenAI SDK — same event loop, no script.

Setup:
  cp .env.example .env          # fill OPENAI_API_KEY (and optionally OPENAI_MODEL)
  pip install openai            # optional extra; core stays zero-dependency
  python -m examples.openai_demo

What to watch (the same three behaviors as the scripted demo, now real):
  1. Ask something that triggers a tool ("what's the status of order A-1234?")
     — the call dispatches in the BACKGROUND while the model keeps streaming.
  2. While the tool is pending (~1.5s), type a NEW message — it is applied as
     an event and answered without waiting for the tool (non-stalling turns).
  3. Interrupt mid-answer (type during streaming) — generation cancels, the
     interrupted delivery is injected as heard/not-heard context next turn.

Type 'quit' to exit. Requires Python >= 3.11. No .env parsing dependency:
the loader below is 10 lines of stdlib.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import contextlib  # noqa: E402  (after sys.path boot)

from eventorchestrator.adapters.memory_repository import InMemoryConversationRepository  # noqa: E402
from eventorchestrator.adapters.terminal import TerminalOutput  # noqa: E402
from eventorchestrator.application.orchestrator import ConversationOrchestrator  # noqa: E402
from eventorchestrator.domain.models import (  # noqa: E402
    AssistantDecision,
    ConversationMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
)
from eventorchestrator.domain.ports import (  # noqa: E402
    DecisionCompleted,
    ModelStreamEvent,
    NoopDispatcherState,
    TextDelta,
)


# ---------------------------------------------------------------- env (stdlib)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=VALUE lines into os.environ; existing env wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------- model port


class OpenAIChatModel:
    """ModelPort on OpenAI chat.completions streaming, tool calls included."""

    def __init__(self, model: str) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
        self.invocations = 0

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
    ) -> AsyncIterator[ModelStreamEvent]:
        self.invocations += 1
        provider_msgs = [{"role": "system", "content": system_prompt}]
        for m in messages:
            entry: dict = {"role": m.role, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.call_id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
            if m.role == "tool":
                entry["tool_call_id"] = m.tool_call_id
            provider_msgs.append(entry)

        kwargs: dict = {"model": self.model, "messages": provider_msgs, "stream": True}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": d.name,
                        "description": d.description,
                        "parameters": d.parameters,
                    },
                }
                for d in tools
            ]

        response = await self.client.chat.completions.create(**kwargs)
        content_parts: list[str] = []
        slots: dict[int, dict] = {}
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
                yield TextDelta(delta.content)
            for tc in delta.tool_calls or []:
                slot = slots.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    slot["name"] = tc.function.name or slot["name"]
                    slot["arguments"] += tc.function.arguments or ""
        content = "".join(content_parts)
        calls = []
        for i, s in sorted(slots.items()):
            try:
                args = json.loads(s["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(call_id=s["id"] or f"call_{i}", name=s["name"], arguments=args))
        yield DecisionCompleted(
            decision=AssistantDecision(content=content, tool_calls=tuple(calls))
        )


# ---------------------------------------------------------------- tools port


BASE_PROMPT = (
    "You are a helpful service agent. Use the provided tools to look things up; "
    "never invent order or weather facts. Answer briefly."
)


class DemoTools:
    """ToolRegistryPort with two tools and REAL simulated latency (asyncio.sleep)."""

    def __init__(self, rtt_s: float) -> None:
        self.rtt_s = rtt_s
        self._defs = (
            ToolDefinition(
                name="lookup_order_status",
                description="Look up the shipping status of one order by id.",
                parameters={
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            ),
            ToolDefinition(
                name="get_weather",
                description="Get today's weather forecast for one city.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        )

    def active_tools(self) -> list[ToolDefinition]:
        return list(self._defs)

    def system_prompt(self) -> str:
        return BASE_PROMPT

    def definition(self, tool_name: str) -> ToolDefinition:
        return next(d for d in self._defs if d.name == tool_name)

    async def execute(self, call: ToolCall) -> ToolExecutionResult:
        await asyncio.sleep(self.rtt_s)  # the real-world latency, simulated
        if call.name == "lookup_order_status":
            content = {
                "order_id": call.arguments.get("order_id"),
                "status": "shipped",
                "eta": "2026-10-16",
            }
        elif call.name == "get_weather":
            content = {"city": call.arguments.get("city"), "forecast": "cloudy, 18°C"}
        else:
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                content=None,
                error=f"unknown tool {call.name}",
            )
        return ToolExecutionResult(call_id=call.call_id, name=call.name, content=content)


# ---------------------------------------------------------------- wiring


async def _playback_pump(orchestrator: ConversationOrchestrator) -> None:
    """Simulate TTS playout: each started response becomes PLAYING -> COMPLETED."""
    seen: set[str] = set()
    while True:
        for response in orchestrator.state.responses.values():
            if response.response_id in seen:
                continue
            if response.delivery_status.value == "pending":
                seen.add(response.response_id)
                await orchestrator.notify_output_started(response.response_id)
                await orchestrator.notify_output_completed(response.response_id)
        await asyncio.sleep(0.01)


async def main() -> None:
    load_dotenv()
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    rtt_s = float(os.environ.get("DEMO_TOOL_RTT_S", "1.5"))
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    output = TerminalOutput(quiet=False)
    orchestrator = ConversationOrchestrator(
        conversation_id=uuid.uuid4().hex,
        model=OpenAIChatModel(model_name),
        tools=DemoTools(rtt_s=rtt_s),
        output=output,
        repository=InMemoryConversationRepository(),
        dispatcher_state=NoopDispatcherState(),
    )
    await orchestrator.start()
    pump = asyncio.create_task(_playback_pump(orchestrator))
    print(f"model={model_name} · tool RTT≈{rtt_s}s — type 'quit' to exit\n")

    try:
        while True:
            text = await asyncio.to_thread(input, "you> ")
            text = text.strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                break
            await orchestrator.send_text(text)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump
        await orchestrator.wait_idle()
        await orchestrator.close()
        print("\n(done)")


if __name__ == "__main__":
    asyncio.run(main())
