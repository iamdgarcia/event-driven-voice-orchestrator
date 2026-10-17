"""LiveKit Agents bridge: roomless AgentSession behind the 5-method BenchHarness.

Transport-mapping summary (mirrors the pipecat adapter's role in this repo):

    generate_reply(user_input=...)   <- harness.send_user_text(t)
    TextOutput sink (capture_text)   -> harness-visible assistant text
    session.interrupt(force=True)    <- harness.interrupt()
    _tool_executor._running_tasks    -> harness.pending_count()

LiveKit keeps frame scheduling, speech playout and the tool executor; the
conformance runner measures the same 4 cells it runs on the reference runtime.
Requires livekit-agents==1.8.8 (pin: first release shipping TypeSafe Jev
decision-models in llm.decisions).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

try:
    from livekit.agents import voice
    from livekit.agents.llm import ChatChunk, ChoiceDelta, FunctionToolCall, LLM, LLMStream
    from livekit.agents.llm.tool_context import function_tool
    from livekit.agents.testing import fake_job_context
    from livekit.agents.voice.io import TextOutput
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "livekit_adapter requires livekit-agents. Install with: "
        "pip install livekit-agents==1.8.8  (the core orchestrator has no dependencies)"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


class ScriptedLLM(LLM):
    """Canned script played as an LLM plugin: str = streamed text, list = tool calls.

    Script entries are consumed one per chat() call and loop on the last entry,
    mirroring adapters/fake_model.py ScriptedModel semantics.
    """

    def __init__(self, script: list) -> None:
        super().__init__()
        self._script = list(script)
        self._cursor = 0
        self.invocations = 0

    @property
    def provider(self) -> str:
        return "scripted"

    @property
    def model(self) -> str:
        return "livekit-conformance"

    def chat(  # noqa: PLR0913 (provider signature)
        self,
        *,
        chat_ctx: Any,
        tools: list | None = None,
        conn_options: Any = None,
        parallel_tool_calls: Any = None,
        tool_choice: Any = None,
        extra_kwargs: Any = None,
    ) -> LLMStream:
        from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

        return ScriptedStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )

    def next_step(self):
        step = self._script[min(self._cursor, len(self._script) - 1)]
        self._cursor += 1
        self.invocations += 1
        return step


class ScriptedStream(LLMStream):
    """Streams the script step: text in word chunks (paced), tool calls in one chunk."""

    def __init__(self, scripted_llm: ScriptedLLM, **kwargs: Any) -> None:
        super().__init__(scripted_llm, **kwargs)
        self._scripted = scripted_llm
        self.chunk_pace_s = 0.015

    async def _run(self) -> None:
        step = self._scripted.next_step()
        if isinstance(step, str):
            words = step.split(" ")
            for i in range(0, len(words), 2):
                text = " ".join(words[i : i + 2]) + (" " if i + 2 < len(words) else "")
                await asyncio.sleep(self.chunk_pace_s)
                self._event_ch.send_nowait(ChatChunk(id="chunk-0", delta=ChoiceDelta(content=text)))
        else:
            calls = [
                FunctionToolCall(
                    type="function", name=name, arguments=json.dumps(args or {}), call_id=cid
                )
                for cid, name, args in step
            ]
            self._event_ch.send_nowait(ChatChunk(id="chunk-0", delta=ChoiceDelta(tool_calls=calls)))


class TextSink(TextOutput):
    """Collects assistant text segments (what the user would hear)."""

    def __init__(self) -> None:
        super().__init__(label="livekit-conformance-sink", next_in_chain=None)
        self.segments: list[str] = []
        self._current: list[str] = []
        self.flushed = 0

    async def capture_text(self, text: str) -> None:
        self._current.append(text)

    def flush(self) -> None:
        if self._current:
            self.segments.append("".join(self._current))
            self._current = []
        self.flushed += 1

    def all_text(self) -> str:
        return "".join(self.segments)


def make_lookup_tool(rtt_s: float = 0.05):
    @function_tool
    async def lookup_order_status(order_id: str) -> dict:
        """Look up the fulfillment status of one order id.

        Args:
            order_id: the order identifier
        """
        await asyncio.sleep(rtt_s)
        return {"order_id": order_id, "status": "shipped", "eta_days": 2}

    return lookup_order_status


def make_script() -> list:
    """Script covering the 4 cells: single tool, parallel batch, ack, long story.

    Cell 3 (mid-pending ack) needs tools pending when 'hello, still there?' is
    sent: in the reference adapter cell-2 leftovers fill that role (racy); here
    the ack turn's own script entry (idx 4) IS a slow tool call, so the cell
    measures the invariant deterministically.
    """
    return [
        [("c1", "lookup_order_status", {"order_id": "A-1234"})],
        "Order A-1234 shipped; ETA 2026-10-16.",
        [
            ("b1", "lookup_order_status", {"order_id": "A-1"}),
            ("b2", "lookup_order_status", {"order_id": "A-2"}),
            ("b3", "lookup_order_status", {"order_id": "A-3"}),
            ("b4", "lookup_order_status", {"order_id": "A-4"}),
        ],
        "All four orders shipped.",
        # cell 3 (mid-pending ack): TWO slow tools so pending_count() >= 2 fires
        # deterministically while the ack turn queues behind them (verified: ack text
        # delivered in ~16ms with both tools running). One tool would peak at 1.
        [
            ("w1", "lookup_order_status", {"order_id": "A-W1"}),
            ("w2", "lookup_order_status", {"order_id": "A-W2"}),
        ],
        "Yes, I'm here.",
        "Once upon a time there was a voice agent that never stalled a user turn, "
        "and every caller lived happily ever after — the end of the long story.",
    ]


class LiveKitHarness:
    """BenchHarness over a roomless AgentSession (the 5-method protocol)."""

    def __init__(self, session: voice.AgentSession, sink: TextSink) -> None:
        self.session = session
        self.sink = sink
        self.interrupted_count = 0

    async def start(self) -> None:
        agent = voice.Agent(instructions="You are a helpful order assistant.")
        # fake_job_context: worker-less JobContext. Omit `room` (passing room=None would
        # be treated as GIVEN and spin up a RoomIO against a null room — verified crash),
        # so the session runs purely on the text path with no WebRTC at all.
        with fake_job_context():
            await self.session.start(agent=agent, record=False)

    async def send_user_text(self, t: str) -> None:
        self.session.generate_reply(user_input=t, input_modality="text")

    async def interrupt(self) -> None:
        handle = self.session.current_speech
        await self.session.interrupt(force=True)
        if handle is not None:
            await handle

    async def wait_idle(self) -> None:
        # session-level: every queued speech finished (generation + playout +
        # tool steps). 'thinking' covers the window after tools complete but
        # before the reply-chain speech is scheduled — without it wait_idle
        # returns mid-turn (verified: cell-1 measured 46ms < 1 tool RTT).
        activity = self.session._activity
        while True:
            current = self.session.current_speech
            busy = (current is not None and not current.done()) or self.session.agent_state in (
                "speaking",
                "thinking",
            )
            queued = bool(getattr(activity, "_speech_q", None)) if activity else False
            if not busy and not queued:
                return
            await asyncio.sleep(0.005)

    def pending_count(self) -> int:
        activity = self.session._activity
        if activity is None:
            return 0
        return len(activity._tool_executor._running_tasks)
