"""Pipecat bridge: map Pipecat frames <-> the event-driven orchestrator.

The bridge is TRANSPORT ONLY, matching the corporate POC's adapter contract:

  TranscriptionFrame            -> orchestrator.send_text()
  text deltas                   -> LLMTextFrame (downstream)
  start/end of response         -> LLMFullResponse{Start,End}Frame
  Bot{Started,Stopped}Speaking  -> notify_output_{started,completed}()
  InterruptionFrame             -> notify_output_interrupted() (+ orchestrator.interrupt())

Pipecat keeps frame scheduling and the audio graph; the orchestrator keeps the
canonical conversation state and the background tools. The bridge owns neither.

Works with any ModelPort — including ScriptedModel (see adapters/fake_model.py),
so pipelines can run WITHOUT an LLM (demos, tests, latency benches).

Requires: pipecat-ai (optional extra; core library stays zero-dep).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        EndFrame,
        Frame,
        InterruptionFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        StartFrame,
        TranscriptionFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pipecat_adapter requires pipecat-ai. Install with: "
        "pip install pipecat-ai  (the core orchestrator has no dependencies)"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eventorchestrator.application.orchestrator import ConversationOrchestrator  # noqa: E402
from eventorchestrator.domain.ports import (  # noqa: E402
    OutputPort,
)


class BridgeOutput(OutputPort):
    """Adapter that forwards orchestrator output events into Pipecat frames."""

    def __init__(self, bridge: EventDrivenOrchestratorBridge) -> None:
        self._bridge = bridge
        self.responses: dict[str, list[str]] = {}
        self.interrupted: list[str] = []

    async def start_response(self, response_id: str, turn_id: str) -> None:
        self.responses[response_id] = []
        await self._bridge.push_frame(LLMFullResponseStartFrame())

    async def text_delta(self, response_id: str, text: str) -> None:
        self.responses.setdefault(response_id, []).append(text)
        await self._bridge.push_frame(LLMTextFrame(text))

    async def end_response(self, response_id: str) -> None:
        await self._bridge.push_frame(LLMFullResponseEndFrame())

    async def cancel_response(self, response_id: str) -> None:
        self.interrupted.append(response_id)
        # Pipecat's own interruption flow flushes the partial response downstream.

    async def tool_started(self, pending_call) -> None: ...

    async def tool_finished(self, pending_call, result) -> None: ...


class EventDrivenOrchestratorBridge(FrameProcessor):
    """Mid-pipeline processor: Pipecat frames in, orchestrator events applied,
    assistant output pushed back down as LLM frames."""

    def __init__(self, *, orchestrator: ConversationOrchestrator, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._orchestrator = orchestrator
        self._output = BridgeOutput(self)
        self._orchestrator._output = self._output
        self._cancelled: set[str] = set()

    async def start(self) -> None:
        await self._orchestrator.start()

    async def cleanup(self) -> None:
        await self._orchestrator.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self.cleanup()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            await self._orchestrator.send_text(frame.text)
        elif isinstance(frame, BotStartedSpeakingFrame):
            # the currently open response (if any) has started playing
            resp = self._latest_response()
            if resp:
                await self._orchestrator.notify_output_started(resp)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            resp = self._latest_response()
            if resp:
                await self._orchestrator.notify_output_completed(resp)
        elif isinstance(frame, InterruptionFrame):
            resp = self._latest_response()
            self._cancelled.add(resp or "")
            if resp:
                await self._orchestrator.notify_output_interrupted(resp, reason="user_barge_in")
            await self._orchestrator.interrupt()

    def _latest_response(self) -> str | None:
        responses = self._output.responses
        return next(reversed(responses)) if responses else None

    def visible_text(self) -> str:
        return "".join("".join(chunks) for chunks in self._output.responses.values())

    def was_cancelled(self) -> bool:
        return bool(self._output.interrupted)
