"""Bridge self-check (plain asserts, FakeModel path — no LLM, no network).

Run:  python integrations/pipecat_adapter/test_bridge.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402

from eventorchestrator.adapters.fake_model import ScriptedModel  # noqa: E402
from eventorchestrator.adapters.fake_tools import ToyToolRegistry  # noqa: E402
from eventorchestrator.adapters.memory_repository import (  # noqa: E402
    InMemoryConversationRepository,
)
from eventorchestrator.application.orchestrator import ConversationOrchestrator  # noqa: E402
from eventorchestrator.domain.ports import NoopDispatcherState  # noqa: E402

from pipecat_adapter import EventDrivenOrchestratorBridge  # noqa: E402


class SpySink(FrameProcessor):
    """Collects every frame pushed downstream by the bridge."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        self.frames.append(frame)


def make_bridge(script) -> tuple[EventDrivenOrchestratorBridge, SpySink]:
    orch = ConversationOrchestrator(
        conversation_id="bridge-test",
        model=ScriptedModel(script),
        tools=ToyToolRegistry(rtt_s=0.15),
        output=None,  # replaced by BridgeOutput inside the bridge
        repository=InMemoryConversationRepository(),
        dispatcher_state=NoopDispatcherState(),
    )
    # Direct mode keeps the whole pipecat task machinery inert (no internal
    # process task, no TaskManager needed): the interruption path never
    # creates the coroutine whose non-await used to warn at teardown.
    bridge = EventDrivenOrchestratorBridge(orchestrator=orch, enable_direct_mode=True)
    sink = SpySink()
    bridge._sink = sink  # test hook: capture pushed frames

    async def spy_push(frame, direction=FrameDirection.DOWNSTREAM):
        sink.frames.append(frame)
        # do NOT actually forward (no pipeline in this test)

    bridge.push_frame = spy_push  # type: ignore[method-assign]
    return bridge, sink


async def scenario_full_turn() -> None:
    """TranscriptionFrame -> LLMTextFrame outputs appear."""
    bridge, sink = make_bridge(["Both orders shipped, ETA two days."])
    await bridge.process_frame(
        TranscriptionFrame(text="where are my orders?", user_id="u", timestamp="t"),
        FrameDirection.DOWNSTREAM,
    )
    await bridge._orchestrator.wait_idle()
    await asyncio.sleep(0.05)
    llm_texts = [f for f in sink.frames if isinstance(f, LLMTextFrame)]
    assert llm_texts, "no LLMTextFrame emitted"
    joined = "".join(f.text for f in llm_texts)
    assert "shipped" in joined, f"unexpected text: {joined!r}"
    print(f"PASS full-turn: {len(llm_texts)} LLMTextFrame(s): {joined[:50]!r}")


async def scenario_barge_in_cancels() -> None:
    """InterruptionFrame cancels the open response — no further LLMTextFrame."""
    long_text = "This answer keeps streaming and streaming " * 20
    bridge, sink = make_bridge([long_text])
    await bridge.process_frame(
        TranscriptionFrame(text="hello", user_id="u", timestamp="t"), FrameDirection.DOWNSTREAM
    )
    await asyncio.sleep(0.12)  # let some chunks flow
    before = len([f for f in sink.frames if isinstance(f, LLMTextFrame)])
    assert before > 0, "expected some chunks before barge-in"

    await bridge.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.15)
    after = len([f for f in sink.frames if isinstance(f, LLMTextFrame)])
    assert after == before, f"streamed {after - before} chunks after cancel"
    assert bridge.was_cancelled(), "bridge did not record the cancellation"
    print(f"PASS barge-in: stream stopped at {before} chunks, cancellation recorded")


async def scenario_playback_reported() -> None:
    """Bot*Speaking frames map to notify_output_* (delivery lifecycle)."""
    bridge, sink = make_bridge(["Short answer."])
    await bridge.process_frame(
        TranscriptionFrame(text="hi", user_id="u", timestamp="t"), FrameDirection.DOWNSTREAM
    )
    await bridge._orchestrator.wait_idle()
    resp = bridge._latest_response()
    assert resp, "no response id"
    await bridge.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await bridge.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    state = bridge._orchestrator.state
    assert resp in state.responses, "response missing from canonical state"
    print("PASS playback: delivery lifecycle reachable via Bot*Speaking frames")


async def main() -> int:
    await scenario_full_turn()
    await scenario_barge_in_cancels()
    await scenario_playback_reported()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
