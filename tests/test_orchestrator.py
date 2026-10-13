"""Plain-assert tests for the event-driven orchestrator (no pytest needed).

Run:  python -m tests.test_orchestrator   (or python tests/test_orchestrator.py)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eventorchestrator.adapters.fake_model import ScriptedModel, tool_call  # noqa: E402
from eventorchestrator.adapters.fake_tools import ToyToolRegistry  # noqa: E402
from eventorchestrator.adapters.memory_repository import (  # noqa: E402
    InMemoryConversationRepository,
)
from eventorchestrator.adapters.terminal import TerminalOutput  # noqa: E402
from eventorchestrator.application.orchestrator import ConversationOrchestrator  # noqa: E402
from eventorchestrator.domain.ports import NoopDispatcherState  # noqa: E402

FINAL = "Order A-1234 shipped, arrives in 2 days."
FILLER = "One moment."


class RecordingOutput(TerminalOutput):
    """TerminalOutput that also auto-reports playback like a TTS transport would."""

    def __init__(self, orchestrator, **kw):
        super().__init__(**kw)
        self._orchestrator = orchestrator

    async def start_response(self, response_id: str, turn_id: str) -> None:
        await super().start_response(response_id, turn_id)
        # simulate the transport starting playout (delivery lifecycle begins)
        await self._orchestrator.notify_output_started(response_id)

    async def end_response(self, response_id: str) -> None:
        await super().end_response(response_id)
        await self._orchestrator.notify_output_completed(response_id)


def make_orchestrator(script, rtt_s=0.3):
    tools = ToyToolRegistry(rtt_s=rtt_s)
    model = ScriptedModel(script)

    orchestrator = ConversationOrchestrator(
        conversation_id="test",
        model=model,
        tools=tools,
        output=None,  # type: ignore[arg-type]
        repository=InMemoryConversationRepository(),
        dispatcher_state=NoopDispatcherState(),
    )
    output = RecordingOutput(orchestrator)
    orchestrator._output = output  # wire after construction (output needs the ref)
    return orchestrator, output, tools


async def wait_for_text(
    output, needle: str, timeout_s: float = 15.0, since: float | None = None
) -> None:
    loop = asyncio.get_running_loop()
    start = since if since is not None else loop.time()
    deadline = start + timeout_s
    seen_text = ""
    while loop.time() < deadline:
        text = "".join("".join(v) for v in output.chunks.values())
        if len(text) > len(seen_text):
            seen_text = text
        if needle in text:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"text never appeared: {needle!r}; seen so far: {seen_text[-200:]!r}")


async def scenario_user_turn_during_pending() -> None:
    """(1) A user turn while tools are pending is applied + acked immediately."""

    # model behavior: 1st invocation -> tool call; reentry with pending context -> filler;
    # after tools complete -> final answer. The ScriptedModel replays script steps in order,
    # and steps repeat as needed.
    filler = FILLER
    final = FINAL
    script = [
        tool_call("t1", "lookup_order_status", {"order_id": "A-1234"}),
        filler,
        filler,
        final,
    ]
    orch, out, tools = make_orchestrator(script, rtt_s=0.4)
    await orch.start()
    await orch.send_text("where is my order?")
    await asyncio.sleep(0.05)  # tool is now pending (rtt 0.4s)
    assert orch.state.pending_calls, "tool should be pending"
    t_mid = asyncio.get_running_loop().time()
    await orch.send_text("actually also: hi!")  # user turn DURING pending
    # generation restarts immediately with pending-call context and emits the filler
    await wait_for_text(out, FILLER, since=t_mid)
    ack_ms = (asyncio.get_running_loop().time() - t_mid) * 1000
    assert ack_ms < 1000, f"ack took {ack_ms:.0f}ms — user turn stalled"
    roles = [m.role for m in orch.state.messages]
    assert roles.count("user") == 2, "both user turns must be in canonical history"
    await orch.close()
    print(f"PASS [user-turn-during-pending]: applied + acked in {ack_ms:.0f}ms")


async def scenario_batch_atomic_commit() -> None:
    """(2) A tool batch commits atomically: assistant msg + ALL results together."""
    script = [
        # one decision, two tool calls in one batch
        __import__("eventorchestrator.adapters.fake_model", fromlist=["x"]).AssistantDecision(
            content="",
            tool_calls=tuple(
                [
                    __import__("eventorchestrator.domain.models", fromlist=["x"]).ToolCall(
                        call_id="t1", name="lookup_order_status", arguments={"order_id": "A"}
                    ),
                    __import__("eventorchestrator.domain.models", fromlist=["x"]).ToolCall(
                        call_id="t2", name="check_weather", arguments={"city": "Madrid"}
                    ),
                ]
            ),
        ),
        FINAL,
    ]
    orch, out, tools = make_orchestrator(script, rtt_s=0.3)
    await orch.start()
    await orch.send_text("go")
    await wait_for_text(out, FINAL)
    await orch.close()

    roles = [m.role for m in orch.state.messages]
    # provider-valid history: the assistant tool-call message appears exactly once,
    # followed by BOTH tool result messages — no unresolved sequence ever committed.
    assert roles.count("assistant") == 1
    assert roles.count("tool") == 2
    assistant_idx = roles.index("assistant")
    assert roles[assistant_idx + 1] == "tool" and roles[assistant_idx + 2] == "tool"
    assert not orch.state.pending_batches, "batch must be committed"
    assert not orch.state.active_calls(), "no active calls after commit"
    assert set(tools.executed) == {"lookup_order_status", "check_weather"}
    print("PASS [atomic-batch-commit]: assistant+2 tool results committed together")


async def scenario_interrupt_generation_tools_survive() -> None:
    """(3) interrupt() cancels generation; delivery INTERRUPTED; tools keep running."""
    from eventorchestrator.domain.models import (
        ResponseDeliveryStatus,
        ResponseGenerationStatus,
    )

    slow_answer = "This is a long answer that will be cancelled midway for the demo."
    script = [
        tool_call("t1", "lookup_order_status", {"order_id": "A-1234"}),
        slow_answer,
        "Recovered after interruption.",
    ]
    orch, out, tools = make_orchestrator(script, rtt_s=0.05)
    await orch.start()
    await orch.send_text("start")
    # poll until the final answer is actually STREAMING, then barge in
    deadline = asyncio.get_running_loop().time() + 10
    while asyncio.get_running_loop().time() < deadline:
        resp = orch.state.latest_response()
        if resp and resp.content.startswith("This is a long"):
            break
        await asyncio.sleep(0.005)
    await orch.interrupt()
    await asyncio.sleep(0.02)

    latest = orch.state.latest_response()
    assert latest is not None
    assert latest.generation_status == ResponseGenerationStatus.CANCELLED
    assert latest.delivery_status == ResponseDeliveryStatus.INTERRUPTED
    assert latest.interruption_reason == "generation_cancelled"

    # the tool (dispatched earlier, completed during generation) still ran;
    # a NEXT turn re-enters the model and the interrupted-delivery note is injected
    await orch.send_text("continue please")
    await wait_for_text(out, "Recovered after interruption.")
    await orch.close()
    assert tools.executed == ["lookup_order_status"], "tool must survive the barge-in"
    print(
        "PASS [interrupt-generation-only]: gen CANCELLED, delivery INTERRUPTED, tool completed, next turn recovers"
    )


async def scenario_no_prefix_dedup() -> None:
    """(4) The clean-room core removed the corporate POC's prefix-match dedup.

    Two answers that legitimately start with the same words are BOTH delivered
    in full. (The corporate POC suppressed 'duplicate prefixes' to de-glitch a
    double-emission artifact; that heuristic also swallowed legitimate
    same-opening answers, so the clean core omits it — event loop emission is
    the single source of visible text.)
    """
    script = ["Yes, I can help with that.", "Yes, I can help with that too — and more."]
    orch, out, _tools = make_orchestrator(script, rtt_s=0.01)
    await orch.start()
    await orch.send_text("help me")
    await asyncio.sleep(0.3)  # let turn 1 fully deliver
    await orch.send_text("and again")
    await wait_for_text(out, "too — and more")
    await orch.close()
    text = "".join("".join(v) for v in out.chunks.values())
    # 'Yes, I can help with that.' appears twice: once standalone, once as a
    # PREFIX of the longer answer — the old prefix-dedup would have swallowed
    # the second one entirely.
    assert "Yes, I can help with that." in text
    assert "Yes, I can help with that too — and more." in text, (
        f"the longer answer (superset prefix) must be delivered in full; got: {text!r}"
    )
    print("PASS [no-prefix-dedup]: identical openings both delivered in full")


async def main() -> None:
    await scenario_user_turn_during_pending()
    await scenario_batch_atomic_commit()
    await scenario_interrupt_generation_tools_survive()
    await scenario_no_prefix_dedup()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
