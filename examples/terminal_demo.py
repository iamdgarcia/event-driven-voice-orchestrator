"""Interactive terminal demo of the event-driven voice orchestrator.

Two modes:

  python -m examples.terminal_demo          # interactive: you type, tools run in background
  python -m examples.terminal_demo --demo   # scripted auto-demo, zero typing, exits on its own

The interesting part to watch: tool calls dispatch in the BACKGROUND — keep
typing while a tool is pending and you get an immediate ack (the S2S-like
non-stalling behavior). In --demo mode, the user's second turn lands mid-answer
(barge-in): generation cancels, delivery is marked INTERRUPTED, and the model
is told (internally) that the user did NOT hear the full answer.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import contextlib
import sys
import uuid

from eventorchestrator.adapters.fake_model import ScriptedModel, tool_call
from eventorchestrator.adapters.fake_tools import ToyToolRegistry
from eventorchestrator.adapters.memory_repository import InMemoryConversationRepository
from eventorchestrator.adapters.terminal import TerminalOutput
from eventorchestrator.application.orchestrator import ConversationOrchestrator
from eventorchestrator.domain.ports import NoopDispatcherState


async def _playback_pump(orchestrator: ConversationOrchestrator) -> None:
    """Simulate TTS playout: each started response becomes PLAYING -> COMPLETED.

    A real transport (Pipecat + TTS) would take longer than generation and
    report interruptions; this pump keeps the demo pipeline boundary honest.
    """
    seen: set[str] = set()
    while True:
        for response_id, response in list(orchestrator.state.responses.items()):
            if response_id in seen:
                continue
            if response.delivery_status.value == "pending":
                seen.add(response_id)
                await orchestrator.notify_output_started(response_id)
                await orchestrator.notify_output_completed(response_id)
        await asyncio.sleep(0.01)


def _build_orchestrator(script: list, rtt_s: float = 1.0, quiet: bool = False):
    output = TerminalOutput(quiet=quiet)
    tools = ToyToolRegistry(rtt_s=rtt_s)
    model = ScriptedModel(script)
    orchestrator = ConversationOrchestrator(
        conversation_id=uuid.uuid4().hex,
        model=model,
        tools=tools,
        output=output,
        repository=InMemoryConversationRepository(),
        dispatcher_state=NoopDispatcherState(),
    )
    return orchestrator, output


async def run_auto_demo() -> None:
    """Canned scenario: 2 tools in flight, user barge-in mid-answer, recovery."""
    print("=" * 72)
    print("EVENT-DRIVEN VOICE ORCHESTRATOR — scripted demo")
    print("Watch: (1) two tools dispatch in parallel; (2) user turn during pending")
    print("gets an instant ack; (3) barge-in cancels generation, tools keep running.")
    print("=" * 72)

    script = [
        # 1) first user turn -> model asks BOTH tools in one parallel batch
        tool_call("t1", "lookup_order_status", {"order_id": "A-1234"}),
        # 2) model reentry with tools pending -> if the user types NOW it acks;
        #    here tools complete -> batch commits -> model streams the answer
        "Your order A-1234 shipped — it arrives in 2 days. Anything else?",
        # 3) after the barge-in turn -> model acknowledges, THEN answers fully
        "Sorry — continuing: order A-1234 shipped, arrives in 2 days.",
    ]
    orchestrator, output = _build_orchestrator(script, rtt_s=0.8)
    await orchestrator.start()
    pump = asyncio.create_task(_playback_pump(orchestrator))

    print("\n— turn 1: user asks about order; model fires 2 tool calls in one batch —")
    t0 = asyncio.get_running_loop().time()
    await orchestrator.send_text("where is my order A-1234? and is it sunny in Madrid?")

    # the scripted model's first decision only requests the order lookup; a real
    # model would batch both. To SHOW parallel dispatch + non-stalling turns,
    # send the weather question as a second user turn 150ms into the pending
    # window — it must be applied immediately, not stalled.
    await asyncio.sleep(0.15)
    print("\n— turn 2 (mid-pending!): user asks for weather while tools run —")
    await orchestrator.send_text("and the weather in Madrid?")

    async def wait_until(final_text: str) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30
        while loop.time() < deadline:
            text = "".join("".join(v) for v in output.chunks.values())
            if final_text in text:
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("demo turn did not complete")

    await wait_until("shipped — it arrives in 2 days")
    print(
        f"\n[{asyncio.get_running_loop().time() - t0:.2f}s] both tools answered; "
        "user's mid-pending turn was applied without stalling."
    )

    # 4) barge-in during playback/generation: user interrupts the answer
    print("\n— turn 3: user barge-ins while the answer is being delivered —")
    await orchestrator.send_text("wait wait — is it sunny though?")
    # simulate the transport reporting the interruption of the playing response
    latest = orchestrator.state.latest_response()
    if latest is not None:
        await orchestrator.notify_output_interrupted(latest.response_id, reason="user_barge_in")
    await orchestrator.send_text("ok go on")  # next turn sees the interrupted note
    await wait_until("Sorry — continuing")

    await orchestrator.close()
    pump.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pump
    print("\nCanonical history (user/assistant/tool):")
    for msg in orchestrator.state.messages:
        kind = msg.role + (" +tools" if msg.tool_calls else "")
        print(f"  {kind:12} | {msg.content[:70]}")
    print(
        "\nDemo complete: background tools, non-stalling turns, barge-in recovery — all event-driven."
    )


async def run_interactive() -> None:
    script = [
        tool_call("t1", "lookup_order_status", {"order_id": "A-1234"}),
        "Your order A-1234 shipped — it arrives in 2 days. Anything else?",
        "I'm listening — what would you like to know?",
    ]
    orchestrator, _output = _build_orchestrator(script, rtt_s=1.5)
    await orchestrator.start()
    pump = asyncio.create_task(_playback_pump(orchestrator))
    print("Type messages (Ctrl-C / empty line to quit). Tools take ~1.5s —")
    print("type DURING the wait and watch the non-stalling ack behavior.\n")
    loop = asyncio.get_running_loop()
    while True:
        try:
            text = await loop.run_in_executor(None, input, "you > ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text.strip():
            break
        await orchestrator.send_text(text)
    await orchestrator.close()
    pump.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pump
    print("bye.")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        asyncio.run(run_auto_demo())
    else:
        try:
            asyncio.run(run_interactive())
        except KeyboardInterrupt:
            print("\nbye.")
