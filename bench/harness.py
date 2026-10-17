"""Bench harness protocol — the minimal seam for framework conformance cells.

The bench contract every runtime must satisfy to be measured by these cells:

    start()                      load state, arm the runtime
    send_user_text(t)            enqueue one user turn (never blocks on tools)
    interrupt()                  cancel active generation only (barge-in)
    wait_idle()                  block until queue + model invocation idle
    pending_count() -> int       count of active background tool calls

ConversationOrchestrator is the reference implementation; a framework adapter
wraps pipecat / LiveKit / LangGraph to run the SAME cells on the SAME protocol
(comparison = same protocol, different runtime).

EventTap: optional instrumentation sink (monotonic timestamps for model-invoke
start/end, tool dispatch/receive, delivery events). Zero behavior change when
None; taps never mutate the walk.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["BenchHarness", "EventTap", "OrchestratorHarness", "TapEvent"]

# Tap event kinds: model-invoke start/end, tool dispatch/receive, delivery events.
MODEL_START = "model_start"
MODEL_END = "model_end"
MODEL_CANCELLED = "model_cancelled"
TOOL_DISPATCH = "tool_dispatch"
TOOL_RECEIVE = "tool_receive"
DELIVERY_START = "delivery_start"
DELIVERY_END = "delivery_end"
DELIVERY_INTERRUPTED = "delivery_interrupted"


@runtime_checkable
class BenchHarness(Protocol):
    """The 5-method contract the bench cells drive. See module docstring."""

    async def start(self) -> None: ...

    async def send_user_text(self, t: str) -> None: ...

    async def interrupt(self) -> None: ...

    async def wait_idle(self) -> None: ...

    def pending_count(self) -> int: ...


@dataclass
class TapEvent:
    """One instrumentation sample: kind + monotonic timestamp + free-form labels."""

    kind: str
    t: float
    labels: dict = field(default_factory=dict)


class EventTap:
    """Optional instrumentation sink. Pass into a harness; disabled = never called.

    Fire-and-forget by contract: emits never await, never raise, never mutate
    runtime state. Assign `tap.sink = callable` or subclass to collect.
    """

    def __init__(self) -> None:
        self.events: list[TapEvent] = []
        self.sink: Callable[[TapEvent], None] | None = None

    def emit(self, kind: str, t: float, **labels) -> None:
        event = TapEvent(kind=kind, t=t, labels=labels)
        self.events.append(event)
        if self.sink is not None:
            self.sink(event)

    def since(self, t0: float, kinds: tuple[str, ...] | None = None) -> list[TapEvent]:
        return [e for e in self.events if e.t >= t0 and (kinds is None or e.kind in kinds)]

    def first(self, kind: str, after: float = 0.0) -> TapEvent | None:
        for e in self.events:
            if e.kind == kind and e.t >= after:
                return e
        return None


class OrchestratorHarness:
    """Reference BenchHarness wrapping the real ConversationOrchestrator.

    tap: optional EventTap — instrumented hooks fire on the orchestrator's own
    callback paths (model stream start/end/cancel, tool dispatch/receive,
    delivery start/end/interrupted). Zero behavior change when tap is None.
    """

    def __init__(self, orchestrator, tap: EventTap | None = None) -> None:
        self.orchestrator = orchestrator
        self._tap = tap

    def _emit(self, kind: str, **labels) -> None:
        if self._tap is not None:
            self._tap.emit(kind, asyncio.get_running_loop().time(), **labels)

    async def start(self) -> None:
        await self.orchestrator.start()

    async def send_user_text(self, t: str) -> None:
        self._emit("user_text", text=t)
        await self.orchestrator.send_text(t)

    async def interrupt(self) -> None:
        self._emit("interrupt_requested")
        await self.orchestrator.interrupt()

    async def wait_idle(self) -> None:
        await self.orchestrator.wait_idle()

    def pending_count(self) -> int:
        return len(self.orchestrator.state.active_calls())

    # ----- instrumentation hooks (called by the orchestrator wiring below) ----

    def instrument_model_stream(self):
        """Build a stream wrapper that taps model_start/model_end/model_cancelled.

        Usage: wrap the model's ORIGINAL bound stream BEFORE reassigning it —

            original = model.stream
            wrapper = harness.instrument_model_stream()
            model.stream = lambda *a, **kw: wrapper(original, **kw)

        The wrapper takes (stream_fn, *, system_prompt, messages, tools) so the
        captured function cannot recurse into the reassigned attribute.
        """

        async def stream_wrapper(stream_fn, *, system_prompt, messages, tools):
            self._emit(MODEL_START)
            cancelled = False
            try:
                async for ev in stream_fn(
                    system_prompt=system_prompt, messages=messages, tools=tools
                ):
                    yield ev
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                self._emit(MODEL_CANCELLED if cancelled else MODEL_END)

        return stream_wrapper

    def tap_delivery(self, output_port):
        """Wrap an OutputPort to tap delivery events (start/end/interrupted)."""

        class _TappedOutput:
            def __init__(self, inner):
                self._inner = inner

            async def start_response(self, response_id, turn_id):
                self._emit_at(DELIVERY_START, response_id=response_id)
                await self._inner.start_response(response_id, turn_id)

            async def text_delta(self, response_id, text):
                await self._inner.text_delta(response_id, text)

            async def end_response(self, response_id):
                self._emit_at(DELIVERY_END, response_id=response_id)
                await self._inner.end_response(response_id)

            async def cancel_response(self, response_id):
                self._emit_at(DELIVERY_INTERRUPTED, response_id=response_id)
                await self._inner.cancel_response(response_id)

            async def tool_started(self, pending_call):
                self._emit_at(TOOL_DISPATCH, call=pending_call.call.name)
                await self._inner.tool_started(pending_call)

            async def tool_finished(self, pending_call, result):
                self._emit_at(TOOL_RECEIVE, call=pending_call.call.name)
                await self._inner.tool_finished(pending_call, result)

            def _emit_at(self, kind, **labels):
                if self._harness._tap is not None:
                    self._harness._tap.emit(kind, asyncio.get_running_loop().time(), **labels)

        tapped = _TappedOutput(output_port)
        tapped._harness = self
        return tapped
