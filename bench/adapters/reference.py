"""Minimal conformance adapter — the template for your own integration.

Copy this file, rename it, and replace `build()` with a harness that wraps YOUR
runtime (its model port, its transport, its storage). Then:

  python -m bench.conformance bench.adapters.reference:build

This adapter wraps the reference ConversationOrchestrator with this repo's
scripted model and toy tools, so it always passes — use it as the baseline
shape and to smoke-test the runner itself.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bench.harness import EventTap, OrchestratorHarness  # noqa: E402
from eventorchestrator.adapters.fake_model import ScriptedModel, tool_call  # noqa: E402
from eventorchestrator.adapters.fake_tools import ToyToolRegistry  # noqa: E402
from eventorchestrator.adapters.memory_repository import (  # noqa: E402
    InMemoryConversationRepository,
)
from eventorchestrator.adapters.terminal import TerminalOutput  # noqa: E402
from eventorchestrator.application.orchestrator import ConversationOrchestrator  # noqa: E402
from eventorchestrator.domain.ports import NoopDispatcherState  # noqa: E402


@dataclass
class ConformanceTarget:
    """What the runner drives: a BenchHarness + optional tap + cell metadata."""

    harness: object  # OrchestratorHarness satisfies BenchHarness
    tap: EventTap | None = None

    def expected_batch_of4(self) -> dict | None:
        """Cell-2 metadata for gap math; None = runner falls back to completion-only."""
        return {"rtt_s": 0.05}


def build() -> ConformanceTarget:
    script = [
        # turn 1: single tool
        tool_call("c1", "lookup_order_status", {"order_id": "A-1234"}),
        "Order A-1234 shipped; ETA 2026-10-16.",
        # turn 2: parallel batch of four
        tool_call("b1", "lookup_order_status", {"order_id": "A-1"}),
        tool_call("b2", "lookup_order_status", {"order_id": "A-2"}),
        tool_call("b3", "lookup_order_status", {"order_id": "A-3"}),
        tool_call("b4", "lookup_order_status", {"order_id": "A-4"}),
        "All four orders shipped.",
        # turn 3 (drives the mid-pending ack cell): TWO slow tools dispatched by
        # this turn itself, so pending_count() >= 2 fires deterministically while
        # "hello, still there?" queues. One tool would peak at 1 pending and the
        # conformance runner would depend on cell-2 leftovers — a race.
        tool_call("w1", "lookup_order_status", {"order_id": "A-W1"}),
        tool_call("w2", "lookup_order_status", {"order_id": "A-W2"}),
        "Yes, I'm here.",
        # turn 4: long answer (drives the barge-in cell)
        "Once upon a time there was a voice agent that never stalled a user turn, "
        "and every caller lived happily ever after — the end of the long story.",
    ]
    orchestrator = ConversationOrchestrator(
        conversation_id=uuid.uuid4().hex,
        model=ScriptedModel(script),
        tools=ToyToolRegistry(rtt_s=0.05),
        output=TerminalOutput(quiet=True),
        repository=InMemoryConversationRepository(),
        dispatcher_state=NoopDispatcherState(),
    )
    tap = EventTap()
    return ConformanceTarget(harness=OrchestratorHarness(orchestrator, tap=tap), tap=tap)
