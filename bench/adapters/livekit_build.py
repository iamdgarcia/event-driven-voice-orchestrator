"""LiveKit conformance target — build() for the conformance runner.

Run:  python -m bench.conformance bench.adapters.livekit_build:build

Recorded run (Windows, Python 3.11, livekit-agents==1.8.8, 2026-10-17):
see the conformance runner output; commit history holds the exact numbers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVEKIT_DIR = ROOT / "integrations" / "livekit_adapter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LIVEKIT_DIR))

from bench.harness import EventTap  # noqa: E402
from livekit_adapter import (  # noqa: E402
    LiveKitHarness,
    ScriptedLLM,
    TextSink,
    make_lookup_tool,
    make_script,
)
from livekit.agents import voice  # noqa: E402


@dataclass
class ConformanceTarget:
    """Same shape as bench.adapters.reference.ConformanceTarget."""

    harness: LiveKitHarness
    tap: EventTap | None = None

    def expected_batch_of4(self) -> dict | None:
        # runner uses this as the serialized-vs-parallel yardstick; RTT here must match
        # make_lookup_tool(rtt_s=...) in build() below
        return {"rtt_s": 0.1}


def build() -> ConformanceTarget:
    sink = TextSink()
    session = voice.AgentSession(
        llm=ScriptedLLM(make_script()),
        # RTT 0.1 (not 0.05): asyncio.sleep on Windows undershoots a 50ms deadline by one
        # 15.6ms timer tick (~46ms), which trips the 'spans >= 1 RTT' invariant spuriously.
        tools=[make_lookup_tool(rtt_s=0.1)],
    )
    session.output.transcription = sink
    return ConformanceTarget(harness=LiveKitHarness(session, sink))
