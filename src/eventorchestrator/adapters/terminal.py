"""Terminal transport: prints lifecycle events with timestamps.

In a real voice stack this is where Pipecat sits: text_delta -> TTS stream,
and the transport reports playback back via notify_output_* methods (here
delivery is simulated as instant PLAYING->COMPLETED after each response).
"""

from __future__ import annotations

import time

from eventorchestrator.domain.models import PendingToolCall, ToolExecutionResult
from eventorchestrator.domain.ports import OutputPort


def _ts() -> str:
    return f"{time.monotonic() % 1000:07.3f}s"


class TerminalOutput(OutputPort):
    """Print every lifecycle event with a monotonic timestamp."""

    def __init__(self, *, quiet: bool = False) -> None:
        self.quiet = quiet
        self.chunks: dict[str, list[str]] = {}
        self.completed: list[str] = []
        self.interrupted: list[tuple[str, str]] = []

    def _say(self, line: str) -> None:
        if not self.quiet:
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                # Windows cp1252 consoles: keep running, strip unprintable glyphs
                print(line.encode("ascii", "replace").decode(), flush=True)

    async def start_response(self, response_id: str, turn_id: str) -> None:
        self.chunks[response_id] = []
        self._say(f"[{_ts()}] ── bot response ({response_id[:8]})")

    async def text_delta(self, response_id: str, text: str) -> None:
        self.chunks.setdefault(response_id, []).append(text)
        self._say(
            f"[{_ts()}] {text}",
        )

    async def end_response(self, response_id: str) -> None:
        self.completed.append(response_id)
        self._say(f"[{_ts()}] ── response end ({response_id[:8]})")

    async def cancel_response(self, response_id: str) -> None:
        self._say(f"[{_ts()}] ── response CANCELLED (barge-in) ({response_id[:8]})")

    async def tool_started(self, pending_call: PendingToolCall) -> None:
        self._say(
            f"[{_ts()}] ⚙ dispatch {pending_call.call.name}({pending_call.call.arguments}) "
            f"[{pending_call.call.call_id}]"
        )

    async def tool_finished(
        self, pending_call: PendingToolCall, result: ToolExecutionResult
    ) -> None:
        self._say(f"[{_ts()}] ✔ {result.name} -> {result.content}")


async def simulate_instant_playback(orchestrator, response_id: str) -> None:
    """Report PENDING->PLAYING->COMPLETED for one response (no real TTS here)."""
    await orchestrator.notify_output_started(response_id)
    await orchestrator.notify_output_completed(response_id)
