"""Turn-latency benchmark: cascade vs event-driven, on the real orchestrator.

Strategies (scripted model, simulated tool RTT):

  A — CASCADE EMULATION: one tool call per batch; the model only emits a
      wait-filler while a batch is pending, so batches serialize and the final
      answer lands after n_tools x RTT.
  B — EVENT-DRIVEN PARALLEL: all tool calls in ONE batch; the runtime
      dispatches them in parallel and re-enters the model once -> 1 x RTT.
  C — EVENT-DRIVEN + USER TURN DURING PENDING: like B, plus a second user
      turn 50ms into the pending window; the orchestrator applies it as an
      event and the model acks immediately -> no stall.

Metric: end-user-perceived turn latency = last user token -> first visible
bot chunk (monotonic clock, medians over N runs).

Run:  python -m bench.bench_turn_latency
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eventorchestrator.adapters.fake_tools import ToyToolRegistry  # noqa: E402
from eventorchestrator.adapters.memory_repository import (  # noqa: E402
    InMemoryConversationRepository,
)
from eventorchestrator.application.orchestrator import ConversationOrchestrator  # noqa: E402
from eventorchestrator.domain.ports import (  # noqa: E402
    DecisionCompleted,
    ModelStreamEvent,
    NoopDispatcherState,
    TextDelta,
)
from eventorchestrator.domain.models import AssistantDecision, ToolCall  # noqa: E402

FINAL_TEXT = "FINAL-ANSWER-MARKER"
FILLER_TEXT = "One moment."
ACK_TEXT = "Te escucho..."
CALL_TEXT = ""
CHUNK_EMIT_S = 0.015


class BenchOutput:
    """Records visible chunks + monotonic marks (OutputPort)."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.first_text_at: float | None = None
        self.marks: dict[str, float] = {}
        self._armed: str | None = None

    def arm(self, tag: str) -> None:
        self._armed = tag

    async def start_response(self, response_id: str, turn_id: str) -> None:
        if self._armed and self._armed not in self.marks:
            self.marks[self._armed] = asyncio.get_running_loop().time()

    async def text_delta(self, response_id: str, text: str) -> None:
        now = asyncio.get_running_loop().time()
        if self.first_text_at is None:
            self.first_text_at = now
        if self._armed and self._armed not in self.marks:
            self.marks[self._armed] = now
        self.chunks.append(text)

    async def end_response(self, response_id: str) -> None: ...

    async def cancel_response(self, response_id: str) -> None: ...

    async def tool_started(self, pending_call) -> None: ...

    async def tool_finished(self, pending_call, result) -> None: ...


class ScriptedBenchModel:
    """Strategy-scripted model (A = serialize, B/C = one parallel batch)."""

    def __init__(self, strategy: str, n_tools: int) -> None:
        self._strategy = strategy
        self._n = n_tools
        self._issued = 0
        self.invocations = 0

    async def stream(self, *, system_prompt: str, messages, tools) -> AsyncIteratorType(
        ModelStreamEvent
    ):
        self.invocations += 1
        pending = "pending operations" in system_prompt

        async def emit(count: int, text: str):
            for _ in range(count):
                yield TextDelta(text)
                await asyncio.sleep(CHUNK_EMIT_S)

        if self._strategy.startswith("A"):
            if pending:
                if self._issued >= self._n:
                    async for ev in emit(6, "y"):
                        yield ev
                    yield TextDelta(FINAL_TEXT)
                    yield DecisionCompleted(decision=AssistantDecision(content=FINAL_TEXT))
                    return
                async for ev in emit(1, FILLER_TEXT):
                    yield ev
                yield DecisionCompleted(decision=AssistantDecision(content=FILLER_TEXT))
                return
            if self._issued < self._n:
                self._issued += 1
                async for ev in emit(3, "x"):
                    yield ev
                call = ToolCall(
                    call_id=f"c{self._issued}", name="lookup_order_status", arguments={}
                )
                yield DecisionCompleted(
                    decision=AssistantDecision(content=CALL_TEXT, tool_calls=(call,))
                )
                return
            async for ev in emit(6, "y"):
                yield ev
            yield TextDelta(FINAL_TEXT)
            yield DecisionCompleted(decision=AssistantDecision(content=FINAL_TEXT))
            return

        # B / C: one parallel batch up front.
        if pending:
            text = ACK_TEXT if self._strategy == "C" else FILLER_TEXT
            async for ev in emit(1, text):
                yield ev
            yield DecisionCompleted(decision=AssistantDecision(content=text))
            return
        if self._issued == 0:
            self._issued = self._n
            async for ev in emit(3, "x"):
                yield ev
            calls = tuple(
                ToolCall(call_id=f"c{i + 1}", name="lookup_order_status", arguments={})
                for i in range(self._n)
            )
            yield DecisionCompleted(decision=AssistantDecision(content=CALL_TEXT, tool_calls=calls))
            return
        async for ev in emit(6, "y"):
            yield ev
        yield TextDelta(FINAL_TEXT)
        yield DecisionCompleted(decision=AssistantDecision(content=FINAL_TEXT))


def AsyncIteratorType(t):  # noqa: N802 — tiny alias to keep the signature honest
    from collections.abc import AsyncIterator

    return AsyncIterator[t]


async def run_scenario(
    strategy: str, *, rtt_ms: float, n_tools: int, double_turn: bool
) -> dict[str, float | int | None]:
    """One wall-clock conversation; latency marks in ms."""
    output = BenchOutput()
    model = ScriptedBenchModel(strategy=strategy, n_tools=n_tools)
    orchestrator = ConversationOrchestrator(
        conversation_id="bench",
        model=model,  # type: ignore[arg-type]
        tools=ToyToolRegistry(rtt_s=rtt_ms / 1000.0),
        output=output,  # type: ignore[arg-type]
        repository=InMemoryConversationRepository(),
        dispatcher_state=NoopDispatcherState(),
    )
    loop = asyncio.get_running_loop()
    await orchestrator.start()

    t0 = loop.time()
    await orchestrator.send_text("check my orders")
    t_send2: float | None = None
    if double_turn:
        await asyncio.sleep(0.05)
        t_send2 = loop.time()
        output.arm("turn2")
        await orchestrator.send_text("and one more thing")

    def turn_done() -> bool:
        if orchestrator.state.pending_batches:
            return False
        return FINAL_TEXT in "".join(output.chunks)

    deadline = loop.time() + 60.0
    while not turn_done():
        if loop.time() > deadline:
            raise RuntimeError("turn did not complete")
        await asyncio.sleep(0.002)
    await orchestrator.wait_idle()
    total_ms = (loop.time() - t0) * 1000.0
    ttfa_ms = (output.first_text_at - t0) * 1000.0 if output.first_text_at else None
    ttfa2_ms = (
        (output.marks["turn2"] - t_send2) * 1000.0
        if double_turn and "turn2" in output.marks and t_send2 is not None
        else None
    )
    await orchestrator.close()
    return {
        "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms is not None else None,
        "total_ms": round(total_ms, 1),
        "ttfa2_ms": round(ttfa2_ms, 1) if ttfa2_ms is not None else None,
        "invocations": model.invocations,
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = max(0, -(-len(ordered) * p // 100) - 1)
    return ordered[int(idx)]


async def run_cell(*, rtt_ms: float, n_tools: int, n_runs: int) -> dict:
    plans = {"A": False, "B": False, "C": True}
    cell: dict = {"rtt_ms": rtt_ms, "tools": n_tools, "variants": {}}
    for name, double in plans.items():
        runs = [
            await run_scenario(name, rtt_ms=rtt_ms, n_tools=n_tools, double_turn=double)
            for _ in range(n_runs)
        ]
        stats: dict[str, float | int] = {}
        for key in ("ttfa_ms", "total_ms", "ttfa2_ms"):
            vals = [float(r[key]) for r in runs if r[key] is not None]
            if vals:
                stats[f"{key}_med"] = round(statistics.median(vals), 1)
        stats["invocations_med"] = statistics.median([r["invocations"] for r in runs])
        cell["variants"][name] = stats
    return cell


def check_invariants(results: list[dict]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    for cell in results:
        n, rtt = cell["tools"], cell["rtt_ms"]
        a, b, c = (cell["variants"][k] for k in ("A", "B", "C"))
        tag = f"rtt={rtt} tools={n}"

        # I1: parallel beats cascade, growing linearly in (tools-1)*RTT
        expected_gap = (n - 1) * rtt
        actual_gap = a["total_ms_med"] - b["total_ms_med"]
        ok = actual_gap >= expected_gap * 0.6 and b["total_ms_med"] <= a["total_ms_med"]
        checks.append(
            (
                f"1-parallel-beats-cascade [{tag}]",
                ok,
                f"A-B={actual_gap:.0f}ms ~ (tools-1)*RTT={expected_gap:.0f}ms",
            )
        )

        # I2: parallel never worse than cascade even at tools=1
        ok = b["total_ms_med"] <= a["total_ms_med"] + 50
        checks.append(
            (f"2-never-worse [{tag}]", ok, f"A={a['total_ms_med']} B={b['total_ms_med']}")
        )

        # I3: user turn during pending gets an ack bounded well under the RTT
        if n >= 2:
            ok = c["ttfa2_ms_med"] < min(rtt * 0.5, 260) + 150
            checks.append(
                (
                    f"3-turn2-ack-bounded [{tag}]",
                    ok,
                    f"ttfa2={c['ttfa2_ms_med']}ms < {min(rtt * 0.5, 260) + 150:.0f}ms",
                )
            )
    return checks


async def main() -> None:
    n_runs = 3
    cells = []
    for rtt in (100, 1000):
        for tools in (1, 4):
            print(f"... running cell rtt={rtt}ms tools={tools} (A/B/C x{n_runs})")
            cells.append(await run_cell(rtt_ms=rtt, n_tools=tools, n_runs=n_runs))

    print("\n=== RESULTS (medians, ms) ===")
    header = f"{'rtt':>5} {'tools':>5} | {'A total':>9} {'B total':>9} {'C total':>9} | {'A ttfa':>8} {'B ttfa':>8} {'C turn2-ack':>11} | inv(A/B/C)"
    print(header)
    print("-" * len(header))
    for c in cells:
        a, b, cc = (c["variants"][k] for k in ("A", "B", "C"))
        print(
            f"{c['rtt_ms']:>5} {c['tools']:>5} | "
            f"{a['total_ms_med']:>9.0f} {b['total_ms_med']:>9.0f} {cc['total_ms_med']:>9.0f} | "
            f"{a.get('ttfa_ms_med', 0):>8.0f} {b.get('ttfa_ms_med', 0):>8.0f} "
            f"{cc.get('ttfa2_ms_med', 0):>11.0f} | "
            f"{a['invocations_med']:.0f}/{b['invocations_med']:.0f}/{cc['invocations_med']:.0f}"
        )

    print("\n=== INVARIANTS ===")
    all_ok = True
    for name, ok, detail in check_invariants(cells):
        print(f"{'PASS' if ok else 'FAIL'} [{name}] {detail}")
        all_ok = all_ok and ok
    print(f"\ninvariants: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
