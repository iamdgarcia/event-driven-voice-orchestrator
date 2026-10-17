"""LiveKit adapter checks (plain asserts, scripted LLM — no network, no keys).

Run:  python -m tests.test_livekit_adapter     (or pytest tests/test_livekit_adapter.py)

Requires livekit-agents==1.8.8 (skips with a message when missing).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "livekit_adapter"))

try:
    import livekit_adapter  # noqa: F401
except ImportError:
    print("SKIP livekit-agents not installed (pip install livekit-agents==1.8.8)")
    sys.exit(0)

from bench.adapters.livekit_build import ConformanceTarget, build  # noqa: E402
from bench.conformance import run_conformance  # noqa: E402
from livekit_adapter import LiveKitHarness, TextSink  # noqa: E402

RTT_S = 0.1


async def t_adapter_constructs() -> None:
    target = build()
    assert isinstance(target, ConformanceTarget)
    assert isinstance(target.harness, LiveKitHarness)
    assert isinstance(target.harness.sink, TextSink)
    batch = target.expected_batch_of4()
    assert batch == {"rtt_s": RTT_S}, batch
    print("PASS adapter constructs; batch metadata rtt=100ms")


async def t_harness_protocol() -> None:
    target = build()
    h = target.harness
    for name in ("start", "send_user_text", "interrupt", "wait_idle", "pending_count"):
        assert callable(getattr(h, name)), name
    assert h.pending_count() == 0
    print("PASS 5-method BenchHarness protocol satisfied")


async def t_cells_and_invariants() -> None:
    target = build()
    results = await run_conformance(lambda: target)
    by_name = dict((name, (ok, detail)) for name, ok, detail in results)
    assert len(results) == 4, results
    for name, ok, detail in results:
        assert ok, f"invariant failed: {name} [{detail}]"
    # honest numbers: turn spans the tool RTT, parallel-4 beats serialization
    total1_ok, total1 = by_name["single-tool turn completed (spans >= 1 tool RTT)"]
    assert total1_ok
    batch_ok = by_name["parallel-4 beats serialized (never worse than k*RTT)"][0]
    assert batch_ok
    print("PASS conformance cells 4/4:", [d for _, _, d in results])


async def t_pending_and_text_visible() -> None:
    target = build()
    h = target.harness
    await h.start()
    await h.send_user_text("status of order A-1234")
    for _ in range(400):
        if h.pending_count() > 0:
            break
        await asyncio.sleep(0.005)
    assert h.pending_count() >= 1, "tool call never became pending"
    while h.pending_count() > 0:
        await asyncio.sleep(0.005)
    await h.wait_idle()
    text = h.sink.all_text()
    assert "Order A-1234" in text, text
    print(f"PASS pending window observed; assistant text delivered ({len(text)} chars)")


async def t_barge_in_interrupts() -> None:
    target = build()
    h = target.harness
    await h.start()
    await h.send_user_text("tell me a long story")
    for _ in range(200):
        if h.sink.all_text():
            break
        await asyncio.sleep(0.005)
    assert h.sink.all_text(), "generation never started"
    partial_before = len(h.sink.all_text())
    await h.interrupt()
    await h.wait_idle()
    # after interrupt the session must be quiet (no unbounded growth)
    await asyncio.sleep(0.05)
    assert h.session.current_speech is None or h.session.current_speech.done()
    assert partial_before > 0
    print("PASS barge-in: generation stopped, partial text captured before stop")


async def main() -> None:
    await t_adapter_constructs()
    await t_harness_protocol()
    await t_pending_and_text_visible()
    await t_barge_in_interrupts()
    await t_cells_and_invariants()  # last: full 4-cell run
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
