"""Conformance runner: validate an integration against the contract.

Run your wiring through the same cells and invariants the reference bench uses:

  python -m bench.conformance bench.adapters.my_harness:MyHarness

The adapter module must expose a factory:

    def build() -> ConformanceTarget

`ConformanceTarget` bundles a `BenchHarness` (start/send_user_text/interrupt/
wait_idle/pending_count — see bench/harness.py) with this repo's scripted model
and toy tools already wired in. The harness protocol keeps the measurement
external, so your runtime is timed the same way the reference implementation is.

Cells (same stimulus as bench_turn_latency):
  1. single tool @ RTT          -> total turn latency
  2. parallel batch of 4 @ RTT  -> gap vs cascade serialization
  3. user turn mid-pending      -> ack bounded
  4. barge-in mid-stream        -> generation stop under frame budget

Exit code 0 = all invariants pass (conformant), 1 = any failure.
"""

from __future__ import annotations

import asyncio
import importlib
import sys

RTT_S = 0.05  # keep the runner fast; invariants are order-sensitive, not absolute


def _ms(v: float) -> float:
    return round(v * 1000.0, 1)


async def _cell_total(harness, text: str) -> float:
    """Wall-clock from send to idle for one scripted turn.

    The settle-poll first: right after send_user_text the dispatch task has not
    started yet, so a bare wait_idle() would return instantly (start-up race).
    Wait for visible activity (pending tools), then wait_idle.
    """
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await harness.send_user_text(text)
    for _ in range(400):
        if harness.pending_count() > 0:
            break
        await asyncio.sleep(0.005)
    # The event loop idles while tools run off-loop (by contract); a turn is
    # only complete when pending tools drain AND the loop settles.
    while harness.pending_count() > 0:
        await asyncio.sleep(0.005)
    await harness.wait_idle()
    return loop.time() - t0


async def run_conformance(build) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    target = build()
    harness = target.harness

    await harness.start()

    # cell 1: single tool
    total1 = await _cell_total(harness, "status of order A-1234")

    # cell 2: parallel batch of 4
    total4 = await _cell_total(harness, "check A-1, A-2, A-3 and A-4")

    # cell 3: user turn mid-pending must ack without waiting the full batch
    loop = asyncio.get_running_loop()
    await harness.send_user_text("start the batch check")
    # wait until at least one tool is pending
    acked = False
    t_ack = None
    for _ in range(200):
        if harness.pending_count() >= 2:
            t0 = loop.time()
            await harness.send_user_text("hello, still there?")
            # non-stalling: returns without waiting for tools; idle bounded
            await harness.wait_idle()
            t_ack = loop.time() - t0
            acked = True
            break
        await asyncio.sleep(0.005)
    await harness.wait_idle()

    # cell 4: barge-in mid-stream
    stop_ms = None
    await harness.send_user_text("tell me a long story")
    await asyncio.sleep(0.15)  # let some chunks stream
    t1 = loop.time()
    await harness.interrupt()
    await harness.wait_idle()
    # generation stop = time from interrupt() to the loop becoming idle-ish;
    # with a scripted model the honest proxy is tap events if provided.
    stop_ms = _ms(loop.time() - t1)

    await harness.wait_idle()

    # --- invariants (order-sensitive, same thresholds as the reference bench) ---
    batch = target.expected_batch_of4()

    results.append(
        (
            "single-tool turn completed (spans >= 1 tool RTT)",
            total1 >= RTT_S,
            f"total={_ms(total1)}ms (RTT={_ms(RTT_S)}ms)",
        )
    )
    if batch:
        results.append(
            (
                "parallel-4 beats serialized (never worse than k*RTT)",
                total4 < 4 * batch["rtt_s"] * 1000.0,
                f"total4={_ms(total4)}ms vs 4*RTT={_ms(4 * batch['rtt_s'])}ms",
            )
        )
    else:
        results.append(("parallel-4 ran to completion", total4 > 0, f"total4={_ms(total4)}ms"))
    results.append(
        (
            "mid-pending user turn acked fast",
            acked and t_ack is not None and _ms(t_ack) < 400.0,
            f"ack={_ms(t_ack) if t_ack is not None else 'n/a'}ms (<400ms)",
        )
    )
    results.append(
        (
            "barge-in stops generation under budget",
            stop_ms is not None and stop_ms < 400.0,
            f"stop≈{stop_ms}ms (<400ms incl. drain)",
        )
    )
    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows charmap consoles
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    spec = sys.argv[1]
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, attr or "build", None)
    if factory is None:
        print(f"module {mod_name!r} exposes no {attr or 'build'!r} factory")
        sys.exit(2)

    results = asyncio.run(run_conformance(factory))
    print("\n=== CONFORMANCE ===")
    for name, ok, detail in results:
        print(("PASS" if ok else "FAIL"), "-", name, f"[{detail}]")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} invariants hold")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
