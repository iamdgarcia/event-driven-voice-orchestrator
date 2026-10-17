# LiveKit Agents conformance adapter (livekit-agents==1.8.8)

Wraps a roomless, headless `livekit.agents.voice.AgentSession` behind this repo's
5-method `BenchHarness` protocol (see `bench/harness.py`), so the SAME conformance
cells that validate the reference orchestrator run against LiveKit.

## Verdict (paper sentence)

**LiveKit Agents 1.8.8 satisfies all four contract invariants** (single-tool turn
spans ≥ 1 RTT, parallel-4 beats serialization, mid-pending user turn acks < 400 ms,
barge-in stop < 400 ms) when driven through the standard text path with
`generate_reply()` — tools DO run as background `asyncio` tasks
(`voice/tool_executor.py:408`), the reply queue is non-stalling, and
`interrupt(force=True)` stops generation within one poll tick. The measured numbers
on this box: turn ≈ 110–125 ms @ RTT=100 ms, parallel-4 ≈ 110–140 ms vs 400 ms
serialized, ack ≈ 94–110 ms, barge-in stop < 1 ms (scripted stream, no TTS playout).

## Why this works headless (1.8.8 specifics)

- `livekit.agents.testing.fake_job_context()` installs a worker-less `JobContext`;
  calling `session.start(agent, record=False)` with `room` OMITTED never touches
  WebRTC (no signal client, no audio device). Verified roomless on Windows.
  **Gotcha: pass `room=None` and it crashes** — `is_given(None)` is True in livekit,
  so an explicit None spins up a `RoomIO` against a null room
  (`agent_session.py:1073` → `AttributeError: 'NoneType' object has no attribute
  'isconnected'`). Omit the kwarg.
- User turns enter via `session.generate_reply(user_input=..., input_modality="text")`
  — the text path LiveKit's own simulation/console mode uses.
- Assistant text is captured with a `TextOutput` sink installed on
  `session.output.transcription` (the hook RoomIO's transcription forwarder uses).
- The scripted LLM subclasses `llm.LLM`/`llm.LLMStream` (exactly what provider plugins
  implement) and streams canned `ChatChunk`s — no network, no API keys.
- Tools are real `@function_tool`s with an `asyncio.sleep(rtt)` — same RTT knob as
  the reference `ToyToolRegistry`. RTT=0.1s (not 0.05): `asyncio.sleep` undershoots a
  50 ms deadline by one 15.6 ms Windows timer tick, tripping the `>= 1 RTT`
  invariant spuriously.
- Pending-tool introspection: `session._activity._tool_executor._running_tasks`
  (private; stable across 1.8.x and the only in-flight-tool count the framework
  exposes).

## Per-cell results (2026-10-17, Windows, Python 3.11, livekit-agents==1.8.8)

| Cell | Stimulus | Invariant | Result |
|---|---|---|---|
| 1. single tool | 1 tool @ RTT=100 ms | turn spans ≥ 1 tool RTT | **PASS** (109–125 ms, 5/5 runs) |
| 2. parallel batch | 4 tools @ RTT=100 ms | never worse than 4×RTT | **PASS** (110–141 ms vs 400 ms; livekit runs them concurrently) |
| 3. mid-pending turn | user turn while 2 tools pending | ack < 400 ms | **PASS** (94–110 ms; turn queues, doesn't stall) |
| 4. barge-in | interrupt mid-stream | stop < 400 ms | **PASS** (<1 ms measured; `interrupt(force=True)` + awaited handle) |

4/4 invariants hold, 5 consecutive runs. Caveats for the paper's honesty box:

- Cell-3's script must keep TWO tools pending (the runner waits for `>= 2`); one
  pending tool peaks at 1 and the cell would measure nothing. The reference adapter
  gets its `>= 2` from cell-2 leftovers (racy); this adapter makes it deterministic.
- The scripted LLM streams at 15 ms/chunk and there is no real TTS playout, so
  absolute numbers include framework overhead only — the same caveat as the
  reference run. Relative pass/fail per invariant is the signal.
- Cell-1's `>= 1 RTT` check is tight against the Windows timer at RTT=50 ms; we run
  RTT=100 ms. Documented, not hidden.

## The Jev plugin note (why 1.8.8)

livekit/agents 1.8.8 ships TypeSafe Jev decision-models natively
(`livekit.agents.llm.decisions`: batched probability/choice/score evaluations run as
background tasks, never delaying the reply). That is exactly the separation the
event-driven orchestrator treats as a contract: decision work leaves the critical
path. The conformance cells here exercise the *turn* path; the decision layer rides
alongside. Pin kept at 1.8.8 because that's where `llm.decisions` landed.

## Install

```bash
pip install livekit-agents==1.8.8   # pulls livekit-rtc, protobuf, openai types, otel (~90 packages)
```

Core library stays zero-dep; the adapter import fails with instructions when the
package is missing (same convention as `pipecat_adapter`).

## Run

```bash
python -m bench.conformance bench.adapters.livekit_build:build
python -m tests.test_livekit_adapter
```
