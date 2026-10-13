# Integrations

Optional adapters that wire the event-driven orchestrator into real frameworks.
The core library stays zero-dependency — every integration here is an extra.

## Pipecat adapter (`pipecat_adapter/`)

`EventDrivenOrchestratorBridge` is a Pipecat `FrameProcessor` that acts as a
**transport bridge** — it owns neither frame scheduling nor conversation state:

| Pipecat frame | Bridge action |
|---|---|
| `TranscriptionFrame` | `orchestrator.send_text(text)` |
| `BotStartedSpeakingFrame` | `notify_output_started(response_id)` |
| `BotStoppedSpeakingFrame` | `notify_output_completed(response_id)` |
| `InterruptionFrame` | `notify_output_interrupted(...)` + `orchestrator.interrupt()` |
| *(orchestrator text deltas)* | push `LLMTextFrame` downstream |
| *(response start/end)* | push `LLMFullResponse{Start,End}Frame` |

### Wiring into a pipeline

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor

from eventorchestrator.adapters.fake_model import ScriptedModel
from eventorchestrator.adapters.fake_tools import ToyToolRegistry
from eventorchestrator.application.orchestrator import ConversationOrchestrator
from eventorchestrator.domain.ports import NoopDispatcherState
from integrations.pipecat_adapter.pipecat_adapter import (
    EventDrivenOrchestratorBridge,
)

# 1. build the orchestrator with ANY ModelPort (ScriptedModel = no LLM needed)
orchestrator = ConversationOrchestrator(
    conversation_id="my-call",
    model=ScriptedModel(["Both orders shipped, ETA two days."]),
    tools=ToyToolRegistry(rtt_s=1.0),
    output=None,                    # the bridge installs its own output adapter
    repository=InMemoryConversationRepository(),
    dispatcher_state=NoopDispatcherState(),
)

# 2. drop the bridge into the pipeline (mid-pipeline, after the transport)
bridge = EventDrivenOrchestratorBridge(orchestrator=orchestrator)
pipeline = Pipeline([transport.input(), bridge, transport.output()])
```

Run the self-check (FakeModel path, no network):

```bash
python integrations/pipecat_adapter/test_bridge.py   # -> ALL CHECKS PASSED
```

### What the bridge does NOT own

- **No LLM context**: Pipecat's context aggregators keep working; the bridge
  never rewrites the provider message history (the orchestrator owns it).
- **No audio**: TTS/STT stages stay wherever they are; the bridge only ships text.
- **No frame scheduling**: Pipecat's clock/metrics/interruption flow pass through
  `super().process_frame()` untouched.

### Other frameworks

The same five ports (`src/eventorchestrator/domain/ports.py`) make any transport
a thin adapter. A livekit-agents plugin (EventSource → `send_text`, playback
callbacks → `notify_output_*`) follows the identical shape — planned.
