"""The event loop: one canonical conversation state, generic events, model reentry.

Design invariants (all asserted by tests/ and bench/):

1. New user turns are accepted and acked WHILE tools are pending — the model
   is re-invoked with an internal description of pending calls; no stall.
2. A completed tool batch commits to history ATOMICALLY (assistant message
   with tool_calls + all tool result messages together) and triggers model
   reentry. The provider never sees an unresolved tool-call sequence.
3. Generation and delivery are independent lifecycles. `interrupt()` (barge-in)
   cancels ONLY generation; playback status is tracked via delivery events;
   an interrupted delivery is injected as internal context on the next turn.
4. No semantic interpretation: the runtime never classifies user wording
   (correction? cancellation? replacement?) — the model owns those decisions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from eventorchestrator.application.tool_dispatcher import ToolDispatcher
from eventorchestrator.domain.events import (
    ConversationEvent,
    OutputDeliveryCompleted,
    OutputDeliveryInterrupted,
    OutputDeliveryStarted,
    ToolCallFinished,
    UserTextReceived,
)
from eventorchestrator.domain.models import (
    AssistantDecision,
    AssistantResponseState,
    ConversationMessage,
    ConversationState,
    PendingToolBatch,
    PendingToolCall,
    RequestStatus,
    ResponseDeliveryStatus,
    ResponseGenerationStatus,
    ToolExecutionResult,
)
from eventorchestrator.domain.ports import (
    ConversationRepositoryPort,
    DecisionCompleted,
    ModelPort,
    OutputPort,
    TextDelta,
    ToolDispatcherStatePort,
    ToolRegistryPort,
)

LOGGER = logging.getLogger(__name__)


class ConversationOrchestrator:
    """Own conversation state and coordinate model and background tools."""

    def __init__(
        self,
        *,
        conversation_id: str,
        model: ModelPort,
        tools: ToolRegistryPort,
        output: OutputPort,
        repository: ConversationRepositoryPort,
        dispatcher_state: ToolDispatcherStatePort,
    ) -> None:
        self._conversation_id = conversation_id
        self._model = model
        self._tools = tools
        self._output = output
        self._repository = repository
        self._state = ConversationState()
        self._events: asyncio.Queue[ConversationEvent] = asyncio.Queue()
        self._processor_task: asyncio.Task[None] | None = None
        self._model_task: asyncio.Task[AssistantDecision] | None = None
        self._response_open = False
        self._current_response_id: str | None = None
        self._current_response_turn_id: str | None = None
        self._closed = False
        self._dispatcher = ToolDispatcher(
            conversation_id, tools, self._publish_tool_event, dispatcher_state
        )

    @property
    def state(self) -> ConversationState:
        """Return the live canonical state for diagnostics."""
        return self._state

    async def start(self) -> None:
        """Load persisted state before receiving turns."""
        self._state = await self._repository.load(self._conversation_id)

    async def send_text(self, text: str) -> None:
        """Enqueue one user turn from any transport. Never blocks on tools."""
        normalized = text.strip()
        if not normalized:
            return
        if self._closed:
            raise RuntimeError("The conversation orchestrator is closed.")
        await self._events.put(UserTextReceived(text=normalized))
        self._ensure_processor()

    async def interrupt(self) -> None:
        """Cancel only the active model response, never background tools."""
        model_task = self._model_task
        if model_task is None or model_task.done():
            return
        model_task.cancel()
        await asyncio.gather(model_task, return_exceptions=True)
        self._model_task = None
        await self._cancel_open_response()

    async def notify_output_started(self, response_id: str) -> None:
        await self._events.put(OutputDeliveryStarted(response_id=response_id))
        self._ensure_processor()

    async def notify_output_completed(self, response_id: str) -> None:
        await self._events.put(OutputDeliveryCompleted(response_id=response_id))
        self._ensure_processor()

    async def notify_output_interrupted(self, response_id: str, *, reason: str) -> None:
        await self._events.put(OutputDeliveryInterrupted(response_id=response_id, reason=reason))
        self._ensure_processor()

    async def wait_idle(self) -> None:
        """Wait until the event queue and the model invocation are idle."""
        while True:
            processor = self._processor_task
            if processor is None and self._events.empty():
                return
            if processor is not None:
                await asyncio.shield(processor)
            else:
                await asyncio.sleep(0)

    async def close(self) -> None:
        """Finish background work and persist the final state."""
        self._closed = True
        await self.interrupt()
        await self._dispatcher.wait_all()
        self._ensure_processor()
        await self.wait_idle()
        await self._repository.save(self._conversation_id, self._state)

    # ----- event loop -----------------------------------------------------

    def _ensure_processor(self) -> None:
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_events())

    async def _process_events(self) -> None:
        try:
            while not self._events.empty():
                event = await self._events.get()
                should_invoke = await self._apply_event(event)
                should_invoke = await self._drain_ready_events(should_invoke)
                if should_invoke:
                    await self._run_interruptible_model()
        finally:
            self._processor_task = None
            if not self._events.empty():
                self._ensure_processor()

    async def _run_interruptible_model(self) -> None:
        """Invoke the model; incoming interrupting events cancel it mid-stream.

        The wait set is {model_task, next-event}: if a user turn (or a tool
        completion, or a barge-in) arrives during generation, the generation is
        cancelled immediately and the loop re-runs with fresh state — the
        S2S-like non-stalling behavior.
        """
        should_invoke = True
        while should_invoke:
            self._model_task = asyncio.create_task(self._run_model_once())
            while True:
                event_task = asyncio.create_task(self._events.get())
                done, _ = await asyncio.wait(
                    {self._model_task, event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if event_task in done:
                    event = event_task.result()
                    event_should_invoke = await self._apply_event(event)
                    if not self._event_interrupts_model(event):
                        continue
                    self._model_task.cancel()
                    await asyncio.gather(self._model_task, return_exceptions=True)
                    self._model_task = None
                    await self._cancel_open_response()
                    should_invoke = await self._drain_ready_events(event_should_invoke)
                    break

                event_task.cancel()
                await asyncio.gather(event_task, return_exceptions=True)
                if self._model_task is None or self._model_task.cancelled():
                    # a concurrent event path already cancelled + cleaned up the model task
                    self._model_task = None
                    await self._cancel_open_response()
                    return
                decision = self._model_task.result()
                self._model_task = None
                await self._handle_decision(decision)
                return

    async def _run_model_once(self) -> AssistantDecision:
        self._current_response_id = uuid.uuid4().hex
        self._current_response_turn_id = uuid.uuid4().hex
        decision: AssistantDecision | None = None
        response_text = ""
        stream = self._model.stream(
            system_prompt=self._system_prompt(),
            messages=tuple(self._state.messages),
            tools=self._tools.active_tools(),
        )
        async for event in stream:
            if isinstance(event, TextDelta):
                response_text += event.text
                await self._emit_text(event.text)
            elif isinstance(event, DecisionCompleted):
                decision = event.decision
        await self._finish_open_response()
        if decision is None:
            raise RuntimeError("The model stream completed without a decision.")
        return decision

    async def _handle_decision(self, decision: AssistantDecision) -> None:
        if not decision.tool_calls:
            self._state.messages.append(
                ConversationMessage(role="assistant", content=decision.content)
            )
            await self._save()
            return

        batch_id = uuid.uuid4().hex
        turn_id = uuid.uuid4().hex
        batch = PendingToolBatch(batch_id=batch_id, turn_id=turn_id, decision=decision)
        self._state.pending_batches[batch_id] = batch
        pending_calls: list[PendingToolCall] = []
        for call in decision.tool_calls:
            pending = PendingToolCall(batch_id=batch_id, turn_id=turn_id, call=call)
            self._state.pending_calls[call.call_id] = pending
            pending_calls.append(pending)
            await self._output.tool_started(pending)
        await self._dispatcher.dispatch(pending_calls)
        await self._save()

    async def _apply_event(self, event: ConversationEvent) -> bool:
        if isinstance(event, UserTextReceived):
            return await self._apply_user_text(event)
        if isinstance(event, ToolCallFinished):
            return await self._apply_tool_result(event)
        if isinstance(event, OutputDeliveryStarted):
            return await self._apply_output_started(event)
        if isinstance(event, OutputDeliveryCompleted):
            return await self._apply_output_completed(event)
        return await self._apply_output_interrupted(event)

    @staticmethod
    def _event_interrupts_model(event: ConversationEvent) -> bool:
        """User turns, tool completions and barge-ins cancel active generation."""
        return isinstance(event, (UserTextReceived, ToolCallFinished, OutputDeliveryInterrupted))

    async def _apply_user_text(self, event: UserTextReceived) -> bool:
        self._state.messages.append(ConversationMessage(role="user", content=event.text))
        LOGGER.info("User turn received: text=%r", event.text)
        await self._save()
        return True

    async def _apply_tool_result(self, event: ToolCallFinished) -> bool:
        result = event.result
        pending = self._state.pending_calls.get(result.call_id)
        if pending is None:
            LOGGER.warning("Ignoring unknown tool result: %s", result.call_id)
            return False
        batch = self._state.pending_batches[pending.batch_id]
        batch.results[result.call_id] = result
        pending.status = (
            RequestStatus.FAILED if result.error is not None else RequestStatus.COMPLETED
        )
        if not pending.status == RequestStatus.FAILED:
            await self._output.tool_finished(pending, result)
        if not batch.is_complete():
            await self._save()
            return False

        # Atomic commit: assistant tool-call message + all results, together.
        self._commit_completed_batch(batch)
        self._state.pending_batches.pop(batch.batch_id, None)
        await self._save()
        return True

    async def _apply_output_started(self, event: OutputDeliveryStarted) -> bool:
        response = self._state.responses.get(event.response_id)
        if response is None:
            LOGGER.warning("Ignoring playback start for unknown: %s", event.response_id)
            return False
        if response.delivery_status == ResponseDeliveryStatus.PENDING:
            response.delivery_status = ResponseDeliveryStatus.PLAYING
            await self._save()
        return False

    async def _apply_output_completed(self, event: OutputDeliveryCompleted) -> bool:
        response = self._state.responses.get(event.response_id)
        if response is None:
            LOGGER.warning("Ignoring playback completion for unknown: %s", event.response_id)
            return False
        if response.delivery_status != ResponseDeliveryStatus.INTERRUPTED:
            response.delivery_status = ResponseDeliveryStatus.COMPLETED
            await self._save()
        return False

    async def _apply_output_interrupted(self, event: OutputDeliveryInterrupted) -> bool:
        response = self._state.responses.get(event.response_id)
        if response is None:
            LOGGER.warning("Ignoring playback interruption for unknown: %s", event.response_id)
            return False
        if response.delivery_status != ResponseDeliveryStatus.COMPLETED:
            response.delivery_status = ResponseDeliveryStatus.INTERRUPTED
            response.interruption_reason = event.reason
            await self._save()
        return False

    # ----- history / output -------------------------------------------------

    def _commit_completed_batch(self, batch: PendingToolBatch) -> None:
        self._state.messages.append(
            ConversationMessage(
                role="assistant",
                content=batch.decision.content,
                tool_calls=batch.decision.tool_calls,
            )
        )
        for call in batch.decision.tool_calls:
            result = batch.results[call.call_id]
            self._state.messages.append(
                ConversationMessage(
                    role="tool",
                    content=self._tool_message_content(result),
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                )
            )

    async def _drain_ready_events(self, should_invoke: bool) -> bool:
        while True:
            try:
                event = self._events.get_nowait()
            except asyncio.QueueEmpty:
                return should_invoke
            should_invoke = await self._apply_event(event) or should_invoke

    async def _emit_text(self, text: str) -> None:
        if not text:
            return
        response_id = self._current_response_id
        turn_id = self._current_response_turn_id
        if response_id is None or turn_id is None:
            raise RuntimeError("Visible output requires an active model response.")
        if not self._response_open:
            self._state.responses[response_id] = AssistantResponseState(
                response_id=response_id, turn_id=turn_id
            )
            await self._output.start_response(response_id, turn_id)
            self._response_open = True
        self._state.responses[response_id].content += text
        await self._output.text_delta(response_id, text)

    async def _finish_open_response(self) -> None:
        if not self._response_open:
            return
        response_id = self._require_current_response_id()
        self._state.responses[response_id].generation_status = ResponseGenerationStatus.COMPLETED
        await self._output.end_response(response_id)
        self._response_open = False

    async def _cancel_open_response(self) -> None:
        if not self._response_open:
            return
        response_id = self._require_current_response_id()
        response = self._state.responses[response_id]
        response.generation_status = ResponseGenerationStatus.CANCELLED
        if response.delivery_status != ResponseDeliveryStatus.INTERRUPTED:
            response.delivery_status = ResponseDeliveryStatus.INTERRUPTED
            response.interruption_reason = "generation_cancelled"
        await self._output.cancel_response(response_id)
        self._response_open = False
        await self._save()

    async def _save(self) -> None:
        await self._repository.save(self._conversation_id, self._state)

    def _system_prompt(self) -> str:
        """Base prompt + INTERNAL runtime context the model must not speak.

        Two injections: (1) currently pending tool calls (the model owns the
        decision to wait, ack, or talk over them); (2) an interrupted-delivery
        note (do not assume the user heard the last answer — the heard /
        not-heard signal that cascade pipelines lose).
        """
        base_prompt = self._tools.system_prompt()
        internal_sections: list[str] = []
        active_calls = self._state.active_calls()
        if active_calls:
            pending_context = "\n".join(
                f"- tool={p.call.name}, arguments={p.call.arguments}, status={p.status}"
                for p in active_calls
            )
            internal_sections.append(
                "Currently pending operations. This block is internal state, "
                "do not mention it to the user:\n"
                f"{pending_context}"
            )
        latest_response = self._state.latest_response()
        if (
            latest_response is not None
            and latest_response.delivery_status == ResponseDeliveryStatus.INTERRUPTED
        ):
            internal_sections.append(
                "Playback of the last visible response was interrupted. Do not "
                "assume the user heard its full content; use their next turn to "
                "decide whether to resume it, summarize it, or move on. Do not "
                "mention this internal state."
            )
        if not internal_sections:
            return base_prompt
        return f"{base_prompt}\n\n" + "\n\n".join(internal_sections)

    async def _publish_tool_event(self, event: ToolCallFinished) -> None:
        await self._events.put(event)
        self._ensure_processor()

    def _require_current_response_id(self) -> str:
        response_id = self._current_response_id
        if response_id is None:
            raise RuntimeError("No active model response identifier is available.")
        return response_id

    @staticmethod
    def _tool_message_content(result: ToolExecutionResult) -> str:
        return json.dumps(result.content, ensure_ascii=False, default=str)
