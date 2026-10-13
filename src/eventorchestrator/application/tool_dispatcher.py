"""Background tool execution without owning agent state.

Tools run as independent asyncio tasks. A barge-in (model-generation cancel)
never touches them; their completion re-enters the model through the event
loop. Dispatched calls are NOT cancelable here — if your product needs
invalidation or cancellation semantics, add an explicit policy port; the
runtime stays generic.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime

from eventorchestrator.domain.events import ToolCallFinished
from eventorchestrator.domain.models import (
    PendingToolCall,
    ToolDispatchRecord,
    ToolDispatcherSnapshot,
    ToolDispatchStatus,
    ToolExecutionResult,
)
from eventorchestrator.domain.ports import ToolDispatcherStatePort, ToolRegistryPort

LOGGER = logging.getLogger(__name__)
EventPublisher = Callable[[ToolCallFinished], Awaitable[None]]


class ToolDispatcher:
    """Execute tool calls in the background."""

    def __init__(
        self,
        conversation_id: str,
        registry: ToolRegistryPort,
        publish_event: EventPublisher,
        state_store: ToolDispatcherStatePort,
    ) -> None:
        self._conversation_id = conversation_id
        self._registry = registry
        self._publish_event = publish_event
        self._state_store = state_store
        self._tasks: set[asyncio.Task[None]] = set()
        self._records: dict[str, ToolDispatchRecord] = {}
        self._state_lock = asyncio.Lock()

    async def dispatch(self, calls: Sequence[PendingToolCall]) -> None:
        """Schedule every tool call independently (parallel execution)."""
        queued_at = _timestamp()
        async with self._state_lock:
            for pending in calls:
                call = pending.call
                self._records[call.call_id] = ToolDispatchRecord(
                    conversation_id=self._conversation_id,
                    batch_id=pending.batch_id,
                    turn_id=pending.turn_id,
                    call_id=call.call_id,
                    tool_name=call.name,
                    arguments=dict(call.arguments),
                    status=ToolDispatchStatus.QUEUED,
                    queued_at=queued_at,
                )
            await self._save_locked()

        for pending in calls:
            task = asyncio.create_task(self._execute(pending))
            self._tasks.add(task)
            task.add_done_callback(self._handle_task_done)

    async def wait_all(self) -> None:
        """Wait for all currently executing tools (shutdown path)."""
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(self, pending: PendingToolCall) -> None:
        call = pending.call
        await self._update_record(
            call.call_id, status=ToolDispatchStatus.RUNNING, started_at=_timestamp()
        )
        try:
            result = await self._registry.execute(call)
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Tool execution failed: %s", call.name)
            result = ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                content={"error": str(error)},
                error=str(error),
            )
        await self._update_record(
            call.call_id,
            status=(ToolDispatchStatus.FAILED if result.error else ToolDispatchStatus.COMPLETED),
            finished_at=_timestamp(),
            result=result.content,
            error=result.error,
        )
        await self._publish_event(ToolCallFinished(result=result))

    async def _update_record(self, call_id: str, **updates: object) -> None:
        async with self._state_lock:
            record = self._records[call_id]
            for field_name, value in updates.items():
                setattr(record, field_name, value)
            await self._save_locked()

    async def _save_locked(self) -> None:
        snapshot = ToolDispatcherSnapshot(
            conversation_id=self._conversation_id,
            updated_at=_timestamp(),
            records=tuple(self._records[call_id] for call_id in sorted(self._records)),
        )
        await self._state_store.save(snapshot)

    def _handle_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            LOGGER.error(
                "Unexpected dispatcher failure",
                exc_info=(type(error), error, error.__traceback__),
            )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
