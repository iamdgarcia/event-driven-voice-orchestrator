"""Canonical domain models for the event-driven voice orchestrator.

The orchestrator owns this state — not the transport (Pipecat, a terminal, a
websocket server). Everything else reads it through ports or receives events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

JsonObject = dict[str, object]
MessageRole = Literal["user", "assistant", "tool"]


class RequestStatus(StrEnum):
    """Lifecycle of one dispatched tool call."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class ResponseGenerationStatus(StrEnum):
    """Lifecycle of assistant response *generation* (model streaming)."""

    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ResponseDeliveryStatus(StrEnum):
    """Lifecycle of assistant response *delivery* (what the user heard).

    Generation and delivery are independent: a barge-in cancels generation
    without touching delivery history, and slow playback outlives generation.
    """

    PENDING = "pending"
    PLAYING = "playing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


class ToolDispatchStatus(StrEnum):
    """Observable execution state of one background tool call."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolCall:
    """One model-requested tool invocation."""

    call_id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True)
class ConversationMessage:
    """Provider-independent conversation message."""

    role: MessageRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class AssistantDecision:
    """Final assistant decision produced by one model invocation."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass
class AssistantResponseState:
    """Track generated text separately from its user-facing delivery."""

    response_id: str
    turn_id: str
    content: str = ""
    generation_status: ResponseGenerationStatus = ResponseGenerationStatus.STREAMING
    delivery_status: ResponseDeliveryStatus = ResponseDeliveryStatus.PENDING
    interruption_reason: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """Tool metadata exposed to the model and dispatcher."""

    name: str
    description: str
    parameters: JsonObject


@dataclass
class PendingToolCall:
    """Runtime lifecycle data for one background tool call."""

    batch_id: str
    turn_id: str
    call: ToolCall
    status: RequestStatus = RequestStatus.ACTIVE


@dataclass
class PendingToolBatch:
    """Assistant tool-call message retained until every result is available.

    The batch is the atomicity unit: the assistant message (with its tool
    calls) plus every tool result message are committed to canonical history
    together, or not at all. A provider never sees an unresolved tool-call.
    """

    batch_id: str
    turn_id: str
    decision: AssistantDecision
    results: dict[str, ToolExecutionResult] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Return whether every tool call in the batch has completed."""
        expected = {call.call_id for call in self.decision.tool_calls}
        return expected == set(self.results)


@dataclass(frozen=True)
class ToolExecutionResult:
    """Normalized result returned by one tool adapter."""

    call_id: str
    name: str
    content: object
    error: str | None = None


@dataclass
class ToolDispatchRecord:
    """Observable execution state for one dispatched tool call."""

    conversation_id: str
    batch_id: str
    turn_id: str
    call_id: str
    tool_name: str
    arguments: JsonObject
    status: ToolDispatchStatus
    queued_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: object | None = None
    error: str | None = None


@dataclass(frozen=True)
class ToolDispatcherSnapshot:
    """Complete point-in-time view of the tool dispatcher."""

    conversation_id: str
    updated_at: str
    records: tuple[ToolDispatchRecord, ...]


@dataclass
class ConversationState:
    """Canonical conversation state owned by the orchestrator."""

    messages: list[ConversationMessage] = field(default_factory=list)
    pending_calls: dict[str, PendingToolCall] = field(default_factory=dict)
    pending_batches: dict[str, PendingToolBatch] = field(default_factory=dict)
    responses: dict[str, AssistantResponseState] = field(default_factory=dict)

    def active_calls(self) -> list[PendingToolCall]:
        """Return currently active (not yet finished) tool calls."""
        return [
            pending
            for pending in self.pending_calls.values()
            if pending.status == RequestStatus.ACTIVE
        ]

    def latest_response(self) -> AssistantResponseState | None:
        """Return the most recently created assistant response."""
        if not self.responses:
            return None
        return next(reversed(self.responses.values()))
