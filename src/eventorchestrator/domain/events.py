"""Events consumed by the orchestrator's event loop.

Every input to the conversation — a user turn, a background tool completion,
a transport playback report — enters as one of these generic events. The
orchestrator never inspects wording or business intent to mutate pending work.
"""

from __future__ import annotations

from dataclasses import dataclass

from eventorchestrator.domain.models import ToolExecutionResult


@dataclass(frozen=True)
class UserTextReceived:
    """One user turn received from any transport."""

    text: str


@dataclass(frozen=True)
class ToolCallFinished:
    """One background tool execution has finished."""

    result: ToolExecutionResult


@dataclass(frozen=True)
class OutputDeliveryStarted:
    """The transport started playing one assistant response."""

    response_id: str


@dataclass(frozen=True)
class OutputDeliveryCompleted:
    """The transport finished playing one assistant response."""

    response_id: str


@dataclass(frozen=True)
class OutputDeliveryInterrupted:
    """The transport stopped one response before playback completed (barge-in)."""

    response_id: str
    reason: str


ConversationEvent = (
    UserTextReceived
    | ToolCallFinished
    | OutputDeliveryStarted
    | OutputDeliveryCompleted
    | OutputDeliveryInterrupted
)
