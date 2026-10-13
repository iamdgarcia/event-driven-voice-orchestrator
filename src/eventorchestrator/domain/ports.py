"""Ports: the seams the orchestrator depends on. Implement these to plug in
your real model, tools, transport, and storage."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from eventorchestrator.domain.models import (
    AssistantDecision,
    ConversationMessage,
    ConversationState,
    PendingToolCall,
    ToolCall,
    ToolDefinition,
    ToolDispatcherSnapshot,
    ToolExecutionResult,
)


@dataclass(frozen=True)
class TextDelta:
    """One visible text chunk streamed by the model."""

    text: str


@dataclass(frozen=True)
class DecisionCompleted:
    """Final structured assistant decision from the model."""

    decision: AssistantDecision


ModelStreamEvent = TextDelta | DecisionCompleted


class ModelPort(Protocol):
    """Stream assistant decisions without owning conversation state."""

    def stream(
        self,
        *,
        system_prompt: str,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream one model invocation."""
        ...


class ToolRegistryPort(Protocol):
    """Expose tools to the model and execute them (possibly with real latency)."""

    def active_tools(self) -> list[ToolDefinition]:
        """Return the tools currently visible to the model."""
        ...

    def system_prompt(self) -> str:
        """Return the base system prompt (tool instructions included)."""
        ...

    def definition(self, tool_name: str) -> ToolDefinition:
        """Return one registered tool definition."""
        ...

    async def execute(self, call: ToolCall) -> ToolExecutionResult:
        """Execute one tool call and normalize its result."""
        ...


class OutputPort(Protocol):
    """Publish transport-neutral assistant and tool lifecycle events.

    A transport adapter turns these into terminal prints, Pipecat frames,
    websocket messages, etc. — and reports playback back as delivery events.
    """

    async def start_response(self, response_id: str, turn_id: str) -> None:
        """Start one visible assistant response."""
        ...

    async def text_delta(self, response_id: str, text: str) -> None:
        """Publish one assistant text chunk."""
        ...

    async def end_response(self, response_id: str) -> None:
        """Finish the active assistant response."""
        ...

    async def cancel_response(self, response_id: str) -> None:
        """Cancel the active assistant response (barge-in)."""
        ...

    async def tool_started(self, pending_call: PendingToolCall) -> None:
        """Publish one background tool start."""
        ...

    async def tool_finished(
        self,
        pending_call: PendingToolCall,
        result: ToolExecutionResult,
    ) -> None:
        """Publish one background tool completion."""
        ...


class ConversationRepositoryPort(Protocol):
    """Persist the canonical conversation state."""

    async def load(self, conversation_id: str) -> ConversationState:
        """Load one conversation or return an empty state."""
        ...

    async def save(self, conversation_id: str, state: ConversationState) -> None:
        """Persist one conversation snapshot."""
        ...


class ToolDispatcherStatePort(Protocol):
    """Persist the current observable state of the tool dispatcher."""

    async def save(self, snapshot: ToolDispatcherSnapshot) -> None:
        """Persist one complete dispatcher snapshot."""
        ...


@dataclass(frozen=True)
class NoopDispatcherState:
    """Drop dispatcher snapshots (tests, demos)."""

    async def save(self, snapshot: ToolDispatcherSnapshot) -> None:
        pass
