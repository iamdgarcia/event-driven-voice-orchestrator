"""In-memory conversation repository (demos, tests). Swap for your durable store."""

from __future__ import annotations

from eventorchestrator.domain.models import ConversationState
from eventorchestrator.domain.ports import ConversationRepositoryPort


class InMemoryConversationRepository(ConversationRepositoryPort):
    def __init__(self) -> None:
        self._store: dict[str, ConversationState] = {}

    async def load(self, conversation_id: str) -> ConversationState:
        state = self._store.get(conversation_id)
        if state is None:
            return ConversationState()
        # deep-enough copy so callers can't mutate a stored snapshot in place
        import copy

        return copy.deepcopy(state)

    async def save(self, conversation_id: str, state: ConversationState) -> None:
        import copy

        self._store[conversation_id] = copy.deepcopy(state)
