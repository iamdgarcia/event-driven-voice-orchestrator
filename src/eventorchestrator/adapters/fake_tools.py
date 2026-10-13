"""Toy tools with configurable (simulated) RTT — the latency knob of demos/benches."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence

from eventorchestrator.domain.models import (
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
)
from eventorchestrator.domain.ports import ToolRegistryPort

LOOKUP_ORDER = ToolDefinition(
    name="lookup_order_status",
    description="Look up the fulfillment status of one order id.",
    parameters={"type": "object", "properties": {"order_id": {"type": "string"}}},
)
CHECK_WEATHER = ToolDefinition(
    name="check_weather",
    description="Check the weather for one city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


class ToyToolRegistry(ToolRegistryPort):
    """Two fake tools; execute() sleeps rtt_s to model a real backend round-trip."""

    def __init__(
        self, rtt_s: float = 1.0, system_prompt: str = "You are a helpful assistant."
    ) -> None:
        self.rtt_s = rtt_s
        self._prompt = system_prompt
        self._tools = {LOOKUP_ORDER.name: LOOKUP_ORDER, CHECK_WEATHER.name: CHECK_WEATHER}
        self.executed: list[str] = []

    def active_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def system_prompt(self) -> str:
        return self._prompt

    def definition(self, tool_name: str) -> ToolDefinition:
        return self._tools[tool_name]

    async def execute(self, call: ToolCall) -> ToolExecutionResult:
        self.executed.append(call.name)
        await asyncio.sleep(self.rtt_s)
        if call.name == LOOKUP_ORDER.name:
            order = str(call.arguments.get("order_id", "unknown"))
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                content={"order_id": order, "status": "shipped", "eta_days": 2},
            )
        if call.name == CHECK_WEATHER.name:
            city = str(call.arguments.get("city", "unknown"))
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                content={"city": city, "forecast": "sunny", "temp_c": 21},
            )
        return ToolExecutionResult(
            call_id=call.call_id or uuid.uuid4().hex,
            name=call.name,
            content={"error": f"unknown tool {call.name}"},
            error=f"unknown tool {call.name}",
        )


def calls(names_ids: Sequence[tuple[str, str]], **kwargs_per_name: dict) -> tuple:
    """Build an AssistantDecision requesting several tool calls in one batch."""
    from eventorchestrator.domain.models import AssistantDecision

    return AssistantDecision(
        content="",
        tool_calls=tuple(
            ToolCall(call_id=cid, name=name, arguments=kwargs_per_name.get(name, {}))
            for name, cid in names_ids
        ),
    )
