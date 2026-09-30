"""Provider-neutral nonstreaming model contract."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelReply:
    public_text: str | None
    tool_calls: tuple[ModelToolCall, ...]
    finish_reason: str
    usage: ModelUsage | None
    provider_request_id: str | None
    # UTF-8 bytes of discarded truncated tool content; no executable calls retained.
    discarded_tool_call_bytes: int = 0


class ModelAdapter(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        output_limit: int,
        request_deadline: float,
    ) -> ModelReply:
        """Complete once; deadline is an absolute asyncio loop monotonic time."""
        ...
