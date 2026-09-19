"""Provider-neutral model requests and compatible endpoint implementation."""

from skillrunner.model.openai_compatible import OpenAICompatibleAdapter, ResolvedCapabilities
from skillrunner.model.protocol import ModelAdapter, ModelReply, ModelToolCall, ModelUsage

__all__ = [
    "ModelAdapter",
    "ModelReply",
    "ModelToolCall",
    "ModelUsage",
    "OpenAICompatibleAdapter",
    "ResolvedCapabilities",
]
