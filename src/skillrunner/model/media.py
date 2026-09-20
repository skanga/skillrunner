"""Explicit PNG/high-detail contract; payload bytes never belong in diagnostics."""

import base64
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any

from skillrunner.domain.errors import RunnerError

IMAGE_ACCOUNTING = "openai-patch-high-v1"


def image_tokens(width: int, height: int) -> int:
    """Published 32px patches, high detail, 1.2 multiplier, plus rounding allowance."""
    if any(type(n) is not int or not 0 < n < 2**31 for n in (width, height)):
        raise RunnerError("unsupported_capability", "Invalid PNG dimensions.")
    scale = min(1.0, 2048 / max(width, height))
    width, height = max(1, int(width * scale)), max(1, int(height * scale))
    if math.ceil(width / 32) * math.ceil(height / 32) > 2500:
        shrink = math.sqrt(32**2 * 2500 / (width * height))
        shrink *= min(
            math.floor(width * shrink / 32) / (width * shrink / 32),
            math.floor(height * shrink / 32) / (height * shrink / 32),
        )
        width, height = max(1, int(width * shrink)), max(1, int(height * shrink))
    patches = math.ceil(width / 32) * math.ceil(height / 32)
    return (patches * 6 + 4) // 5 + 1


def png_dimensions(data: bytes) -> tuple[int, int]:
    """Extract dimensions only; actual format validation is an external prerequisite."""
    if len(data) < 29 or data[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
        raise RunnerError("unsupported_capability", "Only validated PNG images are supported.")
    width, height = struct.unpack(">II", data[16:24])
    image_tokens(width, height)
    return width, height


@dataclass(frozen=True)
class MediaAttachment:
    data: bytes = field(repr=False)
    path: str
    width: int
    height: int

    @property
    def tokens(self) -> int:
        return image_tokens(self.width, self.height)

    def metadata(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "media_type": "image/png",
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "size_bytes": len(self.data),
            "width": self.width,
            "height": self.height,
            "detail": "high",
            "image_accounting": IMAGE_ACCOUNTING,
            "image_token_estimate": self.tokens,
        }

    def content(self, *, diagnostic: bool = False) -> list[dict[str, Any]]:
        metadata = self.metadata()
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": "Runner-provided image (data): " + json.dumps(metadata)}
        ]
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "[IMAGE_BYTES_OMITTED]"
                    if diagnostic
                    else "data:image/png;base64," + base64.b64encode(self.data).decode("ascii"),
                    "detail": "high",
                },
            }
        )
        return parts
