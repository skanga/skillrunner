"""Remove known credentials and declared private reasoning from operational data."""

import math
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_PRIVATE = {
    "reasoning",
    "reasoning_content",
    "chain_of_thought",
    "private_reasoning",
    "analysis",
    "thinking",
    "redacted_thinking",
    "reasoning_summary",
    "encrypted_content",
}
_SECRET_FIELDS = {
    "authorization",
    "proxy-authorization",
    "api_key",
    "apikey",
    "password",
    "access_token",
    "refresh_token",
    "cookie",
    "set-cookie",
    "x-api-key",
}


class Redactor:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = tuple(
            sorted({secret for secret in secrets if secret}, key=len, reverse=True)
        )

    def text(self, value: str) -> str:
        for secret in self._secrets:
            value = value.replace(secret, "[REDACTED]")

        def url(match: re.Match[str]) -> str:
            try:
                parsed = urlsplit(match[0])
                authority = parsed.netloc.rsplit("@", 1)[-1]
                return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))
            except ValueError:
                return "[REDACTED_URL]"

        value = re.sub(r"https?://[^\s<>]+", url, value, flags=re.IGNORECASE)
        return re.sub(
            r"(?i)\b((?:Proxy-)?Authorization\s*:\s*(?:Bearer|Basic))\s+[^\s,;]+",
            r"\1 [REDACTED]",
            value,
        )

    def clean(self, value: Any, *, _depth: int = 0) -> Any:
        if _depth > 32:
            return "[DEPTH_LIMIT]"
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            if any(
                isinstance(value.get(key), str) and value[key].lower() in _PRIVATE
                for key in ("type", "channel")
            ):
                return "[PRIVATE_CONTENT_OMITTED]"
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or key.lower() in _PRIVATE:
                    continue
                result[self.text(key)] = (
                    "[REDACTED]"
                    if key.lower() in _SECRET_FIELDS
                    else self.clean(item, _depth=_depth + 1)
                )
            return result
        if isinstance(value, (list, tuple)):
            return [self.clean(item, _depth=_depth + 1) for item in value]
        if value is None or type(value) in {int, bool}:
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        return "[UNSUPPORTED_VALUE]"
