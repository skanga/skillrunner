"""Opt-in presentation hook; library callers remain silent."""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_enabled: ContextVar[float | None] = ContextVar("progress_started", default=None)


@contextmanager
def reporting(enabled: bool) -> Iterator[None]:
    token = _enabled.set(time.monotonic() if enabled else None)
    try:
        yield
    finally:
        _enabled.reset(token)


def emit(name: str, payload: dict[str, Any]) -> None:
    started = _enabled.get()
    if started is None:
        return
    if name not in {
        "phase",
        "model_request_started",
        "model_request_completed",
        "model_retry_scheduled",
        "tool_started",
        "tool_completed",
        "tool_failed",
        "waiting",
    }:
        return
    # Never render prompts, paths, arguments, credential references, or arbitrary result text.
    labels = []
    for key in (
        "phase",
        "attempt",
        "input_estimate",
        "output_limit",
        "retry_number",
        "model_attempts",
        "charged_tokens",
        "charged_tool_calls",
        "remaining_seconds",
    ):
        if key in payload:
            value = payload[key]
            if isinstance(value, (int, float)) or (
                key == "phase" and isinstance(value, str) and value.isalpha()
            ):
                labels.append(f"{key}={value}")
    print(
        f"[{time.monotonic() - started:.1f}s] {name} {' '.join(labels)}".rstrip(), file=sys.stderr
    )
