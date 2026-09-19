"""Append complete JSONL events, reserving space to disclose log saturation."""

import json
import os
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillrunner.domain.errors import RunnerError
from skillrunner.recording.redaction import Redactor


def _protocol_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, RecursionError):
        return "[INVALID_PROTOCOL_JSON_OMITTED]"


def _structured_content(content: dict[str, Any]) -> dict[str, Any]:
    """Expose protocol JSON to the redactor without mutating request history."""
    messages = content.get("messages")
    if not isinstance(messages, list):
        return content
    structured = []
    for message in messages:
        if not isinstance(message, dict):
            structured.append(message)
            continue
        message = dict(message)
        if message.get("role") == "tool":
            message["content"] = _protocol_json(message.get("content"))
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            structured_calls = []
            for call in calls:
                if isinstance(call, dict) and isinstance(call.get("function"), dict):
                    function = dict(call["function"])
                    function["arguments"] = _protocol_json(function.get("arguments"))
                    call = {**call, "function": function}
                structured_calls.append(call)
            message["tool_calls"] = structured_calls
        structured.append(message)
    return {**content, "messages": structured}


class EventLog:
    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        max_bytes: int,
        secrets: Iterable[str] = (),
        log_content: bool = False,
    ) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Event log limit must be a nonnegative integer")
        self.run_id = run_id
        self.max_bytes = max_bytes
        self.log_content = log_content
        self.redactor = Redactor(secrets)
        self.started = time.monotonic()
        self.sequence = 0
        self.written_bytes = 0
        self.written_events = 0
        self.omitted_bytes = 0
        self.omitted_events = 0
        self.saturated = False
        self._stream = path.open("xb")

    def _encode(
        self, event_type: str, payload: dict[str, Any], identifiers: dict[str, str] | None = None
    ) -> bytes:
        self.sequence += 1
        event = {
            "schema_version": 1,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_ms": int((time.monotonic() - self.started) * 1000),
            "event_type": event_type,
            "payload": payload,
            **(identifiers or {}),
        }
        return (
            json.dumps(
                self.redactor.clean(event),
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    def _write(self, encoded: bytes) -> None:
        try:
            self._stream.write(encoded)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except OSError as exc:
            raise RunnerError("reporting_failed", "Could not persist the event log.") from exc
        self.written_bytes += len(encoded)
        self.written_events += 1

    def emit(
        self,
        event_type: str,
        metadata: dict[str, Any],
        *,
        content: dict[str, Any] | None = None,
        skill_id: str | None = None,
        tool_name: str | None = None,
        call_id: str | None = None,
    ) -> bool:
        payload = dict(metadata)
        if self.log_content and content is not None:
            payload["content"] = _structured_content(content)
        identifiers = {
            key: value
            for key, value in (
                ("skill_id", skill_id),
                ("tool_name", tool_name),
                ("call_id", call_id),
            )
            if value is not None
        }
        encoded = self._encode(event_type, payload, identifiers)
        # Small configured limits may not fit even a truncation record. The
        # manifest still receives summary() so those omissions are observable.
        reserve = min(self.max_bytes, 512)
        if not self.saturated and self.written_bytes + len(encoded) <= self.max_bytes - reserve:
            self._write(encoded)
            return True
        self.omitted_events += 1
        self.omitted_bytes += len(encoded)
        if not self.saturated:
            self.saturated = True
            truncated = self._encode(
                "log_truncated",
                {
                    "omitted_events_at_saturation": self.omitted_events,
                    "omitted_bytes_at_saturation": self.omitted_bytes,
                    "final_counts": "run.json logging summary",
                },
            )
            if self.written_bytes + len(truncated) <= self.max_bytes:
                self._write(truncated)
        return False

    def summary(self) -> dict[str, int | bool]:
        return {
            "written_events": self.written_events,
            "written_bytes": self.written_bytes,
            "omitted_events": self.omitted_events,
            "omitted_bytes": self.omitted_bytes,
            "saturated": self.saturated,
        }

    def close(self) -> None:
        self._stream.close()
