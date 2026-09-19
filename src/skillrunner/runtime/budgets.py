"""Per-run reservations; budget charges are distinct from actual provider usage."""

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from skillrunner.domain.errors import RunnerError

MeasurementQuality = Literal["reported", "estimated", "unknown"]


def _integer(value: int, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ModelReservation:
    attempt: int
    input_estimate: int
    output_limit: int

    @property
    def total(self) -> int:
        return self.input_estimate + self.output_limit


@dataclass(frozen=True)
class UsageRecord:
    attempt: int
    input_tokens: int
    output_tokens: int
    quality: MeasurementQuality


@dataclass(frozen=True)
class ToolBatch:
    number: int
    count: int


class UsageLedger:
    """Owns reservations for one sequential agent loop, never shared between runs."""

    def __init__(self, *, max_steps: int, max_tool_calls: int, max_tokens: int) -> None:
        for name, value in locals().copy().items():
            if name != "self":
                _integer(value, name, minimum=1)
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.model_attempts = 0
        self.requested_tool_calls = 0
        self.charged_tool_calls = 0
        self.records: list[UsageRecord] = []
        self._requests: dict[int, ModelReservation] = {}
        self._batches: dict[int, tuple[ToolBatch, int]] = {}
        self._next_batch = 0

    @property
    def charged_tokens(self) -> int:
        return sum(record.input_tokens + record.output_tokens for record in self.records)

    @property
    def available_tokens(self) -> int:
        return max(
            0,
            self.max_tokens
            - self.charged_tokens
            - sum(request.total for request in self._requests.values()),
        )

    @property
    def available_tool_calls(self) -> int:
        outstanding = sum(remaining for _, remaining in self._batches.values())
        return self.max_tool_calls - self.charged_tool_calls - outstanding

    def reserve_model(
        self, *, input_tokens: int, context_window: int, max_output: int
    ) -> ModelReservation:
        _integer(input_tokens, "input_tokens")
        _integer(context_window, "context_window", minimum=1)
        _integer(max_output, "max_output", minimum=1)
        if self.model_attempts >= self.max_steps:
            raise RunnerError("budget_exhausted", "Model-turn limit reached.")
        if input_tokens >= context_window:
            raise RunnerError(
                "context_capacity_exceeded",
                f"Input estimate {input_tokens} leaves no output capacity "
                f"in {context_window} tokens.",
            )
        output_limit = min(
            max_output, context_window - input_tokens, self.available_tokens - input_tokens
        )
        if output_limit <= 0:
            raise RunnerError("budget_exhausted", "Aggregate token budget exhausted.")
        self.model_attempts += 1
        reservation = ModelReservation(self.model_attempts, input_tokens, output_limit)
        self._requests[reservation.attempt] = reservation
        return reservation

    def release_unstarted_model(self, reservation: ModelReservation) -> None:
        """Release the latest local admission when no adapter call was attempted."""
        if (
            self._requests.get(reservation.attempt) is not reservation
            or reservation.attempt != self.model_attempts
        ):
            raise ValueError("Only the latest outstanding unstarted admission can be released")
        del self._requests[reservation.attempt]
        self.model_attempts -= 1

    def reconcile(
        self,
        reservation: ModelReservation,
        *,
        input_tokens: int,
        output_tokens: int,
        quality: MeasurementQuality,
    ) -> None:
        _integer(input_tokens, "input_tokens")
        _integer(output_tokens, "output_tokens")
        if quality not in {"reported", "estimated", "unknown"}:
            raise ValueError("Invalid measurement quality")
        if self._requests.get(reservation.attempt) is not reservation:
            raise ValueError("Reservation is not outstanding in this ledger")
        self.records.append(UsageRecord(reservation.attempt, input_tokens, output_tokens, quality))
        del self._requests[reservation.attempt]

    def charge_unknown(self, reservation: ModelReservation) -> None:
        self.reconcile(
            reservation,
            input_tokens=reservation.input_estimate,
            output_tokens=reservation.output_limit,
            quality="unknown",
        )

    def admit_tools(self, count: int) -> ToolBatch:
        _integer(count, "count")
        self.requested_tool_calls += count
        if count > self.available_tool_calls:
            raise RunnerError(
                "budget_exhausted", "Tool batch exceeds the remaining tool-call budget."
            )
        self._next_batch += 1
        batch = ToolBatch(self._next_batch, count)
        self._batches[batch.number] = (batch, count)
        return batch

    def charge_tool(self, batch: ToolBatch) -> None:
        state = self._batches.get(batch.number)
        if state is None or state[0] is not batch or state[1] <= 0:
            raise ValueError("No outstanding tool slot for this batch")
        self._batches[batch.number] = (batch, state[1] - 1)
        self.charged_tool_calls += 1

    def close_tools(self, batch: ToolBatch) -> None:
        state = self._batches.get(batch.number)
        if state is None or state[0] is not batch:
            raise ValueError("Batch is not outstanding in this ledger")
        del self._batches[batch.number]

    def summary(self) -> dict[str, Any]:
        return {
            "model_attempts": self.model_attempts,
            "requested_tool_calls": self.requested_tool_calls,
            "charged_tool_calls": self.charged_tool_calls,
            "charged_tokens": self.charged_tokens,
            "input_tokens": sum(record.input_tokens for record in self.records),
            "output_tokens": sum(record.output_tokens for record in self.records),
            "measurement_quality": sorted({record.quality for record in self.records}),
        }


class Deadline:
    """Execution deadline; process shutdown grace is independently supervised."""

    def __init__(self, timeout: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        self.clock = clock
        self.started_at = clock()
        self.expires_at = self.started_at + timeout

    @property
    def remaining(self) -> float:
        return max(0.0, self.expires_at - self.clock())

    def check(self) -> None:
        if self.remaining <= 0:
            raise RunnerError("budget_exhausted", "Execution timeout reached.")


def estimate_text_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    """Approved text-only fallback; not a universal tokenizer upper bound."""
    serialized = json.dumps(
        {"messages": messages, "tools": tools},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return len(serialized.encode("utf-8"))
