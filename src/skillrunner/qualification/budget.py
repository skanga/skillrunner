"""Crash-conservative cumulative accounting for explicitly authorized paid tests."""

import json
import os
import uuid
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from skillrunner.recording.bundle import atomic_write

AUTHORIZED_BUDGET_USD = Decimal("200")


def amount(value: str | Decimal) -> Decimal:
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError("Spending amounts must be finite and nonnegative")
    return result


class SpendingLedger:
    """Requires an existing ledger; an interrupted reservation stays charged."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        if self.path.stat().st_nlink != 1:
            raise ValueError("hard-linked spending ledgers cannot preserve a single history")
        self.lock = self.path.with_suffix(self.path.suffix + ".lock")
        self.state: dict[str, Any] = {}
        self.open = False

    def __enter__(self) -> Self:
        descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "w") as writer:
                writer.write(str(os.getpid()))
            self.state = json.loads(self.path.read_text())
            total = sum((amount(item["charge_usd"]) for item in self.state["requests"]), Decimal(0))
            if total != amount(self.state["charged_usd"]):
                raise ValueError("Spending ledger is inconsistent")
            amount(self.state["budget_usd"])
            self.open = True
            return self
        except BaseException:
            self.lock.unlink()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.open = False
        self.lock.unlink()

    def require_authorized_ceiling(self) -> None:
        if not self.open:
            raise ValueError("Open the spending ledger before checking its ceiling")
        if amount(self.state["budget_usd"]) > AUTHORIZED_BUDGET_USD:
            raise ValueError("Ledger exceeds the user-authorized $200 ceiling")

    def _save(self) -> None:
        durable = atomic_write(
            self.path, (json.dumps(self.state, indent=2, allow_nan=False) + "\n").encode()
        )
        if not durable:
            raise OSError("Paid qualification requires directory-fsync durability")

    def reserve(self, model: str, maximum: Decimal) -> str:
        if not self.open:
            raise ValueError("Open the spending ledger before use")
        if self.state.get("halted"):
            raise ValueError("Qualification spending is halted after a reservation overrun")
        maximum = amount(maximum)
        total = amount(self.state["charged_usd"]) + maximum
        if total > amount(self.state["budget_usd"]):
            raise ValueError("Request reservation exceeds the cumulative spending ceiling")
        request_id = uuid.uuid4().hex
        self.state["requests"].append(
            {
                "id": request_id,
                "model": model,
                "reserved_usd": str(maximum),
                "charge_usd": str(maximum),
                "status": "reserved_outcome_unknown",
            }
        )
        self.state["charged_usd"] = str(total)
        self._save()  # No network request is admitted before this durable write.
        return request_id

    def reconcile(self, request_id: str, charge: Decimal, usage: dict[str, Any]) -> None:
        if not self.open:
            raise ValueError("Open the spending ledger before use")
        charge = amount(charge)
        record = next(item for item in self.state["requests"] if item.get("id") == request_id)
        if record["status"] != "reserved_outcome_unknown":
            raise ValueError("Request is already reconciled")
        previous = amount(record["charge_usd"])
        record.update(charge_usd=str(charge), status="reported_usage", usage=usage)
        self.state["charged_usd"] = str(amount(self.state["charged_usd"]) - previous + charge)
        if charge > previous:
            self.state["halted"] = True
        self._save()
        if charge > previous:
            raise ValueError("Reported usage exceeded the reservation; qualification must stop")


class StandardRates:
    """Conservative published text rates, including cache-write/long-context uplifts."""

    def __init__(self, input_per_million: Decimal, output_per_million: Decimal) -> None:
        self.input = amount(input_per_million)
        self.output = amount(output_per_million)

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be nonnegative")
        long = input_tokens > 272_000
        # Do not assume cache hits; allow the full cache-write surcharge on input.
        input_rate = self.input * Decimal("1.25") * (2 if long else 1)
        output_rate = self.output * (Decimal("1.5") if long else 1)
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class PaidAdapter:
    """Reserve the entire model context, avoiding guessed tokenizer cost bounds."""

    def __init__(
        self,
        adapter: Any,
        profile: Any,
        ledger: SpendingLedger,
        rates: StandardRates,
        *,
        max_output: int = 2048,
    ) -> None:
        if max_output < 1:
            raise ValueError("Output cap must be positive")
        self.adapter, self.profile, self.ledger, self.rates = adapter, profile, ledger, rates
        self.max_output = max_output

    async def discover_capabilities(self, deadline: float) -> Any:
        return await self.adapter.discover_capabilities(deadline)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        output_limit: int,
        request_deadline: float,
    ) -> Any:
        from dataclasses import asdict

        from skillrunner.domain.errors import RunnerError

        capacity = self.profile.context_window_tokens
        if capacity is None:
            raise RunnerError("invalid_configuration", "Qualification needs model capacity.")
        output_limit = min(output_limit, self.max_output)
        maximum = self.rates.cost(capacity, output_limit)
        try:
            request_id = self.ledger.reserve(self.profile.model, maximum)
        except ValueError as error:
            raise RunnerError(
                "budget_exhausted", "Qualification spending ceiling reached."
            ) from error
        reply = await self.adapter.complete(messages, tool_schemas, output_limit, request_deadline)
        if reply.usage is not None:
            charge = self.rates.cost(reply.usage.input_tokens, reply.usage.output_tokens)
            self.ledger.reconcile(request_id, charge, asdict(reply.usage))
        # Missing/failed response usage keeps the full durable reservation charged.
        return reply

    async def aclose(self) -> None:
        await self.adapter.aclose()
