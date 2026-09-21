"""Sequential public model/tool loop; completion remains a coordinator proposal."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from skillrunner.domain.errors import RunnerError
from skillrunner.model.protocol import ModelAdapter, ModelReply, ModelToolCall
from skillrunner.runtime.budgets import Deadline, UsageLedger
from skillrunner.runtime.context import RunContext


class Dispatcher(Protocol):
    def model_schemas(self) -> list[dict[str, Any]]: ...

    async def dispatch_batch(
        self,
        calls: Sequence[ModelToolCall],
        ledger: UsageLedger,
        deadline: Deadline,
        *,
        on_result: Callable[[Any], Awaitable[None]] | None = None,
    ) -> list[Any]: ...


def returned_output_estimate(reply: ModelReply) -> int:
    """Public text and complete tool-call content, explicitly an estimate."""
    value = {
        "content": reply.public_text,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in reply.tool_calls
        ],
    }
    return (
        len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        + reply.discarded_tool_call_bytes
    )


class AgentLoop:
    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        context: RunContext,
        dispatcher: Dispatcher,
        ledger: UsageLedger,
        deadline: Deadline,
        context_window: int,
        max_output: int,
        model_transport_retries: int = 1,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.adapter = adapter
        self.context = context
        self.dispatcher = dispatcher
        self.ledger = ledger
        self.deadline = deadline
        self.context_window = context_window
        self.max_output = max_output
        self.model_transport_retries = model_transport_retries
        self.on_event = on_event or (lambda name, payload: None)
        self.last_reply: ModelReply | None = None

    async def _record_result(self, result: Any) -> None:
        self.context.append_tool_result(result.call_id, result.as_dict())
        self.on_event(
            "tool_completed" if result.ok else "tool_failed",
            {
                "call_id": result.call_id,
                "name": result.name,
                "executed": result.executed,
                "ok": result.ok,
                **({"error_code": result.error.get("code")} if result.error else {}),
                "_content": result.as_dict(),
            },
        )

    async def run(self) -> dict[str, Any]:
        retries_used = 0
        while True:
            await asyncio.sleep(0)
            self.deadline.check()
            schemas = self.dispatcher.model_schemas()
            estimate = self.context.estimate(schemas)
            reservation = self.ledger.reserve_model(
                input_tokens=estimate,
                context_window=self.context_window,
                max_output=self.max_output,
            )
            started = False
            timeout = asyncio.timeout(self.deadline.remaining)
            try:
                self.on_event(
                    "model_request_started",
                    {
                        "attempt": reservation.attempt,
                        "input_estimate": estimate,
                        "output_limit": reservation.output_limit,
                        "estimate_basis": "utf8_bytes+image_contract"
                        if self.context.image_token_estimate
                        else "utf8_bytes",
                        "image_token_estimate": self.context.image_token_estimate,
                        "_content": {
                            "messages": self.context.diagnostic_messages(),
                            "tools": schemas,
                        },
                    },
                )
                messages = self.context.messages()
                async with timeout:
                    started = True
                    reply = await self.adapter.complete(
                        messages,
                        schemas,
                        reservation.output_limit,
                        asyncio.get_running_loop().time() + self.deadline.remaining,
                    )
            except BaseException as exc:
                failure: BaseException = exc
                if isinstance(exc, TimeoutError) and timeout.expired():
                    failure = RunnerError("budget_exhausted", "Execution timeout reached.")
                if started:
                    self.ledger.charge_unknown(reservation)
                    if isinstance(failure, RunnerError):
                        code = failure.code
                    elif isinstance(failure, asyncio.CancelledError):
                        code = "cancelled"
                    else:
                        code = "execution_failed"
                    try:
                        self.on_event(
                            "model_request_failed",
                            {
                                "attempt": reservation.attempt,
                                "error_code": code,
                                "measurement_quality": "unknown",
                                "actual_usage": None,
                                "budget_charge": reservation.total,
                            },
                        )
                    except Exception:
                        message = "Could not persist the model-request failure event."
                        failure.add_note(message)
                        if isinstance(failure, RunnerError):
                            failure.details.setdefault("reporting_errors", []).append(message)
                    if (
                        isinstance(failure, RunnerError)
                        and failure.details.get("retryable") is True
                        and retries_used < self.model_transport_retries
                    ):
                        retries_used += 1
                        retry_delay = min(2.0 ** min(retries_used, 3), self.deadline.remaining / 2)
                        self.on_event(
                            "model_retry_scheduled",
                            {
                                "failed_attempt": reservation.attempt,
                                "retry_number": retries_used,
                                "reason": failure.code,
                                "delay_seconds": retry_delay,
                            },
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                else:
                    self.ledger.release_unstarted_model(reservation)
                if failure is not exc:
                    raise failure from exc
                raise
            if reply.usage is None:
                self.ledger.reconcile(
                    reservation,
                    input_tokens=estimate,
                    output_tokens=returned_output_estimate(reply),
                    quality="estimated",
                )
            else:
                self.ledger.reconcile(
                    reservation,
                    input_tokens=reply.usage.input_tokens,
                    output_tokens=reply.usage.output_tokens,
                    quality="reported",
                )
            self.last_reply = reply
            # A successful response completes this model turn. A later turn gets
            # its own configured retry allowance, even after earlier failures.
            retries_used = 0
            self.on_event(
                "model_request_completed",
                {
                    "attempt": reservation.attempt,
                    "finish_reason": reply.finish_reason,
                    "provider_request_id": reply.provider_request_id,
                    "measurement_quality": self.ledger.records[-1].quality,
                    "input_tokens": self.ledger.records[-1].input_tokens,
                    "output_tokens": self.ledger.records[-1].output_tokens,
                    "_content": {
                        "public_text": reply.public_text,
                        "tool_calls": [
                            {"id": call.id, "name": call.name, "arguments": call.arguments}
                            for call in reply.tool_calls
                        ],
                    },
                },
            )
            if reply.finish_reason == "length":
                raise RunnerError(
                    "budget_exhausted",
                    "The model reached this request's output-token allowance. "
                    "Partial outputs are preserved; no returned tools were dispatched.",
                )
            self.context.append_public_reply(reply.public_text, reply.tool_calls)
            await asyncio.sleep(0)
            self.deadline.check()
            if not reply.tool_calls:
                self.context.append_correction(
                    "Submit an explicit finish_run proposal when finished, or invoke the "
                    "tools needed to continue. Public text alone does not complete the run."
                )
                continue
            results = await self.dispatcher.dispatch_batch(
                reply.tool_calls,
                self.ledger,
                self.deadline,
                on_result=self._record_result,
            )
            self.context.append_media(
                [
                    result.attachment
                    for result in results
                    if result.ok and getattr(result, "attachment", None) is not None
                ]
            )
            for result in results:
                if result.name == "finish_run" and result.ok:
                    if not isinstance(result.value, dict):
                        raise RunnerError("model_protocol_error", "Invalid completion proposal.")
                    return result.value
