import asyncio
import importlib
import json
from dataclasses import dataclass

import pytest

from skillrunner.domain.errors import RunnerError
from skillrunner.model.protocol import ModelReply, ModelToolCall, ModelUsage
from skillrunner.runtime.budgets import Deadline, UsageLedger
from skillrunner.runtime.context import RunContext


def api():
    assert importlib.util.find_spec("skillrunner.runtime.agent") is not None
    return importlib.import_module("skillrunner.runtime.agent")


def reply(name="finish_run", usage=None):
    return ModelReply(
        "public",
        (ModelToolCall("call", name, {"report": "done"}, '{"report":"done"}'),),
        "tool_calls",
        usage,
        "request",
    )


class Adapter:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, messages, tool_schemas, output_limit, request_deadline):
        self.requests.append((messages, output_limit))
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return response


@dataclass
class Result:
    name: str
    call_id: str
    value: dict
    ok: bool = True
    executed: bool = True
    error: dict | None = None

    def as_dict(self):
        return {"ok": self.ok, "value": self.value}


class Dispatcher:
    def model_schemas(self):
        return []

    async def dispatch_batch(self, calls, ledger, deadline, *, on_result=None):
        batch = ledger.admit_tools(len(calls))
        results = []
        try:
            for call in calls:
                ledger.charge_tool(batch)
                result = Result(call.name, call.id, call.arguments)
                results.append(result)
                if on_result:
                    await on_result(result)
        finally:
            ledger.close_tools(batch)
        return results


def setup(replies, *, steps=3, tokens=10000, context_window=20000):
    adapter = Adapter(replies)
    context = RunContext(runner_instructions="rules", prompt="task", catalog=[])
    ledger = UsageLedger(max_steps=steps, max_tool_calls=10, max_tokens=tokens)
    loop = api().AgentLoop(
        adapter=adapter,
        context=context,
        dispatcher=Dispatcher(),
        ledger=ledger,
        deadline=Deadline(5),
        context_window=context_window,
        max_output=2000,
    )
    return loop, adapter, context, ledger


async def test_final_turn_completion_needs_no_extra_model_request():
    loop, adapter, context, ledger = setup([reply("read_text"), reply()], steps=2)
    assert await loop.run() == {"report": "done"}
    assert len(adapter.requests) == 2
    assert ledger.model_attempts == 2
    assert ledger.charged_tool_calls == 2
    assert adapter.requests[1][0][-1]["role"] == "tool"
    assert context.messages()[-1]["role"] == "tool"


async def test_missing_usage_estimates_returned_content_instead_of_full_reservation():
    loop, adapter, _, ledger = setup([reply()])
    await loop.run()
    assert ledger.records[0].quality == "estimated"
    assert ledger.records[0].input_tokens > 0
    assert len("public") < ledger.records[0].output_tokens < 2000
    assert ledger.available_tokens > 7000


async def test_failed_request_charges_unknown_reservation_once():
    loop, _, _, ledger = setup([RunnerError("model_connection_failed", "Unavailable.")])
    with pytest.raises(RunnerError, match="model_connection_failed"):
        await loop.run()
    assert ledger.records[0].quality == "unknown"
    assert ledger.records[0].output_tokens == 2000
    assert ledger.model_attempts == 1


async def test_retryable_transport_failure_gets_one_new_attempt_without_tools(monkeypatch):
    failure = RunnerError(
        "model_transport_error",
        "Unavailable.",
        details={"retryable": True},
    )
    loop, adapter, _, ledger = setup([failure, reply()])
    events = []
    delays = []
    original_sleep = asyncio.sleep

    async def record_sleep(delay):
        delays.append(delay)
        await original_sleep(0)

    monkeypatch.setattr(api().asyncio, "sleep", record_sleep)
    loop.on_event = lambda name, payload: events.append((name, payload))

    assert await loop.run() == {"report": "done"}
    assert len(adapter.requests) == ledger.model_attempts == 2
    assert [name for name, _ in events] == [
        "model_request_started",
        "model_request_failed",
        "model_retry_scheduled",
        "model_request_started",
        "model_request_completed",
        "tool_completed",
    ]
    assert events[2][1]["retry_number"] == 1
    assert any(delay >= 1.0 for delay in delays)


async def test_retry_allowance_resets_after_successful_model_turn():
    failure = RunnerError("model_transport_error", "Unavailable.", details={"retryable": True})
    loop, adapter, _, ledger = setup([failure, reply("read_text"), failure, reply()], steps=4)
    events = []
    loop.on_event = lambda name, payload: events.append((name, payload))

    assert await loop.run() == {"report": "done"}
    assert len(adapter.requests) == ledger.model_attempts == 4
    assert [
        payload["retry_number"] for name, payload in events if name == "model_retry_scheduled"
    ] == [1, 1]
    assert [record.quality for record in ledger.records] == [
        "unknown",
        "estimated",
        "unknown",
        "estimated",
    ]


async def test_context_overflow_stops_before_request():
    loop, adapter, _, ledger = setup([reply()], context_window=10)
    with pytest.raises(RunnerError, match="context_capacity_exceeded"):
        await loop.run()
    assert not adapter.requests
    assert ledger.model_attempts == 0


async def test_cancellation_reconciles_outstanding_model_attempt():
    loop, _, _, ledger = setup([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await loop.run()
    assert len(ledger.records) == 1
    assert ledger.records[0].quality == "unknown"


async def test_reported_usage_can_exhaust_budget_on_final_completion():
    loop, _, _, ledger = setup([reply(usage=ModelUsage(8000, 3000, 11000))])
    assert await loop.run() == {"report": "done"}
    assert ledger.charged_tokens == 11000
    assert ledger.available_tokens == 0


async def test_plain_text_requires_explicit_completion_proposal():
    plain = ModelReply("not a validated completion", (), "stop", None, None)
    loop, adapter, _, _ = setup([plain, reply()])
    assert await loop.run() == {"report": "done"}
    assert len(adapter.requests) == 2
    assert "finish_run" in adapter.requests[1][0][-1]["content"]


async def test_pre_request_journal_failure_does_not_charge_unstarted_request():
    loop, adapter, _, ledger = setup([reply()])
    events = []

    def broken(event, payload):
        events.append(event)
        raise OSError("Journal unavailable")

    loop.on_event = broken
    with pytest.raises(OSError):
        await loop.run()
    assert not adapter.requests
    assert ledger.model_attempts == 0
    assert ledger.available_tokens == ledger.max_tokens
    assert not ledger.records
    assert events == ["model_request_started"]


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (
            RunnerError("model_connection_failed", "secret transport detail"),
            "model_connection_failed",
        ),
        (asyncio.CancelledError("secret cancellation detail"), "cancelled"),
        (RuntimeError("secret provider detail"), "execution_failed"),
    ],
)
async def test_failed_model_event_records_charge_without_exception_content(failure, code):
    loop, adapter, _, ledger = setup([failure])
    events = []
    loop.on_event = lambda name, payload: events.append((name, payload))
    with pytest.raises(type(failure)) as caught:
        await loop.run()
    assert caught.value is failure
    assert len(adapter.requests) == ledger.model_attempts == 1
    assert [name for name, _ in events] == ["model_request_started", "model_request_failed"]
    failed = events[-1][1]
    assert failed["attempt"] == events[0][1]["attempt"] == 1
    assert failed["error_code"] == code
    assert failed["measurement_quality"] == "unknown"
    assert failed["actual_usage"] is None
    assert failed["budget_charge"] == ledger.charged_tokens
    assert "secret" not in json.dumps(failed)


async def test_model_timeout_event_keeps_capacity_charge_and_timeout_code():
    loop, _, _, ledger = setup([])
    entered = []

    async def hang(*args):
        entered.append(True)
        await asyncio.sleep(10)

    loop.adapter.complete = hang
    loop.deadline = Deadline(0.05)
    events = []
    loop.on_event = lambda name, payload: events.append((name, payload))
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await loop.run()
    assert entered
    assert events[-1][0] == "model_request_failed"
    assert events[-1][1]["error_code"] == "budget_exhausted"
    assert events[-1][1]["budget_charge"] == ledger.charged_tokens


async def test_failure_event_write_error_does_not_replace_model_failure():
    failure = RunnerError("model_connection_failed", "Connection lost.")
    loop, _, _, ledger = setup([failure])

    def broken(name, payload):
        if name == "model_request_failed":
            raise RunnerError("reporting_failed", "secret logger detail")

    loop.on_event = broken
    with pytest.raises(RunnerError) as caught:
        await loop.run()
    assert caught.value is failure
    assert failure.details["reporting_errors"]
    assert "secret" not in json.dumps(failure.details)
    assert ledger.model_attempts == 1
    assert ledger.records[0].quality == "unknown"


async def test_content_logging_redacts_serialized_history_without_changing_requests(tmp_path):
    from skillrunner.recording.events import EventLog

    arguments = {
        "reasoning_content": "private reasoning marker",
        "Authorization": "opaque credential marker",
        "public": "public observation",
    }
    response = ModelReply(
        None,
        (ModelToolCall("read", "read_text", arguments, json.dumps(arguments)),),
        "tool_calls",
        None,
        None,
    )
    loop, adapter, _, _ = setup([response, reply()])
    log = EventLog(tmp_path / "events.jsonl", run_id="run", max_bytes=100_000, log_content=True)

    def record(name, payload):
        log.emit(
            name,
            {key: value for key, value in payload.items() if key != "_content"},
            content=payload.get("_content"),
        )

    loop.on_event = record
    try:
        await loop.run()
    finally:
        log.close()
    history = json.dumps(adapter.requests[1][0])
    assert "private reasoning marker" in history
    assert "opaque credential marker" in history
    events = (tmp_path / "events.jsonl").read_text()
    assert "public observation" in events
    assert "private reasoning marker" not in events
    assert "opaque credential marker" not in events


@pytest.mark.parametrize("content", [None, "", "Partial answer"])
@pytest.mark.parametrize("usage", [None, ModelUsage(100, 2000, 2100)])
@pytest.mark.parametrize("calls", [(), reply().tool_calls])
async def test_length_response_stops_after_accounting_without_dispatch(content, usage, calls):
    response = ModelReply(content, calls, "length", usage, "request")
    loop, adapter, context, ledger = setup([response])
    with pytest.raises(RunnerError, match="output-token") as caught:
        await loop.run()
    assert caught.value.code == "budget_exhausted"
    assert len(adapter.requests) == 1
    assert ledger.charged_tool_calls == 0
    assert len(ledger.records) == 1
    assert ledger.records[0].quality == ("reported" if usage else "estimated")
    if usage:
        assert ledger.records[0].output_tokens == 2000
    else:
        assert 0 < ledger.records[0].output_tokens < 2000
        assert ledger.records[0].output_tokens >= len((content or "").encode("utf-8"))
    assert loop.last_reply == response


def empty_completion_error():
    from skillrunner.model.openai_compatible import _normalize

    with pytest.raises(RunnerError) as exc:
        _normalize(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": None}}
                ]
            },
            None,
        )
    return exc.value


async def test_empty_retry_records_field_and_never_replays_tools():
    loop, adapter, _, ledger = setup([reply("read_text"), empty_completion_error(), reply()])
    events = []
    loop.on_event = lambda name, payload: events.append((name, payload))
    assert await loop.run() == {"report": "done"}
    assert ledger.model_attempts == 3
    assert ledger.charged_tool_calls == 2
    assert adapter.requests[1] == adapter.requests[2]
    failure = next(p for name, p in events if name == "model_request_failed")
    assert failure["response_field"] == "completion_consistency"
    assert failure["measurement_quality"] == "unknown"
    assert ledger.records[1].output_tokens == 2000


@pytest.mark.parametrize("retry_count,first_transport", [(0, False), (1, False), (1, True)])
async def test_empty_retry_uses_single_shared_allowance(retry_count, first_transport):
    first = (
        RunnerError("model_transport_error", "Unavailable", details={"retryable": True})
        if first_transport
        else empty_completion_error()
    )
    loop, adapter, _, ledger = setup([first, empty_completion_error(), reply()])
    loop.model_transport_retries = retry_count
    with pytest.raises(RunnerError, match="model_protocol_error"):
        await loop.run()
    assert len(adapter.requests) == 1 + retry_count
    assert ledger.charged_tool_calls == 0


@pytest.mark.parametrize("limit", ["steps", "tokens"])
async def test_empty_retry_cannot_exceed_budget(limit):
    loop, adapter, context, ledger = setup(
        [empty_completion_error(), reply()], steps=1 if limit == "steps" else 3
    )
    if limit == "tokens":
        ledger.max_tokens = context.estimate([]) + loop.max_output
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await loop.run()
    assert len(adapter.requests) == 1
    assert ledger.charged_tool_calls == 0


async def test_empty_retry_respects_deadline(monkeypatch):
    loop, adapter, _, ledger = setup([empty_completion_error(), reply()])
    loop.deadline = Deadline(0.02)
    original_sleep = asyncio.sleep

    async def expire_deadline(delay):
        await original_sleep(0.04)

    monkeypatch.setattr(api().asyncio, "sleep", expire_deadline)
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await loop.run()
    assert len(adapter.requests) == 1
    assert ledger.charged_tool_calls == 0
