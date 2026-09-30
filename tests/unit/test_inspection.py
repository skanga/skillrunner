import asyncio
import importlib
import json

import pytest

from skillrunner.config.models import ModelProfile
from skillrunner.domain.errors import RunnerError
from skillrunner.model.media import MediaAttachment
from skillrunner.model.protocol import ModelReply, ModelToolCall, ModelUsage
from skillrunner.runtime.budgets import Deadline, UsageLedger


def api():
    assert importlib.util.find_spec("skillrunner.runtime.inspection") is not None
    return importlib.import_module("skillrunner.runtime.inspection")


class Adapter:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, messages, schemas, output_limit, request_deadline):
        self.requests.append((messages, schemas, output_limit))
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return value


def setup(replies):
    adapter = Adapter(replies)
    ledger = UsageLedger(max_steps=5, max_tool_calls=10, max_tokens=50000)
    profile = ModelProfile(
        base_url="http://example.invalid/v1",
        model="vision",
        context_window_tokens=16000,
        max_output_tokens=1000,
        input_modalities=["text", "image"],
        image_accounting="gemma4-image-max-v1",
    )
    attachment = MediaAttachment(
        b"validated-png-fixture", "scratch/check.png", 20, 10, "gemma4-image-max-v1"
    )
    events = []
    kwargs = dict(
        adapter=adapter,
        profile=profile,
        ledger=ledger,
        deadline=Deadline(10),
        attachment=attachment,
        question="Is the title clipped?",
        max_result_bytes=10000,
        on_event=lambda name, data: events.append((name, data)),
    )
    return adapter, ledger, events, kwargs


def report(text="Title is clipped on the right."):
    args = {"report": text}
    return ModelReply(
        None,
        (ModelToolCall("inspection", "finish_run", args, json.dumps(args)),),
        "tool_calls",
        ModelUsage(1500, 30, 1530),
        None,
    )


def test_inspection_shares_budget_and_returns_only_report_and_metadata():
    adapter, ledger, events, kwargs = setup([report()])
    # Reserve/charge one executor turn first, proving inspection uses the same ledger.
    reservation = ledger.reserve_model(input_tokens=10, context_window=100, max_output=10)
    ledger.reconcile(reservation, input_tokens=10, output_tokens=5, quality="reported")
    result = asyncio.run(api().inspect_png(**kwargs))
    assert result["report"] == "Title is clipped on the right."
    assert result["image"]["path"] == "scratch/check.png"
    assert ledger.model_attempts == 2 and ledger.charged_tokens == 1545
    assert ledger.charged_tool_calls == 1
    assert [x["function"]["name"] for x in adapter.requests[0][1]] == ["finish_run"]
    assert "data:image/png;base64," in json.dumps(adapter.requests[0][0])
    assert "data:image/png;base64," not in json.dumps(events)
    assert "data:image/png;base64," not in json.dumps(result)
    assert all(
        data["role"] == "image_inspector" and data["model"] == "vision" for _, data in events
    )


def test_inspection_length_reconciles_usage_and_stops_without_tool_dispatch():
    reply = ModelReply(None, report().tool_calls, "length", ModelUsage(1500, 1000, 2500), None)
    adapter, ledger, _, kwargs = setup([reply])
    with pytest.raises(RunnerError, match="output-token allowance"):
        asyncio.run(api().inspect_png(**kwargs))
    assert ledger.charged_tokens == 2500
    assert ledger.charged_tool_calls == 0 and len(adapter.requests) == 1


def test_inspection_cannot_bypass_shared_turn_limit():
    adapter, ledger, _, kwargs = setup([report()])
    ledger.max_steps = 1
    reservation = ledger.reserve_model(input_tokens=10, context_window=100, max_output=10)
    ledger.charge_unknown(reservation)
    with pytest.raises(RunnerError, match="Model-turn limit"):
        asyncio.run(api().inspect_png(**kwargs))
    assert not adapter.requests


@pytest.mark.parametrize(
    "change",
    [
        {"context_window_tokens": None},
        {"input_modalities": ["text"]},
        {"image_accounting": "openai-patch-high-v1"},
    ],
)
def test_inspection_rejects_missing_capacity_or_mismatched_image_contract(change):
    adapter, _, _, kwargs = setup([report()])
    kwargs["profile"] = kwargs["profile"].model_copy(update=change)
    with pytest.raises(RunnerError):
        asyncio.run(api().inspect_png(**kwargs))
    assert not adapter.requests
