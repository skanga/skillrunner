"""Reservation and reconciliation contracts, independent of provider SDKs."""

import importlib

import pytest


def budget_api():
    assert importlib.util.find_spec("skillrunner.runtime.budgets") is not None, (
        "Budget ledger has not been implemented"
    )
    return importlib.import_module("skillrunner.runtime.budgets")


def test_request_reservation_uses_smallest_cap():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=40, max_tool_calls=100, max_tokens=1000)
    request = ledger.reserve_model(input_tokens=100, context_window=500, max_output=200)
    assert request.output_limit == 200
    assert ledger.model_attempts == 1
    assert ledger.available_tokens == 700
    ledger.reconcile(request, input_tokens=90, output_tokens=30, quality="reported")
    assert ledger.charged_tokens == 120
    assert ledger.available_tokens == 880


def test_context_capacity_stops_without_charging_model_attempt():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=40, max_tool_calls=100, max_tokens=1000)
    with pytest.raises(ValueError, match="context_capacity_exceeded"):
        ledger.reserve_model(input_tokens=500, context_window=500, max_output=100)
    assert ledger.model_attempts == 0


def test_budget_caps_request_output():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=40, max_tool_calls=100, max_tokens=200)
    request = ledger.reserve_model(input_tokens=100, context_window=1000, max_output=400)
    assert request.output_limit == 100
    ledger.reconcile(request, input_tokens=100, output_tokens=101, quality="reported")
    assert ledger.charged_tokens == 201
    with pytest.raises(ValueError, match="budget_exhausted"):
        ledger.reserve_model(input_tokens=1, context_window=1000, max_output=400)


def test_missing_usage_reconciles_estimate_not_reserved_maximum():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=40, max_tool_calls=100, max_tokens=1000)
    request = ledger.reserve_model(input_tokens=100, context_window=1000, max_output=500)
    ledger.reconcile(request, input_tokens=100, output_tokens=12, quality="estimated")
    assert ledger.charged_tokens == 112
    assert ledger.records[-1].quality == "estimated"


def test_no_response_charges_full_reservation_and_retry_is_new_attempt():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=2, max_tool_calls=100, max_tokens=1000)
    request = ledger.reserve_model(input_tokens=100, context_window=1000, max_output=300)
    ledger.charge_unknown(request)
    assert ledger.charged_tokens == 400
    assert ledger.records[-1].quality == "unknown"
    retry = ledger.reserve_model(input_tokens=100, context_window=1000, max_output=300)
    ledger.reconcile(retry, input_tokens=80, output_tokens=20, quality="reported")
    assert ledger.charged_tokens == 500
    with pytest.raises(ValueError, match="budget_exhausted"):
        ledger.reserve_model(input_tokens=10, context_window=1000, max_output=100)
    assert ledger.model_attempts == 2


def test_batch_rejected_before_any_call_is_charged():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=40, max_tool_calls=2, max_tokens=1000)
    with pytest.raises(ValueError, match="budget_exhausted"):
        ledger.admit_tools(3)
    assert ledger.requested_tool_calls == 3
    assert ledger.charged_tool_calls == 0


def test_invalid_attempts_consume_slots_unused_reservations_do_not():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=40, max_tool_calls=3, max_tokens=1000)
    batch = ledger.admit_tools(3)
    ledger.charge_tool(batch)  # validation-denied call is an attempt
    ledger.charge_tool(batch)
    ledger.close_tools(batch)  # cancellation before the third call
    assert ledger.charged_tool_calls == 2
    assert ledger.available_tool_calls == 1
    next_batch = ledger.admit_tools(1)
    ledger.charge_tool(next_batch)
    ledger.close_tools(next_batch)
    assert ledger.charged_tool_calls == 3


def test_reservations_cannot_be_reconciled_twice():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=40, max_tool_calls=100, max_tokens=1000)
    request = ledger.reserve_model(input_tokens=100, context_window=1000, max_output=300)
    ledger.charge_unknown(request)
    with pytest.raises(ValueError):
        ledger.reconcile(request, input_tokens=0, output_tokens=0, quality="reported")
    assert ledger.charged_tokens == 400


def test_clock_budget_checks_monotonic_deadline():
    api = budget_api()
    now = [10.0]
    deadline = api.Deadline(timeout=2.0, clock=lambda: now[0])
    assert deadline.remaining == 2
    now[0] = 11
    assert deadline.remaining == 1
    now[0] = 12
    with pytest.raises(ValueError, match="budget_exhausted"):
        deadline.check()


def test_fallback_counts_utf8_bytes_and_tool_schemas():
    api = budget_api()
    messages = [{"role": "user", "content": "é"}]
    assert api.estimate_text_tokens(messages, []) > len("é")
    assert api.estimate_text_tokens(messages, [{"name": "read_text"}]) > api.estimate_text_tokens(
        messages, []
    )


def test_foreign_reservation_and_invalid_usage_do_not_mutate_ledger():
    api = budget_api()
    first = api.UsageLedger(max_steps=2, max_tool_calls=2, max_tokens=1000)
    second = api.UsageLedger(max_steps=2, max_tool_calls=2, max_tokens=1000)
    request = first.reserve_model(input_tokens=10, context_window=100, max_output=10)
    own = second.reserve_model(input_tokens=10, context_window=100, max_output=10)
    with pytest.raises(ValueError):
        second.charge_unknown(request)
    with pytest.raises(ValueError):
        second.reconcile(own, input_tokens=-1, output_tokens=0, quality="reported")
    assert second.available_tokens == 980
    assert second.charged_tokens == 0


def test_outstanding_batch_capacity_and_foreign_ownership():
    api = budget_api()
    first = api.UsageLedger(max_steps=2, max_tool_calls=2, max_tokens=1000)
    second = api.UsageLedger(max_steps=2, max_tool_calls=2, max_tokens=1000)
    batch = first.admit_tools(2)
    second.admit_tools(2)
    with pytest.raises(ValueError):
        second.charge_tool(batch)
    with pytest.raises(ValueError, match="budget_exhausted"):
        first.admit_tools(1)
    assert first.charged_tool_calls == second.charged_tool_calls == 0


def test_context_window_can_be_the_smallest_output_cap():
    api = budget_api()
    ledger = api.UsageLedger(max_steps=2, max_tool_calls=2, max_tokens=1000)
    request = ledger.reserve_model(input_tokens=90, context_window=100, max_output=400)
    assert request.output_limit == 10
