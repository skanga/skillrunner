import json
from decimal import Decimal

import pytest


def ledger_api():
    import importlib

    assert importlib.util.find_spec("skillrunner.qualification.budget") is not None
    return importlib.import_module("skillrunner.qualification.budget")


@pytest.fixture(autouse=True)
def accounting_storage(monkeypatch):
    # Accounting tests simulate durable storage on every OS. The dedicated
    # undurable-write test separately enforces the paid harness's fsync gate.
    module = ledger_api()
    original = module.atomic_write

    def durable(path, content):
        original(path, content)
        return True

    monkeypatch.setattr(module, "atomic_write", durable)


def seed(path):
    path.write_text(
        json.dumps(
            {
                "budget_usd": "5",
                "charged_usd": "0.0015080",
                "requests": [{"charge_usd": "0.0015080"}],
            }
        )
    )


def test_budget_keeps_prior_spending_and_reserves_before_request(tmp_path):
    path = tmp_path / "spending.json"
    seed(path)
    with ledger_api().SpendingLedger(path) as ledger:
        request = ledger.reserve("luna", Decimal("1.25"))
        assert Decimal(json.loads(path.read_text())["charged_usd"]) == Decimal("1.2515080")
        ledger.reconcile(request, Decimal("0.02"), {"input_tokens": 5})
    assert Decimal(json.loads(path.read_text())["charged_usd"]) == Decimal("0.0215080")
    assert len(json.loads(path.read_text())["requests"]) == 2


def test_authorized_qualification_ceiling_accepts_35_but_rejects_more(tmp_path):
    path = tmp_path / "spending.json"
    seed(path)
    state = json.loads(path.read_text())
    state["budget_usd"] = "35"
    path.write_text(json.dumps(state))
    with ledger_api().SpendingLedger(path) as ledger:
        ledger.require_authorized_ceiling()
        assert ledger.state["charged_usd"] == "0.0015080"
    state["budget_usd"] = "35.01"
    path.write_text(json.dumps(state))
    with (
        ledger_api().SpendingLedger(path) as ledger,
        pytest.raises(ValueError, match=r"authorized \$35 ceiling"),
    ):
        ledger.require_authorized_ceiling()
    assert json.loads(path.read_text())["charged_usd"] == "0.0015080"


def test_unreturned_request_remains_charged_after_reopen(tmp_path):
    path = tmp_path / "spending.json"
    seed(path)
    with ledger_api().SpendingLedger(path) as ledger:
        ledger.reserve("luna", Decimal("4"))
    with ledger_api().SpendingLedger(path) as ledger, pytest.raises(ValueError, match="ceiling"):
        ledger.reserve("luna", Decimal("1"))
    assert Decimal(json.loads(path.read_text())["charged_usd"]) == Decimal("4.0015080")


def test_budget_never_reinitializes_or_allows_concurrent_writer(tmp_path):
    path = tmp_path / "spending.json"
    with pytest.raises(FileNotFoundError), ledger_api().SpendingLedger(path):
        pass
    seed(path)
    with (
        ledger_api().SpendingLedger(path),
        pytest.raises(FileExistsError),
        ledger_api().SpendingLedger(path),
    ):
        pass
    assert not path.with_suffix(".json.lock").exists()


def test_invalid_amount_and_understated_ledger_fail_closed(tmp_path):
    path = tmp_path / "spending.json"
    seed(path)
    with ledger_api().SpendingLedger(path) as ledger:
        for value in ("NaN", "Infinity", "-1"):
            with pytest.raises(ValueError):
                ledger.reserve("luna", Decimal(value))
    state = json.loads(path.read_text())
    state["charged_usd"] = "0"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="inconsistent"), ledger_api().SpendingLedger(path):
        pass


async def test_paid_adapter_precharges_and_reconciles_or_keeps_unknown(tmp_path):
    from types import SimpleNamespace

    from skillrunner.model.protocol import ModelReply, ModelUsage

    module = ledger_api()
    path = tmp_path / "spending.json"
    seed(path)
    profile = SimpleNamespace(model="test", context_window_tokens=10000)

    class Adapter:
        async def complete(self, messages, schemas, output_limit, deadline):
            assert output_limit == 100
            assert Decimal(json.loads(path.read_text())["charged_usd"]) > Decimal("0.0015080")
            return ModelReply("answer", (), "stop", ModelUsage(10, 20, 30), None)

    rates = module.StandardRates(Decimal("0.20"), Decimal("1.20"))
    with module.SpendingLedger(path) as ledger:
        adapter = module.PaidAdapter(Adapter(), profile, ledger, rates, max_output=100)
        await adapter.complete([], [], 1000, 5)
    state = json.loads(path.read_text())
    assert state["requests"][-1]["status"] == "reported_usage"
    assert Decimal(state["charged_usd"]) == Decimal("0.0015345")


async def test_missing_usage_or_failed_request_keeps_full_charge(tmp_path):
    from types import SimpleNamespace

    from skillrunner.model.protocol import ModelReply

    module = ledger_api()
    for failure in (False, True):
        path = tmp_path / f"spending-{failure}.json"
        seed(path)

        class Adapter:
            async def complete(self, *args, failure=failure):
                if failure:
                    raise ConnectionError("No response")
                return ModelReply("answer", (), "stop", None, None)

        with module.SpendingLedger(path) as ledger:
            adapter = module.PaidAdapter(
                Adapter(),
                SimpleNamespace(model="test", context_window_tokens=10000),
                ledger,
                module.StandardRates(Decimal("0.2"), Decimal("1.2")),
                max_output=100,
            )
            if failure:
                with pytest.raises(ConnectionError):
                    await adapter.complete([], [], 100, 5)
            else:
                await adapter.complete([], [], 100, 5)
        state = json.loads(path.read_text())
        record = state["requests"][-1]
        assert record["status"] == "reserved_outcome_unknown"
        assert Decimal(state["charged_usd"]) == Decimal("0.0015080") + Decimal(
            record["reserved_usd"]
        )


async def test_128k_paid_context_cap_limits_gpt55_unknown_request_reservation(tmp_path):
    from types import SimpleNamespace

    from skillrunner.model.protocol import ModelReply
    from skillrunner.qualification.run import apply_request_context_cap

    module = ledger_api()
    profile = SimpleNamespace(
        model="gpt-5.5", context_window_tokens=1_050_000, max_output_tokens=4096
    )
    settings = SimpleNamespace(base_url=None)
    rates = module.StandardRates(Decimal("5"), Decimal("30"))
    assert apply_request_context_cap(settings, profile, rates, 128000) == (1_050_000, 128000)
    assert profile.context_window_tokens == 128000

    class NoUsageAdapter:
        async def complete(self, *args):
            return ModelReply("answer", (), "stop", None, None)

    path = tmp_path / "spending.json"
    seed(path)
    with module.SpendingLedger(path) as ledger:
        await module.PaidAdapter(
            NoUsageAdapter(), profile, ledger, rates, max_output=4096
        ).complete([], [], 4096, 5)
    record = json.loads(path.read_text())["requests"][-1]
    assert Decimal(record["reserved_usd"]) == Decimal("0.92288")
    assert record["status"] == "reserved_outcome_unknown"


def test_paid_context_cap_rejects_unsafe_values_before_request():
    from types import SimpleNamespace

    from skillrunner.qualification.run import apply_request_context_cap

    rates = ledger_api().StandardRates(Decimal("5"), Decimal("30"))
    settings = SimpleNamespace(base_url=None)
    profile = SimpleNamespace(context_window_tokens=1_050_000, max_output_tokens=4096)
    for cap in (0, -1, True, 2048):
        with pytest.raises(ValueError, match="cap"):
            apply_request_context_cap(settings, profile, rates, cap)
    assert profile.context_window_tokens == 1_050_000


def test_paid_context_cap_updates_direct_endpoint_profile_without_losing_published_limit():
    from types import SimpleNamespace

    from skillrunner.qualification.run import apply_request_context_cap

    rates = ledger_api().StandardRates(Decimal("5"), Decimal("30"))
    settings = SimpleNamespace(base_url="http://localhost/v1", direct_model=SimpleNamespace())
    profile = SimpleNamespace(context_window_tokens=1_050_000, max_output_tokens=4096)
    assert apply_request_context_cap(settings, profile, rates, 128000) == (1_050_000, 128000)
    assert settings.direct_model.context_window_tokens == 128000
    assert settings.direct_model.max_output_tokens == 4096


def test_overrun_halts_further_requests_but_records_actual_charge(tmp_path):
    path = tmp_path / "spending.json"
    seed(path)
    with ledger_api().SpendingLedger(path) as ledger:
        request = ledger.reserve("test", Decimal("0.01"))
        with pytest.raises(ValueError, match="exceeded"):
            ledger.reconcile(request, Decimal("0.02"), {})
        with pytest.raises(ValueError, match="halted"):
            ledger.reserve("test", Decimal("0.01"))


def test_alias_cannot_fork_cumulative_ledger(tmp_path):
    path = tmp_path / "spending.json"
    alias = tmp_path / "alias.json"
    seed(path)
    try:
        alias.symlink_to(path)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with (
        ledger_api().SpendingLedger(path),
        pytest.raises(FileExistsError),
        ledger_api().SpendingLedger(alias),
    ):
        pass
    assert alias.is_symlink()


def test_hardlinked_ledger_is_rejected(tmp_path):
    import os

    path = tmp_path / "spending.json"
    seed(path)
    try:
        os.link(path, tmp_path / "alias.json")
    except OSError:
        pytest.skip("Hardlinks unavailable")
    with pytest.raises(ValueError, match="hard.link"), ledger_api().SpendingLedger(path):
        pass


def test_undurable_reservation_fails_before_admission(tmp_path, monkeypatch):
    path = tmp_path / "spending.json"
    seed(path)
    module = ledger_api()
    monkeypatch.setattr(module, "atomic_write", lambda path, content: False)
    with module.SpendingLedger(path) as ledger, pytest.raises(OSError, match="durability"):
        ledger.reserve("test", Decimal("1"))
