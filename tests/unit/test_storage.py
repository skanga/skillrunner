import asyncio

import pytest

from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.storage import monitor_operation


async def test_storage_stop_preserves_command_cleanup_diagnostics():
    checks = 0

    async def operation():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise RunnerError("process_cleanup_failed", "Owned process could not close.") from None

    def check():
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RunnerError("budget_exhausted", "Workspace limit reached.")

    with pytest.raises(RunnerError) as caught:
        await monitor_operation(operation, check)
    assert caught.value.code == "budget_exhausted"
    assert caught.value.details["cleanup_errors"]


async def test_plain_cancelled_operation_does_not_invent_cleanup_failure():
    checks = 0

    async def operation():
        await asyncio.sleep(10)

    def check():
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RunnerError("budget_exhausted", "Workspace limit reached.")

    with pytest.raises(RunnerError) as caught:
        await monitor_operation(operation, check)
    assert "cleanup_errors" not in caught.value.details


async def test_original_operation_error_is_not_its_own_cleanup_failure():
    async def operation():
        raise RunnerError("command_not_allowed", "Denied before launch.")

    with pytest.raises(RunnerError) as caught:
        await monitor_operation(operation, lambda: None)
    assert caught.value.code == "command_not_allowed"
    assert "cleanup_errors" not in caught.value.details
