import hashlib
import importlib
import sys
from pathlib import Path

import pytest

from skillrunner.config.models import ExternalValidator, Policy
from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.environment import ChildEnvironment
from skillrunner.runtime.processes import ProcessSupervisor


def api():
    assert importlib.util.find_spec("skillrunner.artifacts.external") is not None
    return importlib.import_module("skillrunner.artifacts.external")


async def validate(tmp_path, code, *, allow=True, timeout=5):
    path = tmp_path / "candidate λ ;literal.pdf"
    path.write_bytes(b"candidate")
    executable = str(Path(sys.executable).resolve())
    info = Path(executable).stat()
    policy = Policy(
        allowed_executables=[executable] if allow else [],
        executable_identities=(
            {executable: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)}
            if allow
            else {}
        ),
    )
    owner = ProcessSupervisor(policy, shutdown_grace=0.01, max_output_bytes=64)
    return await api().validate_external(
        path,
        ExternalValidator(command=executable, args=["-c", code, "{path}"]),
        supervisor=owner,
        environment=ChildEnvironment({}, {}),
        deadline=Deadline(timeout),
        expected_digest=hashlib.sha256(b"candidate").hexdigest(),
        expected_size=9,
        writers_stopped=True,
    )


async def test_validator_receives_one_literal_path_and_records_identity(tmp_path):
    expected = str(tmp_path / "candidate λ ;literal.pdf")
    result = await validate(
        tmp_path, f"import sys; assert sys.argv[1:] == [{expected!r}]; print('literal-path-ok')"
    )
    assert result.validation.valid
    assert result.validation.validation_level == "external"
    assert result.command == str(Path(sys.executable).resolve())
    assert result.process.returncode == 0
    assert result.process.stdout.strip() == b"literal-path-ok"


async def test_nonzero_validator_rejects_with_bounded_diagnostics(tmp_path):
    result = await validate(tmp_path, "import sys; print('x'*10000); sys.exit(4)")
    assert not result.validation.valid
    assert len(result.process.stdout) <= 64
    assert result.process.stdout_truncated


async def test_validator_cannot_change_accepted_bytes(tmp_path):
    with pytest.raises(RunnerError, match="artifact_invalid"):
        await validate(
            tmp_path, "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('changed')"
        )


async def test_validator_requires_allowlisted_executable(tmp_path):
    with pytest.raises(RunnerError, match="command_not_allowed"):
        await validate(tmp_path, "pass", allow=False)


async def test_validator_timeout_preserves_budget_stop(tmp_path):
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await validate(tmp_path, "import time; time.sleep(10)", timeout=0.02)


@pytest.mark.parametrize(
    "stopped,timeout,error", [(False, 5, "artifact_invalid"), (True, 0, "budget_exhausted")]
)
async def test_stopped_or_expired_validation_does_not_launch(tmp_path, stopped, timeout, error):
    from unittest.mock import AsyncMock

    owner = AsyncMock()
    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    now[0] = 2.0 if timeout == 0 else 0.0
    path = tmp_path / "candidate"
    path.write_bytes(b"candidate")
    with pytest.raises(RunnerError, match=error):
        await api().validate_external(
            path,
            ExternalValidator(command="never", args=["{path}"]),
            supervisor=owner,
            environment=ChildEnvironment({}, {}),
            deadline=deadline,
            expected_digest=hashlib.sha256(b"candidate").hexdigest(),
            expected_size=9,
            writers_stopped=stopped,
        )
    owner.run.assert_not_awaited()


async def test_pending_cancellation_never_launches_validator(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock

    owner = AsyncMock()
    path = tmp_path / "candidate"
    path.write_bytes(b"candidate")

    async def stopped_operation():
        asyncio.current_task().cancel()
        await api().validate_external(
            path,
            ExternalValidator(command="never", args=["{path}"]),
            supervisor=owner,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(5),
            expected_digest=hashlib.sha256(b"candidate").hexdigest(),
            expected_size=9,
            writers_stopped=True,
        )

    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(stopped_operation())
    owner.run.assert_not_awaited()
