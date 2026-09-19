"""Explicit allowlisted format parsers under normal process supervision."""

import asyncio
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from skillrunner.artifacts.validation import ValidationResult
from skillrunner.config.models import ExternalValidator
from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.environment import ChildEnvironment
from skillrunner.runtime.processes import CommandResult, ProcessSupervisor


@dataclass(frozen=True)
class ExternalValidationResult:
    validation: ValidationResult
    command: str
    process: CommandResult


async def _verify(path: Path, size: int, digest: str, deadline: Deadline) -> None:
    await asyncio.sleep(0)
    deadline.check()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        with os.fdopen(os.open(path, flags), "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                raise RunnerError("artifact_invalid", "Validator candidate changed.")
            actual = hashlib.sha256()
            total = 0
            while chunk := source.read(65536):
                await asyncio.sleep(0)
                deadline.check()
                total += len(chunk)
                if total > size:
                    raise RunnerError("artifact_invalid", "Validator candidate changed.")
                actual.update(chunk)
            after = os.fstat(source.fileno())
            if (
                total != size
                or actual.hexdigest() != digest
                or (before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise RunnerError("artifact_invalid", "Validator candidate changed.")
    except OSError as exc:
        raise RunnerError("artifact_invalid", "Cannot read validator candidate.") from exc


async def validate_external(
    path: Path,
    validator: ExternalValidator,
    *,
    supervisor: ProcessSupervisor,
    environment: ChildEnvironment,
    deadline: Deadline,
    expected_digest: str,
    expected_size: int,
    writers_stopped: bool,
) -> ExternalValidationResult:
    """Diagnostic bytes are bounded; callers redact them before report/event persistence."""
    deadline.check()
    if not writers_stopped:
        raise RunnerError("artifact_invalid", "Stop artifact writers before validation.")
    path = path.absolute()
    await _verify(path, expected_size, expected_digest, deadline)
    arguments = [argument.replace("{path}", str(path)) for argument in validator.args]
    await asyncio.sleep(0)
    deadline.check()
    result = await supervisor.run(
        validator.command,
        arguments,
        cwd=path.parent,
        environment=environment,
        deadline=deadline,
    )
    await _verify(path, expected_size, expected_digest, deadline)
    valid = result.returncode == 0
    return ExternalValidationResult(
        ValidationResult(
            valid,
            "external",
            "application/octet-stream",
            () if valid else ("Configured format validator rejected the candidate.",),
        ),
        validator.command,
        result,
    )
