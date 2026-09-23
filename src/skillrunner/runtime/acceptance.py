"""Operator-provisioned task checks over isolated candidate bytes."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from skillrunner.artifacts.external import validate_external
from skillrunner.config.models import Acceptance, Policy
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.environment import build_child_environment
from skillrunner.runtime.processes import ProcessSupervisor


async def check_candidate(
    path: Path,
    *,
    size: int,
    digest: str,
    acceptance: Acceptance,
    policy: Policy,
    environ: Mapping[str, str],
    supervisor: ProcessSupervisor,
    deadline: Deadline,
    on_event: Callable[[str, dict[str, Any]], None],
) -> dict[str, Any]:
    checks = []
    for name, validator in acceptance.checks.items():
        result = await validate_external(
            path,
            validator,
            supervisor=supervisor,
            environment=build_child_environment(
                environ,
                references=policy.command_env.get(validator.command, {}),
            ),
            deadline=deadline,
            expected_digest=digest,
            expected_size=size,
            writers_stopped=True,
        )
        checked = {
            "name": name,
            "passed": result.validation.valid,
            "command": result.command,
            "returncode": result.process.returncode,
            "stdout": result.process.stdout.decode("utf-8", errors="replace"),
            "stderr": result.process.stderr.decode("utf-8", errors="replace"),
            "stdout_truncated": result.process.stdout_truncated,
            "stderr_truncated": result.process.stderr_truncated,
        }
        on_event(
            "acceptance_check_completed",
            {
                "name": name,
                "passed": checked["passed"],
                "candidate_sha256": digest,
                "command": result.command,
                "returncode": result.process.returncode,
                "_content": checked,
            },
        )
        checks.append(checked)
    return {"passed": all(c["passed"] for c in checks), "sha256": digest, "checks": checks}
