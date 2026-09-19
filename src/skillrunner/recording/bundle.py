"""Durable run evidence; primary publication has a separate commit boundary."""

import errno
import json
import os
import platform
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_from_bytes

from skillrunner.domain.errors import RunnerError
from skillrunner.recording.events import EventLog
from skillrunner.recording.redaction import Redactor

TERMINAL_EXITS = {
    "succeeded": {0},
    "invalid_request": {2},
    "no_matching_skill": {3},
    "blocked": {4},
    "needs_input": {5},
    "failed": {6},
    "limit_exceeded": {7},
    "cancelled": {130, 143},
}


def atomic_write(path: Path, content: bytes) -> bool:
    """Replace one file atomically; return whether directory fsync was available."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as writer:
            writer.write(content)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
        if sys.platform == "win32":
            return False
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        except OSError as exc:
            if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                return False
            raise
        finally:
            os.close(directory)
        return True
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunBundle:
    def __init__(self, root: Path, state: dict[str, Any], redactor: Redactor) -> None:
        self.root = root
        self.state = state
        self.redactor = redactor
        self.events: EventLog | None = None
        self._receipt: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        output_root: Path,
        *,
        invocation_directory: Path,
        secrets: Iterable[str] = (),
        max_event_bytes: int = 10_485_760,
        log_content: bool = False,
    ) -> "RunBundle":
        secret_values = tuple(secrets)
        try:
            output_root = output_root.resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
            root = output_root / run_id
            root.mkdir(exist_ok=False)
            for relative in (
                "artifacts",
                "work/inputs",
                "work/skills",
                "work/scratch",
                "work/staging",
            ):
                (root / relative).mkdir(parents=True, exist_ok=False)
            state: dict[str, Any] = {
                "identity": {
                    "schema_version": "1",
                    "run_id": run_id,
                    "created_at": _now(),
                    "updated_at": _now(),
                    "finished_at": None,
                    "invocation_directory": str(invocation_directory.resolve()),
                },
                "lifecycle": {
                    "status": "initializing",
                    "phase": "preparing",
                    "exit_code": None,
                    "stop_reason": None,
                    "cleanup": None,
                },
                "request": {},
                "provenance": {
                    "inputs": [],
                    "activated_skills": [],
                    "platform": platform.system(),
                    "python": platform.python_version(),
                },
                "model": {},
                "controls": {},
                "usage": {},
                "outputs": {
                    "primary_output": None,
                    "artifacts": [],
                    "report_path": str(root / "result.md"),
                    "publication": {"state": "not_requested"},
                },
                "diagnostics": {"assumptions": [], "warnings": [], "errors": [], "logging": {}},
            }
            bundle = cls(root, state, Redactor(secret_values))
            bundle.save()
            atomic_write(
                root / "result.md", b"# Run initializing\n\nNo completed result is available yet.\n"
            )
            bundle.events = EventLog(
                root / "events.jsonl",
                run_id=run_id,
                max_bytes=max_event_bytes,
                secrets=secret_values,
                log_content=log_content,
            )
            bundle.events.emit("run_initialized", {})
            return bundle
        except OSError as exc:
            raise RunnerError(
                "reporting_failed", "Cannot initialize a writable run bundle."
            ) from exc

    def save(self) -> None:
        self.state["identity"]["updated_at"] = _now()
        safe = self.redactor.clean(self.state)
        encoded = (json.dumps(safe, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()
        directory_synced = atomic_write(self.root / "run.json", encoded)
        warning = "Directory fsync is unavailable; crash durability follows filesystem semantics."
        warnings = self.state["diagnostics"]["warnings"]
        if not directory_synced and warning not in warnings:
            warnings.append(warning)
            safe = self.redactor.clean(self.state)
            atomic_write(self.root / "run.json", (json.dumps(safe, indent=2) + "\n").encode())

    def _output_link(self, path: str, description: str) -> str:
        safe_path = self.redactor.text(path)
        target = Path(safe_path)
        if target.is_relative_to(self.root):
            href = quote_from_bytes(os.fsencode(target.relative_to(self.root).as_posix()))
        elif target.is_absolute():
            href = target.as_uri()
        else:
            href = quote_from_bytes(os.fsencode(safe_path))
        label = self.redactor.text(description).replace("\n", " ").replace("\\", "\\\\")
        label = label.replace("[", "\\[").replace("]", "\\]")
        return f"[{label}]({href})"

    def _report(self, answer: str) -> bytes:
        status = self.state["lifecycle"]["status"]
        lines = [f"# Run {status}", "", self.redactor.text(answer), ""]
        usage = self.state["usage"]
        timing = (
            ("execution_elapsed_seconds", "Execution elapsed"),
            (
                "post_execution_elapsed_seconds",
                "Post-execution work (cleanup, validation, publication)",
            ),
            ("reporting_elapsed_seconds_at_report", "Reporting elapsed at sample"),
            ("overall_elapsed_seconds_at_report", "Overall elapsed at report sample"),
        )
        for key, label in timing:
            if key in usage:
                lines.append(f"{label}: {usage[key]:.3f} seconds")
        if usage.get("timing_boundary"):
            lines.extend([self.redactor.text(usage["timing_boundary"]), ""])
        primary = self.state["outputs"]["primary_output"]
        if primary:
            lines.extend([f"Primary output: {self._output_link(primary, Path(primary).name)}", ""])
        for artifact in self.state["outputs"]["artifacts"]:
            path = str(artifact["path"])
            description = str(artifact.get("description", Path(path).name))
            artifact_status = str(artifact.get("status", "incomplete"))
            lines.extend(
                [f"Artifact ({artifact_status}): {self._output_link(path, description)}", ""]
            )
        for key, title in (
            ("assumptions", "Assumptions"),
            ("incomplete_work", "Incomplete work"),
            ("warnings", "Warnings"),
        ):
            items = self.state["diagnostics"].get(key, [])
            if items:
                lines.extend([f"## {title}", ""])
                lines.extend(f"- {item}" for item in items)
                lines.append("")
        for error in self.state["diagnostics"]["errors"]:
            lines.extend([f"Error ({error['code']}): {error['message']}", ""])
            if error.get("suggested_action"):
                lines.extend([f"Next action: {error['suggested_action']}", ""])
        return self.redactor.text("\n".join(lines)).encode("utf-8", errors="backslashreplace")

    def finalize(
        self,
        *,
        status: str,
        exit_code: int,
        answer: str,
        reconcile_outcome: Callable[[str, int], tuple[str, int]] | None = None,
        elapsed_seconds: Callable[[], float] | None = None,
    ) -> dict[str, Any]:
        if self._receipt is not None:
            return self._receipt
        if status not in TERMINAL_EXITS or exit_code not in TERMINAL_EXITS[status]:
            raise ValueError("Invalid terminal status/exit pair")
        self.state["lifecycle"].update(status=status, phase="finalizing", exit_code=exit_code)

        def record_finish() -> None:
            if elapsed_seconds is not None:
                self.state["usage"]["elapsed_seconds"] = elapsed_seconds()
            self.state["identity"]["finished_at"] = _now()

        try:
            reporting_started = elapsed_seconds() if elapsed_seconds is not None else None
            # A bounded preliminary write lets the terminal report include time spent
            # entering finalization without recursively trying to time its own last write.
            atomic_write(self.root / "result.md", self._report(answer))
            if elapsed_seconds is not None:
                report_sample = elapsed_seconds()
                assert reporting_started is not None
                self.state["usage"].update(
                    reporting_elapsed_seconds_at_report=report_sample - reporting_started,
                    overall_elapsed_seconds_at_report=report_sample,
                    timing_boundary=(
                        "Timing sampled after the preliminary report write; it excludes the "
                        "final report write and later finalization work."
                    ),
                )
            atomic_write(self.root / "result.md", self._report(answer))
            if reconcile_outcome is not None:
                updated = reconcile_outcome(status, exit_code)
                if updated != (status, exit_code):
                    status, exit_code = updated
                    self.state["lifecycle"].update(status=status, exit_code=exit_code)
                    atomic_write(self.root / "result.md", self._report(answer))
            if self.events:
                self.events.emit("run_finalized", {"status": status, "exit_code": exit_code})
                self.state["diagnostics"]["logging"] = self.events.summary()
                self.events.close()
            record_finish()
            self.save()
        except (OSError, RunnerError):
            if reconcile_outcome is not None:
                status, exit_code = reconcile_outcome(status, exit_code)
                self.state["lifecycle"].update(status=status, exit_code=exit_code)
            committed = self.state["outputs"]["publication"]["state"] == "committed"
            code = "post_publication_reporting_failed" if committed else "reporting_failed"
            if status == "succeeded":
                self.state["lifecycle"].update(status="failed", exit_code=6)
            self.state["diagnostics"]["errors"].append(
                {
                    "code": code,
                    "message": "Final reporting failed; preserved outputs remain available.",
                    "stage": "finalizing",
                    "skill": None,
                    "tool": None,
                    "retryable": False,
                    "outcome_certainty": "known",
                    "suggested_action": "Inspect preserved paths and filesystem availability.",
                }
            )
            with suppress(OSError, RunnerError):
                atomic_write(self.root / "result.md", self._report(answer))
            record_finish()
            with suppress(OSError, RunnerError):
                self.save()
        finally:
            if self.events:
                with suppress(OSError, RunnerError):
                    self.events.close()
        self._receipt = self.redactor.clean(
            {
                "schema_version": "1",
                "run_id": self.state["identity"]["run_id"],
                "status": self.state["lifecycle"]["status"],
                "exit_code": self.state["lifecycle"]["exit_code"],
                "primary_output": self.state["outputs"]["primary_output"],
                "report_path": str(self.root / "result.md"),
                "manifest_path": str(self.root / "run.json"),
                "artifact_paths": [item["path"] for item in self.state["outputs"]["artifacts"]],
                "errors": self.state["diagnostics"]["errors"],
            }
        )
        return self._receipt
