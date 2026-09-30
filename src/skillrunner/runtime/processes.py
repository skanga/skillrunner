"""One supervised launch path for commands and local MCP protocol pipes.

Shutdown always reserves the full configured grace, including after a normal
leader exit, because descendants can outlive the leader with all pipes closed.
Thus each command adds up to shutdown_grace latency; zero forces cleanup at once.
The capture cap is shared between stdout and stderr. MCP owns its own framing
and output consumption through open(); that path does not capture protocol bytes.
"""

import asyncio
import errno
import math
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from skillrunner.config.models import Policy
from skillrunner.config.sources import verify_executable
from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.environment import ChildEnvironment
from skillrunner.runtime.versions import parse_version, runtime_probe


@dataclass(frozen=True)
class CommandResult:
    pid: int
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool


class PipeReader:
    """Nonblocking bounded reads without worker threads or read-ahead buffers."""

    def __init__(self, pipe: IO[bytes]) -> None:
        self.pipe = pipe
        os.set_blocking(pipe.fileno(), False)

    async def read(self, count: int = 65536) -> bytes:
        if count < 1:
            raise ValueError("A positive bounded read size is required")
        while True:
            try:
                chunk = os.read(self.pipe.fileno(), min(count, 65536))
                await asyncio.sleep(0)
                return chunk
            except BlockingIOError:
                await asyncio.sleep(0.005)
            except OSError as exc:
                # Windows reports a closed pipe as ERROR_BROKEN_PIPE.
                if os.name == "nt" and getattr(exc, "winerror", None) == 109:
                    return b""
                raise


class PipeWriter:
    def __init__(self, pipe: IO[bytes]) -> None:
        self.pipe = pipe
        self.pending = bytearray()
        os.set_blocking(pipe.fileno(), False)

    def write(self, data: bytes) -> None:
        if len(self.pending) + len(data) > 1_048_576:
            raise ValueError("Drain the bounded stdin buffer before writing more")
        self.pending.extend(data)

    async def drain(self) -> None:
        while self.pending:
            try:
                written = os.write(self.pipe.fileno(), self.pending)
                del self.pending[:written]
            except BlockingIOError:
                await asyncio.sleep(0.005)

    def close(self) -> None:
        self.pipe.close()


class SupervisedProcess:
    def __init__(self, native: Any) -> None:
        self.native = native
        self.pid: int = native.pid
        self.stdin = PipeWriter(native.stdin) if native.stdin is not None else None
        self.stdout = PipeReader(native.stdout)
        self.stderr = PipeReader(native.stderr)
        self.drainers: list[asyncio.Task[None]] = []

    async def wait(self) -> int:
        while (code := self.native.poll()) is None:
            await asyncio.sleep(0.005)
        return int(code)


class ProcessSupervisor:
    def __init__(
        self,
        policy: Policy,
        *,
        shutdown_grace: float = 5,
        max_output_bytes: int = 1_048_576,
        on_stopped: Callable[[int, int], None] | None = None,
        on_provenance: Callable[[str, dict[str, Any]], None] | None = None,
        run_deadline: Deadline | None = None,
    ) -> None:
        if not math.isfinite(shutdown_grace) or shutdown_grace < 0:
            raise ValueError("Shutdown grace must be finite and nonnegative")
        if type(max_output_bytes) is not int or max_output_bytes < 0:
            raise ValueError("Output cap must be a nonnegative integer")
        self.policy = policy
        self.shutdown_grace = shutdown_grace
        self.max_output_bytes = max_output_bytes
        self.on_stopped = on_stopped
        self.on_provenance = on_provenance
        self.run_deadline = run_deadline
        self._observed: set[tuple[int, int, int, int]] = set()
        self._provenance_lock = asyncio.Lock()
        self.active: dict[int, SupervisedProcess] = {}
        self.cleanup_errors: list[str] = []

    def _cleanup_failure(self, child: SupervisedProcess, *, retained: bool) -> RunnerError:
        message = (
            "Could not terminate the owned process group; ownership is retained for cleanup retry."
            if retained
            else "Owned process cleanup reported an OS error."
        )
        self.cleanup_errors.append(message)
        return RunnerError(
            "process_cleanup_failed",
            message,
            details={"owned_pid": child.pid, "ownership_retained": retained},
        )

    async def _cleanup(self, child: SupervisedProcess) -> None:
        errors: list[Exception] = []
        try:
            child.native.terminate(force=False)
        except OSError as exc:
            # Darwin may deny a cooperative group signal during child exit.
            # A subsequent successful force request and group check recover
            # this race; other cooperative failures remain observable.
            if not (
                getattr(child.native, "is_darwin", False)
                and isinstance(exc, PermissionError)
                and exc.errno == errno.EPERM
            ):
                errors.append(exc)
        # Preserve the full independent grace for descendants, even when
        # the group leader has exited and closed all of its pipes.
        await asyncio.sleep(self.shutdown_grace)
        try:
            child.native.terminate(force=True)
        except OSError as exc:
            # An unsuccessful force request gives no assurance that wait will
            # finish. Keep native handles, pipes, drainers and ownership intact
            # for aclose() to retry; report failure without an unbounded wait.
            raise self._cleanup_failure(child, retained=True) from exc
        reaped = False
        try:
            returncode = await child.wait()
            child.native.reap()
            reaped = True
            verify_group = getattr(child.native, "verify_terminated_group", None)
            if verify_group is not None:
                try:
                    verify_group()
                except OSError as exc:
                    errors.append(exc)
            if child.drainers:
                done, pending = await asyncio.wait(child.drainers, timeout=1)
                for task in pending:
                    task.cancel()
                results = await asyncio.gather(*child.drainers, return_exceptions=True)
                errors.extend(result for result in results if isinstance(result, Exception))
                if pending:
                    errors.append(OSError("Owned output pipes did not close after termination"))
        finally:
            child.native.close()
        if reaped:
            self.active.pop(child.pid, None)
            if self.on_stopped is not None:
                try:
                    self.on_stopped(child.pid, returncode)
                except Exception as exc:
                    message = "Could not persist the process-stop event."
                    if errors:
                        failure = self._cleanup_failure(child, retained=False)
                        failure.details["reporting_errors"] = [message]
                        raise failure from errors[0]
                    raise RunnerError("reporting_failed", message) from exc
        if errors:
            raise self._cleanup_failure(child, retained=False) from errors[0]

    async def aclose(self) -> None:
        """Retry known ownership after operations settle, attempting every child.

        A failed force request remains registered and is reported to the caller;
        finalization can retry and report that cleanup is incomplete.
        """
        first_error: BaseException | None = None
        for child in list(self.active.values()):
            try:
                await self._finish(child)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    self._attach_cleanup(first_error, exc)
        if first_error is not None:
            raise first_error

    @staticmethod
    def _attach_cleanup(original: BaseException, cleanup: BaseException) -> None:
        if isinstance(cleanup, asyncio.CancelledError) and not getattr(cleanup, "__notes__", []):
            return  # A repeated cancellation is not a cleanup failure.
        message = "Process cleanup also reported an error; see supervisor cleanup diagnostics."
        original.add_note(message)
        if isinstance(original, RunnerError):
            original.details.setdefault("cleanup_errors", []).append(
                {
                    "code": cleanup.code
                    if isinstance(cleanup, RunnerError)
                    else "process_cleanup_failed",
                    "message": message,
                    "details": cleanup.details if isinstance(cleanup, RunnerError) else {},
                }
            )

    async def _finish(self, child: SupervisedProcess) -> None:
        task = asyncio.create_task(self._cleanup(child))
        cancelled: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if cancelled is None:
                    cancelled = exc
            except Exception:
                break  # Retrieve the error below without erasing a prior cancellation.
        try:
            task.result()
        except Exception as cleanup:
            if cancelled is not None:
                self._attach_cleanup(cancelled, cleanup)
                raise cancelled from cleanup
            raise
        if cancelled is not None:
            raise cancelled

    def _check_admission(self, deadline: Deadline) -> None:
        deadline.check()
        if self.run_deadline is not None:
            self.run_deadline.check()
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError

    async def _observe(
        self, executable: str, *, cwd: Path, environment: ChildEnvironment, deadline: Deadline
    ) -> None:
        report = self.on_provenance
        if report is None:
            return
        probe = runtime_probe(executable)
        if probe is None:
            return
        # Serialize concurrent first launches without caching beyond this run.
        async with self._provenance_lock:
            self._check_admission(deadline)
            verify_executable(self.policy, executable)
            identity = self.policy.executable_identities[executable]
            if identity in self._observed:
                return
            self._observed.add(identity)
            family, args, pattern = probe
            record: dict[str, Any] = {
                "family": family,
                "executable": executable,
                "identity": list(identity),
                "version": None,
                "outcome": "unavailable",
            }
            pid: int | None = None
            primary_error: BaseException | None = None

            def started(child_pid: int) -> None:
                nonlocal pid
                pid = child_pid
                report("runtime_provenance_started", {**record, "pid": pid})

            try:
                if args:
                    remaining = min(2.0, deadline.remaining)
                    if self.run_deadline is not None:
                        remaining = min(remaining, self.run_deadline.remaining)
                    self._check_admission(deadline)
                    if remaining <= 0:
                        raise RunnerError("budget_exhausted", "Execution timeout reached.")
                    result = await self._run(
                        executable,
                        args,
                        cwd=cwd,
                        environment=environment,
                        deadline=Deadline(remaining),
                        on_started=started,
                    )
                    if result.returncode == 0:
                        if not result.stdout_truncated and not result.stderr_truncated:
                            record["version"] = parse_version(pattern, result.stdout, result.stderr)
                        if record["version"] is not None:
                            record["outcome"] = "detected"
                    else:
                        record["outcome"] = "failed"
            except RunnerError as exc:
                record["outcome"] = "failed"
                record["error_code"] = exc.code
                if exc.code != "process_start_failed":
                    primary_error = exc
                    raise
            except BaseException as exc:
                primary_error = exc
                record["outcome"] = (
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                )
                raise
            finally:
                pending = []
                if pid is not None:
                    pending.append(
                        (
                            "runtime_provenance_stopped",
                            {
                                "pid": pid,
                                "cleanup_complete": pid not in self.active,
                            },
                        )
                    )
                pending.append(("runtime_provenance_outcome", record))
                reporting_errors = []
                for name, payload in pending:
                    try:
                        report(name, payload)
                    except Exception:
                        reporting_errors.append(
                            {
                                "code": "reporting_failed",
                                "message": "Could not persist a runtime provenance event.",
                            }
                        )
                if reporting_errors:
                    message = "Runtime provenance reporting also failed."
                    if primary_error is not None:
                        primary_error.add_note(message)
                        if isinstance(primary_error, RunnerError):
                            primary_error.details.setdefault("reporting_errors", []).extend(
                                reporting_errors
                            )
                    else:
                        raise RunnerError(
                            "reporting_failed",
                            message,
                            details={"reporting_errors": reporting_errors},
                        ) from None

    @asynccontextmanager
    async def open(
        self,
        executable: str,
        args: Sequence[str],
        *,
        cwd: Path,
        environment: ChildEnvironment,
        deadline: Deadline,
        stdin_pipe: bool = True,
    ) -> AsyncIterator[SupervisedProcess]:
        self._check_admission(deadline)
        if not isinstance(environment, ChildEnvironment):
            raise TypeError("An explicit ChildEnvironment is required")
        verify_executable(self.policy, executable)
        await self._observe(executable, cwd=cwd, environment=environment, deadline=deadline)
        async with self._open(
            executable,
            args,
            cwd=cwd,
            environment=environment,
            deadline=deadline,
            stdin_pipe=stdin_pipe,
        ) as child:
            yield child

    @asynccontextmanager
    async def _open(
        self,
        executable: str,
        args: Sequence[str],
        *,
        cwd: Path,
        environment: ChildEnvironment,
        deadline: Deadline,
        stdin_pipe: bool = True,
        on_started: Callable[[int], None] | None = None,
    ) -> AsyncIterator[SupervisedProcess]:
        """Lifetime and protocol operations share the execution deadline."""
        self._check_admission(deadline)
        if not isinstance(environment, ChildEnvironment):
            raise TypeError("An explicit ChildEnvironment is required")
        if os.name == "nt":
            from skillrunner.runtime import windows

            factory: Any = windows.OwnedProcess
        else:
            from skillrunner.runtime import posix

            factory = posix.OwnedProcess
        # No await between identity verification, launch, and registration: a
        # cancellation can never leave an unregistered successfully spawned child.
        command = verify_executable(self.policy, executable)
        try:
            native = factory(
                command,
                list(args),
                cwd=cwd,
                environment=dict(environment.values),
                stdin_pipe=stdin_pipe,
            )
        except OSError as exc:
            raise RunnerError(
                "process_start_failed", "Could not start a supervised process."
            ) from exc
        child = SupervisedProcess.__new__(SupervisedProcess)
        child.native = native
        child.pid = native.pid
        child.drainers = []
        self.active[child.pid] = child
        try:
            SupervisedProcess.__init__(child, native)
            if on_started is not None:
                on_started(child.pid)
            remaining = deadline.remaining
            if self.run_deadline is not None:
                remaining = min(remaining, self.run_deadline.remaining)
            execution_timeout = asyncio.timeout(remaining)
            try:
                async with execution_timeout:
                    yield child
            except TimeoutError:
                if execution_timeout.expired():
                    raise RunnerError("budget_exhausted", "Execution timeout reached.") from None
                raise
        except BaseException as original:
            try:
                await self._finish(child)
            except BaseException as cleanup:
                self._attach_cleanup(original, cleanup)
            raise
        else:
            await self._finish(child)

    async def _run(
        self,
        executable: str,
        args: Sequence[str],
        *,
        cwd: Path,
        environment: ChildEnvironment,
        deadline: Deadline,
        observe: bool = False,
        on_started: Callable[[int], None] | None = None,
    ) -> CommandResult:
        captured = [bytearray(), bytearray()]
        counts = [0, 0]
        remaining = self.max_output_bytes

        async def drain(reader: PipeReader, index: int) -> None:
            nonlocal remaining
            while chunk := await reader.read():
                counts[index] += len(chunk)
                kept = min(remaining, len(chunk))
                captured[index].extend(chunk[:kept])
                remaining -= kept

        if observe:
            self._check_admission(deadline)
            if not isinstance(environment, ChildEnvironment):
                raise TypeError("An explicit ChildEnvironment is required")
            verify_executable(self.policy, executable)
            await self._observe(executable, cwd=cwd, environment=environment, deadline=deadline)
        async with self._open(
            executable,
            args,
            cwd=cwd,
            environment=environment,
            deadline=deadline,
            stdin_pipe=False,
            on_started=on_started,
        ) as child:
            child.drainers = [
                asyncio.create_task(drain(child.stdout, 0)),
                asyncio.create_task(drain(child.stderr, 1)),
            ]
            code = await child.wait()
            # Context cleanup stops descendants before waiting for pipe EOF.
        return CommandResult(
            child.pid,
            code,
            bytes(captured[0]),
            bytes(captured[1]),
            counts[0],
            counts[1],
            counts[0] > len(captured[0]),
            counts[1] > len(captured[1]),
        )

    async def run(
        self,
        executable: str,
        args: Sequence[str],
        *,
        cwd: Path,
        environment: ChildEnvironment,
        deadline: Deadline,
    ) -> CommandResult:
        return await self._run(
            executable, args, cwd=cwd, environment=environment, deadline=deadline, observe=True
        )
