"""One supervised launch path for commands and local MCP protocol pipes.

Shutdown always reserves the full configured grace, including after a normal
leader exit, because descendants can outlive the leader with all pipes closed.
Thus each command adds up to shutdown_grace latency; zero forces cleanup at once.
The capture cap is shared between stdout and stderr. MCP owns its own framing
and output consumption through open(); that path does not capture protocol bytes.
"""

import asyncio
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
    ) -> None:
        if not math.isfinite(shutdown_grace) or shutdown_grace < 0:
            raise ValueError("Shutdown grace must be finite and nonnegative")
        if type(max_output_bytes) is not int or max_output_bytes < 0:
            raise ValueError("Output cap must be a nonnegative integer")
        self.policy = policy
        self.shutdown_grace = shutdown_grace
        self.max_output_bytes = max_output_bytes
        self.on_stopped = on_stopped
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
        """Lifetime and protocol operations share the execution deadline."""
        deadline.check()
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
            execution_timeout = asyncio.timeout(deadline.remaining)
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

    async def run(
        self,
        executable: str,
        args: Sequence[str],
        *,
        cwd: Path,
        environment: ChildEnvironment,
        deadline: Deadline,
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

        async with self.open(
            executable, args, cwd=cwd, environment=environment, deadline=deadline, stdin_pipe=False
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
