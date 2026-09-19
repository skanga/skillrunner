"""POSIX session ownership; defer reaping so a group ID cannot be reused."""

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import IO

IS_DARWIN = sys.platform == "darwin"


class OwnedProcess:
    def __init__(
        self,
        executable: str,
        args: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        stdin_pipe: bool,
    ) -> None:
        if not hasattr(os, "WNOWAIT"):
            raise OSError("Non-reaping process status is unavailable")
        self.process = subprocess.Popen(
            [executable, *args],
            executable=executable,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            bufsize=0,
        )
        self.pid = self.process.pid
        self.stdin: IO[bytes] | None = self.process.stdin
        self.stdout: IO[bytes] = self.process.stdout  # type: ignore[assignment]
        self.stderr: IO[bytes] = self.process.stderr  # type: ignore[assignment]
        self._darwin_exited_group_denied = False

    def poll(self) -> int | None:
        if sys.platform == "win32":
            raise OSError("POSIX process ownership is unavailable on Windows")
        else:
            status = os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if status is None:
                return None
            if status.si_code == os.CLD_EXITED:
                return status.si_status
            return -status.si_status

    def terminate(self, *, force: bool) -> None:
        if sys.platform == "win32":
            raise OSError("POSIX process ownership is unavailable on Windows")
        else:
            with suppress(ProcessLookupError):
                try:
                    os.killpg(self.pid, signal.SIGKILL if force else signal.SIGTERM)
                except PermissionError:
                    # Darwin can report EPERM for a group containing only its
                    # unreaped, already-exited leader. Keep the leader unreaped
                    # until the normal cleanup boundary, then verify the group
                    # disappears after reaping it.
                    if not IS_DARWIN or self.poll() is None:
                        raise
                    self._darwin_exited_group_denied = True

    def verify_terminated_group(self) -> None:
        if sys.platform == "win32" or not self._darwin_exited_group_denied:
            return
        try:
            os.killpg(self.pid, 0)
        except ProcessLookupError:
            return
        raise OSError("Owned process group remains after reaping its exited leader")

    def reap(self) -> int:
        return self.process.wait()

    def close(self) -> None:
        for pipe in (self.stdin, self.stdout, self.stderr):
            if pipe is not None:
                pipe.close()
