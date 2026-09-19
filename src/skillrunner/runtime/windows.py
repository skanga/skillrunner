"""Windows suspended launch with a kill-on-close Job Object.

Native qualification requires Windows tests. No unsupervised fallback is used.
CPython's standard-library CreateProcess wrapper builds STARTUPINFOEX's explicit
handle list; ctypes supplies the Job Object and primary-thread bindings.
"""

# mypy: disable-error-code="attr-defined"
# Windows-only stdlib declarations are absent from POSIX typeshed views.

import ctypes
import os
import subprocess
from ctypes import wintypes
from pathlib import Path
from typing import IO, Any


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IOCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IOCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel() -> Any:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": (
            [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
            wintypes.BOOL,
        ),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "GenerateConsoleCtrlEvent": ([wintypes.DWORD, wintypes.DWORD], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (args, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes = args
        function.restype = result
    return kernel


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
        if os.name != "nt":
            raise OSError("Windows process supervision requires Windows")
        import _winapi
        import msvcrt

        self.api = _winapi
        self.kernel = _kernel()
        self.job = self.kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle: int | None = None
        self.stdin: IO[bytes] | None = None
        descriptors: set[int] = set()
        thread: int | None = None
        try:
            limits = _ExtendedLimits()
            # KILL_ON_JOB_CLOSE; neither breakaway flag is enabled.
            limits.BasicLimitInformation.LimitFlags = 0x00002000
            if not self.kernel.SetInformationJobObject(
                self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            stdout_read, stdout_write = os.pipe()
            descriptors.update((stdout_read, stdout_write))
            stderr_read, stderr_write = os.pipe()
            descriptors.update((stderr_read, stderr_write))
            if stdin_pipe:
                stdin_read, stdin_write = os.pipe()
                descriptors.update((stdin_read, stdin_write))
            else:
                stdin_read = os.open(os.devnull, os.O_RDONLY)
                descriptors.add(stdin_read)
                stdin_write = None
            child_fds = [stdin_read, stdout_write, stderr_write]
            child_handles = [msvcrt.get_osfhandle(fd) for fd in child_fds]
            for handle in child_handles:
                os.set_handle_inheritable(handle, True)
            startup = subprocess.STARTUPINFO()
            startup.dwFlags = subprocess.STARTF_USESTDHANDLES
            startup.hStdInput, startup.hStdOutput, startup.hStdError = child_handles
            startup.lpAttributeList = {"handle_list": child_handles}
            # CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP. _winapi adds
            # EXTENDED_STARTUPINFO_PRESENT for the explicit inherited handles.
            self.handle, thread, self.pid, _ = _winapi.CreateProcess(
                executable,
                subprocess.list2cmdline([executable, *args]),
                None,
                None,
                True,
                0x00000004 | 0x00000200,
                environment,
                str(cwd),
                startup,
            )
            if not self.kernel.AssignProcessToJobObject(self.job, self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            for fd in child_fds:
                os.close(fd)
                descriptors.remove(fd)
            self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
            descriptors.remove(stdout_read)
            self.stderr = os.fdopen(stderr_read, "rb", buffering=0)
            descriptors.remove(stderr_read)
            if stdin_write is not None:
                self.stdin = os.fdopen(stdin_write, "wb", buffering=0)
                descriptors.remove(stdin_write)
            # All ownership and pipe setup succeed before child code may run.
            if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            # A failed assignment still leaves a suspended owned process. Stop
            # and wait for it explicitly before releasing the creation handle.
            if self.handle is not None:
                _winapi.TerminateProcess(self.handle, 1)
                _winapi.WaitForSingleObject(self.handle, _winapi.INFINITE)
            self.close()
            raise
        finally:
            if thread is not None:
                _winapi.CloseHandle(thread)
            for fd in descriptors:
                os.close(fd)

    def poll(self) -> int | None:
        assert self.handle is not None
        if self.api.WaitForSingleObject(self.handle, 0) == self.api.WAIT_TIMEOUT:
            return None
        return int(self.api.GetExitCodeProcess(self.handle))

    def terminate(self, *, force: bool) -> None:
        if force:
            if not self.kernel.TerminateJobObject(self.job, 1):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            # CTRL_BREAK is cooperative only for a child sharing our console.
            # A headless parent has no console event to deliver; force follows
            # after the same independently configured grace.
            self.kernel.GenerateConsoleCtrlEvent(1, self.pid)

    def reap(self) -> int:
        assert self.handle is not None
        self.api.WaitForSingleObject(self.handle, self.api.INFINITE)
        return int(self.api.GetExitCodeProcess(self.handle))

    def close(self) -> None:
        for name in ("stdin", "stdout", "stderr"):
            pipe = getattr(self, name, None)
            if pipe is not None:
                pipe.close()
        if self.handle is not None:
            self.api.CloseHandle(self.handle)
            self.handle = None
        if self.job:
            if not self.kernel.CloseHandle(self.job):
                raise ctypes.WinError(ctypes.get_last_error())
            self.job = None
