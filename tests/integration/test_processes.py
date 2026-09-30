"""Real owned children exercise bounded pipes and process-tree cleanup."""

import asyncio
import errno
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import pytest

from skillrunner.config.models import Policy
from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.environment import ChildEnvironment, build_child_environment


def supervisor(**kwargs):
    from skillrunner.runtime.processes import ProcessSupervisor

    executable = str(Path(sys.executable).resolve())
    stat = Path(executable).stat()
    policy = Policy(
        allowed_executables=[executable],
        executable_identities={
            executable: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        },
    )
    return ProcessSupervisor(policy, shutdown_grace=0.04, **kwargs), executable


async def invoke(tmp_path, code, args=(), **kwargs):
    owner, executable = supervisor(**kwargs)
    return await owner.run(
        executable,
        ["-c", code, *args],
        cwd=tmp_path,
        environment=ChildEnvironment({}, {}),
        deadline=Deadline(5),
    )


async def test_arguments_stdin_and_nonzero(tmp_path):
    result = await invoke(
        tmp_path,
        "import sys,json; print(json.dumps(sys.argv[1:])); "
        "print(len(sys.stdin.read())); sys.exit(7)",
        ["two words", "λ", "$(echo unsafe)"],
    )
    lines = result.stdout.decode().splitlines()
    assert json.loads(lines[0]) == ["two words", "λ", "$(echo unsafe)"]
    assert lines[1] == "0"
    assert result.returncode == 7
    assert result.pid > 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signaling")
async def test_darwin_zombie_only_group_signal_denial_cleans_up(tmp_path, monkeypatch):
    from skillrunner.runtime import posix

    monkeypatch.setattr(posix, "IS_DARWIN", True, raising=False)
    real_killpg = os.killpg

    def darwin_killpg(pid, selected_signal):
        if selected_signal in (signal.SIGTERM, signal.SIGKILL):
            status = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if status is not None:
                raise PermissionError(1, "Zombie-only process group")
        return real_killpg(pid, selected_signal)

    monkeypatch.setattr(posix.os, "killpg", darwin_killpg)
    result = await invoke(tmp_path, "print('finished')")
    assert result.returncode == 0
    assert result.stdout == b"finished\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signaling")
async def test_darwin_graceful_eperm_is_recovered_by_forced_cleanup(tmp_path, monkeypatch):
    from skillrunner.runtime import posix

    monkeypatch.setattr(posix, "IS_DARWIN", True)
    original = posix.OwnedProcess.terminate

    def deny_graceful(self, *, force):
        if not force:
            raise PermissionError(errno.EPERM, "Native group signal denied")
        return original(self, force=force)

    monkeypatch.setattr(posix.OwnedProcess, "terminate", deny_graceful)
    result = await invoke(tmp_path, "print('finished')")
    assert result.returncode == 0
    assert result.stdout == b"finished\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signaling")
def test_darwin_force_group_denial_kills_leader_and_verifies_group(tmp_path, monkeypatch):
    from skillrunner.runtime import posix

    monkeypatch.setattr(posix, "IS_DARWIN", True)
    child = posix.OwnedProcess(
        sys.executable,
        ["-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        environment=dict(os.environ),
        stdin_pipe=False,
    )
    real_killpg = os.killpg

    def deny_group_force(pid, selected_signal):
        if selected_signal == signal.SIGKILL:
            raise PermissionError(errno.EPERM, "Group signal denied")
        return real_killpg(pid, selected_signal)

    try:
        monkeypatch.setattr(posix.os, "killpg", deny_group_force)
        child.terminate(force=True)
        deadline = time.monotonic() + 2
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child.poll() is not None
        child.reap()
        child.verify_terminated_group()
    finally:
        monkeypatch.setattr(posix.os, "killpg", real_killpg)
        if child.process.poll() is None:
            child.terminate(force=True)
        child.reap()
        child.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signaling")
def test_darwin_live_group_signal_denial_remains_cleanup_failure(tmp_path, monkeypatch):
    from skillrunner.runtime import posix

    monkeypatch.setattr(posix, "IS_DARWIN", True, raising=False)
    child = posix.OwnedProcess(
        sys.executable,
        ["-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        environment=dict(os.environ),
        stdin_pipe=False,
    )
    real_killpg = os.killpg
    real_kill = os.kill

    def deny(_pid, _signal):
        raise PermissionError(1, "denied")

    try:
        monkeypatch.setattr(posix.os, "killpg", deny)
        monkeypatch.setattr(posix.os, "kill", deny)
        with pytest.raises(PermissionError):
            child.terminate(force=True)
    finally:
        monkeypatch.setattr(posix.os, "killpg", real_killpg)
        monkeypatch.setattr(posix.os, "kill", real_kill)
        child.terminate(force=True)
        child.reap()
        child.close()


@pytest.mark.parametrize("outcome", ["normal", "timeout", "cancel"])
async def test_stop_callback_only_after_owned_process_is_reaped(tmp_path, outcome):
    owner, executable = supervisor()
    stopped = []

    def observe(pid, returncode):
        assert pid not in owner.active
        stopped.append((pid, returncode))

    owner.on_stopped = observe
    task = asyncio.create_task(
        owner.run(
            executable,
            [
                "-c",
                "import sys; sys.exit(7)" if outcome == "normal" else "import time; time.sleep(60)",
            ],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(0.2 if outcome == "timeout" else 5),
        )
    )
    if outcome == "normal":
        result = await task
        assert stopped == [(result.pid, 7)]
    else:
        if outcome == "cancel":
            async with asyncio.timeout(5):
                while not owner.active:
                    await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RunnerError, match="budget_exhausted"):
                await task
        assert len(stopped) == 1
        assert isinstance(stopped[0][1], int)
    assert not owner.active
    await owner.aclose()
    assert len(stopped) == 1


@pytest.mark.parametrize("timeout", [False, True])
async def test_stop_callback_failure_does_not_retain_reaped_process(tmp_path, timeout):
    owner, executable = supervisor()
    attempts = []

    def broken(pid, returncode):
        attempts.append(pid)
        assert pid not in owner.active
        raise OSError("private logging failure")

    owner.on_stopped = broken
    with pytest.raises(RunnerError) as raised:
        await owner.run(
            executable,
            ["-c", "import time; time.sleep(60)" if timeout else "pass"],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(0.2 if timeout else 5),
        )
    assert raised.value.code == ("budget_exhausted" if timeout else "reporting_failed")
    if timeout:
        assert raised.value.details["cleanup_errors"][0]["code"] == "reporting_failed"
    assert not owner.active
    await owner.aclose()
    assert len(attempts) == 1


async def test_flood_both_pipes_remains_bounded(tmp_path):
    result = await invoke(
        tmp_path,
        'import os\nfor i in range(512):\n os.write(1,b"x"*4096); os.write(2,b"y"*4096)',
        max_output_bytes=1024,
    )
    assert len(result.stdout) + len(result.stderr) <= 1024
    assert result.stdout_bytes == result.stderr_bytes == 512 * 4096
    assert result.stdout_truncated and result.stderr_truncated
    assert result.returncode == 0


async def test_denied_and_changed_identity_never_launch(tmp_path):
    owner, executable = supervisor()
    with pytest.raises(RunnerError, match="command_not_allowed"):
        await owner.run(
            "/not/allowed",
            [],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(1),
        )
    copied = tmp_path / "python"
    shutil.copy(executable, copied)
    stat = copied.stat()
    owner.policy.executable_identities[str(copied)] = (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    os.utime(copied, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(RunnerError, match="missing_dependency"):
        await owner.run(
            str(copied),
            ["-c", 'raise Exception("launched")'],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(1),
        )


async def test_timeout_and_forced_termination(tmp_path):
    owner, executable = supervisor()
    started = time.monotonic()
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await owner.run(
            executable,
            [
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)",
            ],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(0.15),
        )
    assert time.monotonic() - started < 2
    assert not owner.active


@pytest.mark.skipif(os.name != "posix", reason="POSIX fork fixture")
async def test_descendant_outlives_parent_and_closes_pipes(tmp_path):
    marker = tmp_path / "escaped"
    code = (
        "import os,time,signal\n"
        "if os.fork()==0:\n"
        " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        " os.close(1); os.close(2); time.sleep(.4)\n"
        f' open({str(marker)!r},"w").write("alive")\n'
        " os._exit(0)\n"
        "os._exit(0)"
    )
    await invoke(tmp_path, code)
    await asyncio.sleep(0.5)
    assert not marker.exists()


async def test_repeated_cancellation_finishes_cleanup(tmp_path):
    owner, executable = supervisor()
    task = asyncio.create_task(
        owner.run(
            executable,
            ["-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(10),
        )
    )
    while not owner.active:
        await asyncio.sleep(0.005)
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as raised:
        await task
    assert not getattr(raised.value, "__notes__", [])
    assert not owner.active


async def test_supervised_stdio_context(tmp_path):
    owner, executable = supervisor()
    async with owner.open(
        executable,
        ["-u", "-c", "import sys; print(sys.stdin.readline())"],
        cwd=tmp_path,
        environment=ChildEnvironment({}, {}),
        deadline=Deadline(2),
        stdin_pipe=True,
    ) as child:
        child.stdin.write(b"hello\n")
        await child.stdin.drain()
        assert b"hello" in await child.stdout.read(100)
    assert not owner.active


async def test_shutdown_grace_is_independent_of_deadline(tmp_path):
    from skillrunner.runtime.processes import ProcessSupervisor

    existing, executable = supervisor()
    owner = ProcessSupervisor(existing.policy, shutdown_grace=0.15)
    started = time.monotonic()
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await owner.run(
            executable,
            ["-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(0.05),
        )
    assert time.monotonic() - started >= 0.19
    assert not owner.active


async def test_zero_capture_still_drains(tmp_path):
    result = await invoke(
        tmp_path, 'import sys; sys.stdout.buffer.write(b"discarded\\n")', max_output_bytes=0
    )
    assert result.stdout == result.stderr == b""
    assert result.stdout_bytes == 10
    assert result.stdout_truncated


def test_windows_adapter_available_but_not_qualified_on_posix():
    from skillrunner.runtime import windows

    assert callable(windows.OwnedProcess)


@pytest.mark.parametrize("assignment_succeeds", [True, False])
def test_windows_job_assignment_precedes_resume(monkeypatch, tmp_path, assignment_succeeds):
    """Binding-order test only; cannot qualify native Windows behavior."""
    import types

    from skillrunner.runtime import windows

    events = []

    class Kernel:
        def CreateJobObjectW(self, *args):
            events.append("job")
            return 11

        def SetInformationJobObject(self, job, kind, limits, size):
            assert limits._obj.BasicLimitInformation.LimitFlags == 0x2000
            events.append("limits")
            return True

        def AssignProcessToJobObject(self, job, process):
            events.append("assign")
            return assignment_succeeds

        def ResumeThread(self, thread):
            events.append("resume")
            return 1

        def CloseHandle(self, handle):
            return True

    def create(*args):
        assert args[5] & 4  # CREATE_SUSPENDED
        assert len(args[8].lpAttributeList["handle_list"]) == 3
        events.append("create_suspended")
        return 20, 21, 22, 23

    api = types.SimpleNamespace(
        CreateProcess=create,
        TerminateProcess=lambda *args: events.append("terminate"),
        WaitForSingleObject=lambda *args: events.append("wait"),
        CloseHandle=lambda *args: None,
        INFINITE=-1,
    )
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "_winapi", api)
        patch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(get_osfhandle=lambda fd: fd))
        patch.setattr(windows.os, "name", "nt")
        patch.setattr(windows.os, "set_handle_inheritable", lambda *args: None, raising=False)
        patch.setattr(windows.subprocess, "STARTUPINFO", types.SimpleNamespace, raising=False)
        patch.setattr(windows.subprocess, "STARTF_USESTDHANDLES", 256, raising=False)
        patch.setattr(
            windows.ctypes, "WinError", lambda code: OSError("native failure"), raising=False
        )
        patch.setattr(windows.ctypes, "get_last_error", lambda: 5, raising=False)
        patch.setattr(windows, "_kernel", Kernel)
        if assignment_succeeds:
            child = windows.OwnedProcess(
                "python", [], cwd=tmp_path, environment={}, stdin_pipe=False
            )
            child.close()
        else:
            with pytest.raises(OSError):
                windows.OwnedProcess("python", [], cwd=tmp_path, environment={}, stdin_pipe=False)
    assert events[:4] == ["job", "limits", "create_suspended", "assign"]
    if assignment_succeeds:
        assert events[4:] == ["resume"]
    else:
        assert events[4:] == ["terminate", "wait"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX fork fixture")
async def test_descendant_holding_pipes_cannot_deadlock(tmp_path):
    result = await invoke(
        tmp_path, "import os,time\nif os.fork()==0: time.sleep(60)\nelse: os._exit(0)"
    )
    assert result.returncode == 0


async def test_cleanup_error_is_observable_and_force_still_runs(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX signal injection")
    from skillrunner.runtime.posix import OwnedProcess

    original = OwnedProcess.terminate
    calls = []

    def fail_cooperative(self, *, force):
        calls.append(force)
        if not force:
            raise PermissionError("injected failure")
        return original(self, force=force)

    monkeypatch.setattr(OwnedProcess, "terminate", fail_cooperative)
    owner, executable = supervisor()
    with pytest.raises(RunnerError, match="process_cleanup_failed"):
        await owner.run(
            executable,
            ["-c", "pass"],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(1),
        )
    assert calls == [False, True]
    assert owner.cleanup_errors


async def test_zero_grace_has_no_added_shutdown_delay(tmp_path):
    from skillrunner.runtime.processes import ProcessSupervisor

    existing, executable = supervisor()
    owner = ProcessSupervisor(existing.policy, shutdown_grace=0)
    started = time.monotonic()
    result = await owner.run(
        executable,
        ["-c", "pass"],
        cwd=tmp_path,
        environment=ChildEnvironment({}, {}),
        deadline=Deadline(2),
    )
    assert result.returncode == 0
    assert time.monotonic() - started < 1
    assert not owner.active


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal injection")
@pytest.mark.parametrize("stop", ["timeout", "cancel", "cancel_during_cleanup"])
async def test_original_stop_survives_cleanup_error(tmp_path, monkeypatch, stop):
    from skillrunner.runtime.posix import OwnedProcess

    original = OwnedProcess.terminate
    cleanup_started = asyncio.Event()

    def fail_cooperative(self, *, force):
        if not force:
            cleanup_started.set()
            raise PermissionError("injected failure")
        return original(self, force=force)

    monkeypatch.setattr(OwnedProcess, "terminate", fail_cooperative)
    owner, executable = supervisor()
    code = "pass" if stop == "cancel_during_cleanup" else "import time; time.sleep(60)"
    task = asyncio.create_task(
        owner.run(
            executable,
            ["-c", code],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(0.05 if stop == "timeout" else 3),
        )
    )
    if stop == "cancel":
        while not owner.active:
            await asyncio.sleep(0.005)
        task.cancel()
    if stop == "cancel_during_cleanup":
        await cleanup_started.wait()
        task.cancel()
    if stop == "timeout":
        with pytest.raises(RunnerError) as raised:
            await task
        assert raised.value.code == "budget_exhausted"
        assert raised.value.details["cleanup_errors"]
    else:
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await task
        assert cancelled.value.__notes__
    assert owner.cleanup_errors
    assert not owner.active


async def test_body_timeout_error_is_not_execution_deadline(tmp_path):
    owner, executable = supervisor()
    failure = TimeoutError("protocol operation timed out")
    with pytest.raises(TimeoutError) as raised:
        async with owner.open(
            executable,
            ["-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(60),
        ):
            raise failure
    assert raised.value is failure
    assert not owner.active


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal injection")
async def test_failed_force_returns_diagnostics_and_retains_child_for_retry(tmp_path, monkeypatch):
    from skillrunner.runtime.posix import OwnedProcess

    original = OwnedProcess.terminate

    def fail_force(self, *, force):
        if force:
            raise PermissionError("injected force failure")
        return original(self, force=force)

    owner, executable = supervisor()
    stopped = []
    owner.on_stopped = lambda pid, returncode: stopped.append((pid, returncode))
    monkeypatch.setattr(OwnedProcess, "terminate", fail_force)

    async def operate():
        async with owner.open(
            executable,
            [
                "-u",
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                'import os; os.write(1,b"ready"); time.sleep(0.02); '
                'os.write(1,b"\\n"); time.sleep(60)',
            ],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(0.15),
        ) as child:
            ready = bytearray()
            while len(ready) < len(b"ready\n"):
                chunk = await child.stdout.read(len(b"ready\n") - len(ready))
                assert chunk, "Child exited before readiness"
                ready.extend(chunk)
            assert ready == b"ready\n"
            await asyncio.sleep(60)

    task = asyncio.create_task(operate())
    done, pending = await asyncio.wait({task}, timeout=0.6)
    if pending:
        # Rescue the owned fixture even when the old implementation hangs.
        for child in owner.active.values():
            original(child.native, force=True)
        await asyncio.gather(task, return_exceptions=True)
    try:
        assert done, "Force failure must not leave cleanup waiting indefinitely"
        with pytest.raises(RunnerError) as raised:
            await task
        assert raised.value.code == "budget_exhausted"
        assert raised.value.details["cleanup_errors"]
        assert owner.cleanup_errors
        assert len(owner.active) == 1
        assert not stopped
        child = next(iter(owner.active.values()))
        assert child.native.poll() is None
        assert not child.native.stdout.closed
    finally:
        monkeypatch.setattr(OwnedProcess, "terminate", original)
        for child in list(owner.active.values()):
            original(child.native, force=True)
        if hasattr(owner, "aclose"):
            await owner.aclose()
    assert not owner.active
    assert len(stopped) == 1


async def test_runtime_version_actual_python_once_and_denied_never_probed(tmp_path):
    import platform

    events = []
    owner, executable = supervisor()
    owner.on_provenance = lambda name, payload: events.append((name, payload.copy()))
    for _ in range(2):
        result = await owner.run(
            executable,
            ["-c", "print('task')"],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(5),
        )
        assert result.stdout.splitlines() == [b"task"]
    observed = [payload for name, payload in events if name == "runtime_provenance_outcome"]
    assert len(observed) == 1
    assert observed[0]["version"] == platform.python_version()
    assert observed[0]["outcome"] == "detected"
    assert observed[0]["executable"] == executable
    assert observed[0]["family"] == "python"
    assert [name for name, _ in events] == [
        "runtime_provenance_started",
        "runtime_provenance_stopped",
        "runtime_provenance_outcome",
    ]
    before = list(events)
    with pytest.raises(RunnerError, match="allowlist"):
        await owner.run(
            "denied-python",
            [],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(5),
        )
    assert events == before
    assert not owner.active


def runtime_script(tmp_path, probe_code, *, name="python", cap=1024):
    from skillrunner.runtime.processes import ProcessSupervisor

    executable = tmp_path / name
    executable.write_text(
        f"#!{sys.executable}\nimport sys,os,time\n"
        "if sys.argv[1:] == ['--version']:\n"
        + "\n".join("    " + line for line in probe_code.splitlines())
        + "\nelse:\n    open('task-started', 'w').write('yes')\n"
    )
    executable.chmod(0o755)
    stat = executable.stat()
    policy = Policy(
        allowed_executables=[str(executable)],
        executable_identities={
            str(executable): (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        },
    )
    events = []
    owner = ProcessSupervisor(
        policy,
        shutdown_grace=0.01,
        max_output_bytes=cap,
        on_provenance=lambda name, payload: events.append((name, payload.copy())),
    )
    return owner, str(executable), events


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
@pytest.mark.parametrize(
    "probe_code,cap,outcome",
    [
        ("print('unrecognized TOP_SECRET')", 1024, "unavailable"),
        ("print('Python 3.14.789secret')", 13, "unavailable"),
        ("print('TOP_SECRET'); sys.exit(1)", 1024, "failed"),
        ("print('Python 3.14.7'); print('TOP_SECRET', file=sys.stderr)", 1024, "detected"),
    ],
)
async def test_runtime_output_is_bounded_and_only_parsed_version_persisted(
    tmp_path, probe_code, cap, outcome
):
    owner, executable, events = runtime_script(tmp_path, probe_code, cap=cap)
    await owner.run(
        executable, [], cwd=tmp_path, environment=ChildEnvironment({}, {}), deadline=Deadline(5)
    )
    record = events[-1][1]
    assert record["outcome"] == outcome
    assert "TOP_SECRET" not in json.dumps(events)
    assert (tmp_path / "task-started").exists()
    assert not owner.active


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_probe_deadline_is_terminal_and_never_starts_task(tmp_path):
    owner, executable, events = runtime_script(tmp_path, "time.sleep(5)")
    started = time.monotonic()
    with pytest.raises(RunnerError) as error:
        await owner.run(
            executable,
            [],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(0.08),
        )
    assert error.value.code == "budget_exhausted"
    assert time.monotonic() - started < 1
    assert not (tmp_path / "task-started").exists()
    assert not owner.active
    assert events[-1][1]["error_code"] == "budget_exhausted"


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_probe_cancellation_cleans_up_without_task(tmp_path):
    owner, executable, events = runtime_script(tmp_path, "time.sleep(5)")
    task = asyncio.create_task(
        owner.run(
            executable,
            [],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(10),
        )
    )
    while not owner.active:
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not owner.active
    assert not (tmp_path / "task-started").exists()
    assert events[-1][1]["outcome"] == "cancelled"


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_probe_revalidates_executable_and_original_deadline(tmp_path):
    owner, executable, _ = runtime_script(
        tmp_path, "open(__file__, 'a').write('# changed'); print('Python 3.14.7')"
    )
    with pytest.raises(RunnerError) as error:
        await owner.run(
            executable, [], cwd=tmp_path, environment=ChildEnvironment({}, {}), deadline=Deadline(5)
        )
    assert error.value.code == "missing_dependency"
    assert not (tmp_path / "task-started").exists()


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_probe_before_protocol_has_closed_stdin_same_environment_cwd(tmp_path):
    owner, executable, events = runtime_script(
        tmp_path,
        "assert sys.stdin.read() == ''; assert os.environ['CHECK'] == 'yes'; "
        "assert os.getcwd() == os.environ['EXPECTED_CWD']; print('Python 3.14.7')",
    )
    async with owner.open(
        executable,
        [],
        cwd=tmp_path,
        environment=ChildEnvironment({"CHECK": "yes", "EXPECTED_CWD": str(tmp_path)}, {}),
        deadline=Deadline(5),
    ) as child:
        assert child.stdin is not None
        assert await child.wait() == 0
    assert events[-1][1]["outcome"] == "detected"
    assert not owner.active


async def test_runtime_actual_node_if_installed(tmp_path):
    from skillrunner.runtime.processes import ProcessSupervisor

    found = shutil.which("node", path=os.defpath) or shutil.which("node")
    if found is None:
        pytest.skip("Node is not installed")
    executable = str(Path(found).resolve())
    stat = Path(executable).stat()
    records = []
    owner = ProcessSupervisor(
        Policy(
            allowed_executables=[executable],
            executable_identities={
                executable: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            },
        ),
        shutdown_grace=0,
        on_provenance=lambda name, record: records.append((name, record)),
    )
    result = await owner.run(
        executable,
        ["-p", "process.versions.node"],
        cwd=tmp_path,
        environment=build_child_environment(os.environ, references={}),
        deadline=Deadline(5),
    )
    assert records[-1][1]["outcome"] == "detected"
    assert records[-1][1]["version"] == result.stdout.decode().strip()
    assert records[-1][1]["family"] == "node"


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_probe_shutdown_grace_does_not_restart_operation_deadline(tmp_path):
    owner, executable, events = runtime_script(tmp_path, "print('Python 3.14.7')")
    owner.shutdown_grace = 0.12
    with pytest.raises(RunnerError) as error:
        await owner.run(
            executable,
            [],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(0.1),
        )
    assert error.value.code == "budget_exhausted"
    assert events[-1][1]["outcome"] == "detected"
    assert not (tmp_path / "task-started").exists()
    assert not owner.active


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_probe_bound_by_run_deadline_and_two_second_cap(tmp_path):
    owner, executable, _ = runtime_script(tmp_path, "time.sleep(5)")
    owner.run_deadline = Deadline(0.08)
    with pytest.raises(RunnerError) as error:
        await owner.run(
            executable,
            [],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(10),
        )
    assert error.value.code == "budget_exhausted"
    assert not owner.active
    assert not (tmp_path / "task-started").exists()
    # A fresh run has a fresh identity cache and the fixed two-second probe cap.
    owner, executable, _ = runtime_script(tmp_path, "time.sleep(5)")
    started = time.monotonic()
    with pytest.raises(RunnerError) as error:
        await owner.run(
            executable,
            [],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(10),
        )
    assert error.value.code == "budget_exhausted"
    assert 1.8 < time.monotonic() - started < 3
    assert not owner.active


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_unavailable_shell_has_no_version_process(tmp_path):
    owner, executable, events = runtime_script(
        tmp_path, "raise AssertionError('no probe')", name="dash"
    )
    await owner.run(
        executable, [], cwd=tmp_path, environment=ChildEnvironment({}, {}), deadline=Deadline(5)
    )
    assert [name for name, _ in events] == ["runtime_provenance_outcome"]
    assert events[0][1]["family"] == "shell"
    assert events[0][1]["outcome"] == "unavailable"
    assert events[0][1]["version"] is None


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_probe_cleanup_failure_is_terminal(tmp_path, monkeypatch):
    owner, executable, events = runtime_script(tmp_path, "print('Python 3.14.7')")
    original = owner._cleanup

    async def cleanup(child):
        await original(child)
        raise RunnerError("process_cleanup_failed", "Fixture cleanup error")

    monkeypatch.setattr(owner, "_cleanup", cleanup)
    with pytest.raises(RunnerError) as error:
        await owner.run(
            executable, [], cwd=tmp_path, environment=ChildEnvironment({}, {}), deadline=Deadline(5)
        )
    assert error.value.code == "process_cleanup_failed"
    assert not (tmp_path / "task-started").exists()
    assert events[-1][1]["error_code"] == "process_cleanup_failed"
    assert not owner.active


@pytest.mark.parametrize(
    "executable,output,version",
    [
        ("python3.14", b"Python 3.14.7\n", "3.14.7"),
        ("node.exe", b"v22.4.0\n", "22.4.0"),
        ("bash", b"GNU bash, version 5.2.21(1)-release (x86_64-pc-linux-gnu)\n", "5.2.21"),
        ("zsh", b"zsh 5.9 (x86_64-ubuntu-linux-gnu)\n", "5.9"),
        ("python", b"Python 3.14.7 secret\n", None),
    ],
)
def test_runtime_version_recognition(executable, output, version):
    from skillrunner.runtime.versions import parse_version, runtime_probe

    _, args, pattern = runtime_probe(executable)
    assert args == ("--version",)
    assert parse_version(pattern, output, b"") == version


@pytest.mark.skipif(os.name != "posix", reason="Executable script fixture")
async def test_runtime_probe_pending_cancellation_blocks_task_launch(tmp_path):
    owner, executable, events = runtime_script(tmp_path, "print('Python 3.14.7')")
    report = owner.on_provenance

    def cancel_after_probe(name, payload):
        report(name, payload)
        if name == "runtime_provenance_outcome":
            asyncio.current_task().cancel()

    owner.on_provenance = cancel_after_probe
    task = asyncio.create_task(
        owner.run(
            executable, [], cwd=tmp_path, environment=ChildEnvironment({}, {}), deadline=Deadline(5)
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events[-1][1]["outcome"] == "detected"
    assert not (tmp_path / "task-started").exists()
    assert not owner.active


@pytest.mark.skipif(os.name != "posix", reason="POSIX native launch fixture")
async def test_runtime_probe_launch_error_is_recorded_but_task_can_run(tmp_path, monkeypatch):
    from skillrunner.runtime import posix

    owner, executable, events = runtime_script(tmp_path, "print('Python 3.14.7')")
    original = posix.OwnedProcess
    launches = []

    def launch(command, args, **kwargs):
        launches.append(args)
        if args == ["--version"]:
            raise OSError("TOP_SECRET")
        return original(command, args, **kwargs)

    monkeypatch.setattr(posix, "OwnedProcess", launch)
    await owner.run(
        executable, [], cwd=tmp_path, environment=ChildEnvironment({}, {}), deadline=Deadline(5)
    )
    assert launches == [["--version"], []]
    assert events[-1][1]["outcome"] == "failed"
    assert events[-1][1]["error_code"] == "process_start_failed"
    assert "TOP_SECRET" not in json.dumps(events)
    assert (tmp_path / "task-started").exists()


@pytest.mark.parametrize(
    "primary", ["cancelled", "budget_exhausted", "process_cleanup_failed", None]
)
async def test_runtime_reporting_failure_preserves_primary_and_attempts_outcome(
    tmp_path, monkeypatch, primary
):
    owner, executable = supervisor()
    attempts = []
    original_cleanup = owner._cleanup

    async def cleanup(child):
        await original_cleanup(child)
        if primary == "process_cleanup_failed":
            raise RunnerError("process_cleanup_failed", "Fixture cleanup failure")

    monkeypatch.setattr(owner, "_cleanup", cleanup)

    def report(name, record):
        attempts.append(name)
        if name == "runtime_provenance_started":
            if primary == "cancelled":
                asyncio.current_task().cancel()
            elif primary == "budget_exhausted":
                raise RunnerError("budget_exhausted", "Fixture budget exhausted")
        if name == "runtime_provenance_stopped":
            raise RunnerError("reporting_failed", "TOP_SECRET")
        if name == "runtime_provenance_outcome":
            raise OSError("SECOND_SECRET")

    owner.on_provenance = report
    task = asyncio.create_task(
        owner.run(
            executable,
            ["-c", "raise AssertionError('task launched')"],
            cwd=tmp_path,
            environment=ChildEnvironment({}, {}),
            deadline=Deadline(5),
        )
    )
    expected = asyncio.CancelledError if primary == "cancelled" else RunnerError
    with pytest.raises(expected) as caught:
        await task
    if primary != "cancelled":
        assert caught.value.code == (primary or "reporting_failed")
    assert attempts == [
        "runtime_provenance_started",
        "runtime_provenance_stopped",
        "runtime_provenance_outcome",
    ]
    assert not owner.active
    diagnostics = str(caught.value) + repr(getattr(caught.value, "details", {}))
    diagnostics += repr(getattr(caught.value, "__notes__", []))
    assert "TOP_SECRET" not in diagnostics
    assert "SECOND_SECRET" not in diagnostics
    if primary:
        assert "report" in repr(getattr(caught.value, "__notes__", [])).lower()
