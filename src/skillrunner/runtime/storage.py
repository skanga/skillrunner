"""Observed workspace byte accounting; not a host filesystem quota."""

import asyncio
import os
import stat
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from skillrunner.domain.errors import RunnerError


def check_tree_bytes(roots: Iterable[Path], limit: int, check: Callable[[], None]) -> int:
    total = 0
    pending = list(roots)
    while pending:
        check()
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    check()
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue  # Live commands may remove their own temporary files.
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
                        if total > limit:
                            raise RunnerError(
                                "budget_exhausted", "Observed workspace byte limit exceeded."
                            )
        except FileNotFoundError:
            continue
    return total


async def monitor_operation[T](
    operation: Callable[[], Awaitable[T]],
    check: Callable[[], None],
) -> T:
    """Observe storage during a supervised command and settle it on a stop."""
    await asyncio.sleep(0)
    check()

    async def invoke() -> T:
        return await operation()

    task = asyncio.create_task(invoke())
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.1)
            check()
        return task.result()
    except BaseException as original:
        task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            task.result()
        except BaseException as cleanup:
            notes = getattr(cleanup, "__notes__", [])
            if cleanup is not original and (
                not isinstance(cleanup, asyncio.CancelledError) or notes
            ):
                message = (
                    f"Command cleanup failed: {cleanup.code}."
                    if isinstance(cleanup, RunnerError)
                    else "Command cleanup reported an additional failure."
                )
                original.add_note(message)
                if isinstance(original, RunnerError):
                    original.details.setdefault("cleanup_errors", []).append(message)
        raise
