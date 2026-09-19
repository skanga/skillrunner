"""CLI-owned signal handlers, restored before returning the durable receipt."""

import asyncio
import signal
from collections.abc import Mapping
from contextlib import ExitStack
from types import FrameType
from typing import Any

from skillrunner.config.models import ResolvedSettings
from skillrunner.domain.request import RunRequest
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter
from skillrunner.runtime.coordinator import AdapterFactory, Coordinator


async def run_with_signals(
    request: RunRequest,
    settings: ResolvedSettings,
    *,
    environ: Mapping[str, str],
    adapter_factory: AdapterFactory | None = None,
) -> dict[str, Any]:
    factory = adapter_factory or (
        lambda profile, key: OpenAICompatibleAdapter(profile, api_key=key)
    )
    coordinator = Coordinator(request, settings, environ, factory)
    task = asyncio.current_task()
    assert task is not None
    owned_cancellations = 0

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal owned_cancellations
        if coordinator.terminal_committed or coordinator.signal_number is not None:
            return
        coordinator.signal_number = signum
        if hasattr(coordinator, "bundle"):
            coordinator.cancelled()
            if coordinator.bundle.state["lifecycle"]["phase"] == "finalizing":
                return
        if task.cancel():
            owned_cancellations += 1

    with ExitStack() as handlers:
        for signum in (signal.SIGINT, signal.SIGTERM):
            original = signal.signal(signum, interrupt)
            handlers.callback(signal.signal, signum, original)
        try:
            return await coordinator.run()
        finally:
            for _ in range(owned_cancellations):
                task.uncancel()
