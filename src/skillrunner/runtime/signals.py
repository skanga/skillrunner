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
from skillrunner.runtime.progress import emit


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

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(5)
                emit(
                    "waiting",
                    {
                        **coordinator.ledger.summary(),
                        "remaining_seconds": round(coordinator.deadline.remaining, 1),
                    },
                )

        ticker = asyncio.create_task(heartbeat())
        try:
            return await coordinator.run()
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            for _ in range(owned_cancellations):
                task.uncancel()
