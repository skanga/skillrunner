import importlib
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillrunner.config.models import ModelProfile, ResolvedSettings
from skillrunner.domain.request import RunRequest
from skillrunner.model.protocol import ModelReply, ModelToolCall, ModelUsage


def setup(tmp_path):
    package = tmp_path / "skills/writer"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "---\nname: writer\ndescription: Write text.\n---\nWrite clearly."
    )
    settings = ResolvedSettings(
        skills_dir=tmp_path / "skills",
        output_dir=tmp_path / "outputs",
        default_model="test",
        models={
            "test": ModelProfile(
                base_url="http://unused.invalid/v1",
                model="test",
                auth_mode="none",
                context_window_tokens=100000,
                max_output_tokens=1000,
            )
        },
    )
    request = RunRequest(
        prompt="Write",
        invocation_directory=tmp_path,
        required_skills=["writer"],
        output=tmp_path / "answer.md",
    )
    return request, settings


class Adapter:
    def __init__(self, profile, interrupt=None):
        self.profile = profile
        self.interrupt = interrupt
        self.closed = False

    async def discover_capabilities(self, deadline):
        return SimpleNamespace(profile=self.profile, sources={})

    async def complete(self, *args):
        import asyncio

        if self.interrupt is not None:
            signal.raise_signal(self.interrupt)
            await asyncio.sleep(10)
        proposal = {"outcome": "succeeded", "report": "Finished output."}
        return ModelReply(
            None,
            (ModelToolCall("finish", "finish_run", proposal, json.dumps(proposal)),),
            "tool_calls",
            ModelUsage(10, 10, 20),
            None,
        )

    async def aclose(self):
        self.closed = True


def api():
    assert importlib.util.find_spec("skillrunner.runtime.signals") is not None
    return importlib.import_module("skillrunner.runtime.signals")


@pytest.mark.parametrize("signum,exit_code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
async def test_signal_cancels_run_and_restores_handlers(tmp_path, signum, exit_code):
    request, settings = setup(tmp_path)
    original = signal.getsignal(signum)
    adapters = []

    def factory(profile, key):
        adapter = Adapter(profile, signum)
        adapters.append(adapter)
        return adapter

    receipt = await api().run_with_signals(request, settings, environ={}, adapter_factory=factory)
    assert receipt["status"] == "cancelled"
    assert receipt["exit_code"] == exit_code
    assert adapters[0].closed
    assert not request.output.exists()
    assert signal.getsignal(signum) == original
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    expected_code = "cancelled" if signum == signal.SIGINT else "terminated"
    assert manifest["lifecycle"]["exit_code"] == exit_code
    assert manifest["lifecycle"]["status"] == "cancelled"
    assert manifest["lifecycle"]["stop_reason"] == expected_code
    assert manifest["diagnostics"]["errors"][0] == receipt["errors"][0]
    assert receipt["errors"][0]["code"] == expected_code
    assert expected_code in Path(receipt["report_path"]).read_text()


async def test_signal_during_reporting_retains_committed_output_and_reports_cancel(
    tmp_path, monkeypatch
):
    from skillrunner.recording import bundle

    request, settings = setup(tmp_path)
    original = bundle.atomic_write
    signalled = False

    def during_report(path, content):
        nonlocal signalled
        result = original(path, content)
        if path.name == "result.md" and b"Finished output." in content and not signalled:
            signalled = True
            signal.raise_signal(signal.SIGTERM)
        return result

    monkeypatch.setattr(bundle, "atomic_write", during_report)
    receipt = await api().run_with_signals(
        request, settings, environ={}, adapter_factory=lambda profile, key: Adapter(profile)
    )
    assert receipt["status"] == "cancelled"
    assert receipt["exit_code"] == 143
    assert request.output.read_text() == "Finished output."
    assert receipt["primary_output"] == str(request.output)
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["status"] == "cancelled"


async def test_signal_preserves_cleanup_diagnostics(tmp_path):
    import asyncio

    request, settings = setup(tmp_path)

    class BrokenCleanup(Adapter):
        async def complete(self, *args):
            signal.raise_signal(signal.SIGINT)
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError as error:
                error.add_note("Owned process cleanup failed after cancellation.")
                raise

    receipt = await api().run_with_signals(
        request, settings, environ={}, adapter_factory=lambda profile, key: BrokenCleanup(profile)
    )
    assert receipt["exit_code"] == 130
    assert "Owned process cleanup failed" in json.dumps(receipt["errors"])


@pytest.mark.parametrize("interrupt,expected", [(None, 6), (signal.SIGTERM, 143)])
async def test_cancelled_resource_close_still_finalizes(tmp_path, interrupt, expected):
    import asyncio

    request, settings = setup(tmp_path)

    class CancelledClose(Adapter):
        async def aclose(self):
            raise asyncio.CancelledError("Resource cleanup cancelled")

    receipt = await api().run_with_signals(
        request,
        settings,
        environ={},
        adapter_factory=lambda profile, key: CancelledClose(profile, interrupt),
    )
    assert receipt["exit_code"] == expected
    assert any(error["code"] == "cleanup_failed" for error in receipt["errors"])
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["exit_code"] == expected
