"""Boundary and safety coverage for usability additions."""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from skillrunner.cli.app import app
from skillrunner.config.models import ModelProfile
from skillrunner.config.sources import resolve_settings
from skillrunner.domain.errors import RunnerError
from skillrunner.domain.request import RunRequest
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter
from skillrunner.recording.bundle import RunBundle
from skillrunner.runtime.coordinator import run_task
from tests.integration.test_coordinator import Adapter, call, finish, fixture


@pytest.mark.parametrize("option", ["--output-file", "--output-directory"])
def test_output_flags_conflict_without_side_effects(tmp_path, monkeypatch, option):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "Write", "-o", "old.md", option, "new", "-j"])
    assert result.exit_code == 2
    assert not list(tmp_path.iterdir())


async def test_explicit_output_directory_publishes_new_directory(tmp_path):
    settings = fixture(tmp_path)
    receipt = await run_task(
        RunRequest(
            prompt="Write",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            output=Path("new"),
            output_kind="directory",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, [finish()]),
    )
    assert receipt["exit_code"] == 0, receipt["errors"]
    assert Path(receipt["primary_output"]).parent == tmp_path / "new"
    assert Path(receipt["primary_output"]).suffix == ".md"


async def test_two_reads_in_batch_leave_response_room(tmp_path):
    settings = fixture(tmp_path)
    settings.models["test"].context_window_tokens = 32768
    (tmp_path / "notes.txt").write_text("word " * 4096)
    observed = []

    def inspect(messages):
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        observed.extend(results)
        return finish()

    receipt = await run_task(
        RunRequest(
            prompt="Read previews",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            inputs=[Path("notes.txt")],
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(
            profile,
            [
                [
                    call("read_text", {"path": "input-1/notes.txt"}, "a"),
                    call("read_text", {"path": "input-1/notes.txt"}, "b"),
                ],
                inspect,
            ],
        ),
    )
    assert receipt["exit_code"] == 0, receipt["errors"]
    assert len(observed) == 2
    assert all(result["value"]["truncated"] for result in observed)


def test_initializer_authenticated_references_secret_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVICE_KEY", "private-secret-value")
    result = CliRunner().invoke(
        app,
        [
            "init",
            "--template",
            "authenticated",
            "--base-url",
            "https://service.invalid/v1",
            "--model",
            "model",
            "--context-window",
            "32000",
            "--max-output",
            "4096",
            "--api-key-env",
            "SERVICE_KEY",
        ],
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / "skillrun.toml").read_text()
    assert 'auth_mode = "bearer"' in text
    assert "SERVICE_KEY" in text
    assert "private-secret-value" not in text + result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--context-window", "10", "--max-output", "20"],
        ["--context-window", "0", "--max-output", "20"],
    ],
)
def test_initializer_invalid_capacities_create_nothing(tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["init", "--model", "test", *args])
    assert result.exit_code == 2
    assert not list(tmp_path.iterdir())


def test_alias_conflicts_are_explicit(tmp_path):
    fixture(tmp_path)
    with pytest.raises(RunnerError, match="cannot be combined"):
        resolve_settings(tmp_path, {"model_alias": "test", "base_url": "http://test.invalid"}, {})


def test_ambient_endpoint_warning_and_no_secret_resolution(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://other.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "private-secret-value")
    result = CliRunner().invoke(app, ["config", "show", "-m", "literal", "-j"])
    assert result.exit_code == 0
    shown = json.loads(result.stdout)
    assert shown["selection_mode"] == "direct"
    assert shown["warnings"]
    assert "private-secret-value" not in result.output


def test_model_transport_uses_explicit_proxy_and_tls_options():
    selected = ModelProfile(
        base_url="http://test.invalid",
        model="test",
        auth_mode="none",
        proxy_url="http://proxy.invalid:8080",
    )
    observed = {}

    class RecordingClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            observed.update(kwargs)
            super().__init__(**kwargs)

    with patch("skillrunner.model.openai_compatible.httpx.AsyncClient", RecordingClient):
        adapter = OpenAICompatibleAdapter(selected)
        assert observed["proxy"] == "http://proxy.invalid:8080"
        assert observed["trust_env"] is False
        assert observed["verify"] is True
        import asyncio

        asyncio.run(adapter.aclose())


def test_invalid_ca_has_actionable_safe_diagnostic(tmp_path):
    selected = ModelProfile(
        base_url="https://test.invalid",
        model="test",
        auth_mode="none",
        ca_bundle=tmp_path / "missing.pem",
    )
    with pytest.raises(RunnerError, match="ca_bundle") as caught:
        OpenAICompatibleAdapter(selected)
    assert caught.value.details["suggested_action"]


def test_cleanup_skips_malformed_and_owned_process_bundles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundle = RunBundle.create(tmp_path / "outputs", invocation_directory=tmp_path)
    bundle.state["lifecycle"]["cleanup"] = {"owned_pids_remaining": [123]}
    bundle.finalize(status="failed", exit_code=6, answer="incomplete")
    invalid = tmp_path / "outputs/not-a-run"
    invalid.mkdir()
    (invalid / "run.json").write_text('{"identity":{"run_id":"not-a-run"},"lifecycle":[]}')
    result = CliRunner().invoke(app, ["runs", "clean", "--yes", "-j"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["removed"] == []
    assert bundle.root.exists() and invalid.exists()


async def test_exclusions_persist_in_manifest_and_skip_only_selected_files(tmp_path):
    settings = fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / ".env").write_text("private-secret-value")
    (source / "notes.txt").write_text("notes")
    adapters = []

    def factory(profile, key):
        adapter = Adapter(profile, [finish()])
        adapters.append(adapter)
        return adapter

    receipt = await run_task(
        RunRequest(
            prompt="Write",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            inputs=[source],
            input_excludes=[".env"],
        ),
        settings,
        environ={},
        adapter_factory=factory,
    )
    assert receipt["exit_code"] == 0
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["request"]["input_excludes"] == [".env"]
    assert [item["relative_path"] for item in manifest["provenance"]["inputs"][0]["files"]] == [
        "notes.txt"
    ]
    assert "private-secret-value" not in json.dumps(adapters[0].requests)
