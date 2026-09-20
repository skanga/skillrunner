"""CLI to real adapter contracts using only in-process HTTP transport."""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from skillrunner.cli.app import app
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter


def test_installed_cli_json_mode_emits_one_receipt_without_tool_chatter(tmp_path):
    command = Path(sys.executable).parent / ("skillrun.exe" if os.name == "nt" else "skillrun")
    assert command.is_file()
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("SKILLRUN_")
    }
    completed = subprocess.run(
        [
            str(command),
            "run",
            "No installed skill applies",
            "--json",
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 3
    assert len(completed.stdout.splitlines()) == 1
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "no_matching_skill"
    assert receipt["exit_code"] == completed.returncode
    assert Path(receipt["report_path"]).is_file()
    assert Path(receipt["manifest_path"]).is_file()
    assert "no_matching_skill" in completed.stderr


@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("failure", ["initialization", "report", "manifest"])
def test_cli_filesystem_reporting_failure_is_nonzero_and_keeps_evidence(
    tmp_path, monkeypatch, failure, quiet
):
    from skillrunner.recording import bundle
    from skillrunner.runtime import signals
    from tests.integration.test_coordinator import Adapter, artifact_calls, finish, fixture

    fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    adapters = []

    def factory(profile, *, api_key):
        adapter = Adapter(profile, [*artifact_calls(), finish()])
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(signals, "OpenAICompatibleAdapter", factory)
    original_write = bundle.atomic_write
    injected = []

    def write(path, content):
        fail_report = (
            failure == "report"
            and path.name == "result.md"
            and not content.startswith(b"# Run initializing")
        )
        fail_manifest = (
            failure == "manifest"
            and path.name == "run.json"
            and json.loads(content)["lifecycle"]["phase"] == "finalizing"
        )
        if fail_report or fail_manifest:
            injected.append(path.name)
            raise OSError("synthetic filesystem failure")
        return original_write(path, content)

    monkeypatch.setattr(bundle, "atomic_write", write)
    if failure == "initialization":
        (tmp_path / "outputs").write_text("preexisting file")
    arguments = ["run", "Write a report", "--skill", "writer", "--json"]
    if quiet:
        arguments.append("--quiet")
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 6, result.output
    assert len(result.stdout.splitlines()) == 1
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "failed" and receipt["exit_code"] == 6
    assert any(error["code"] == "reporting_failed" for error in receipt["errors"])
    assert "reporting_failed:" in result.stderr
    assert "synthetic filesystem failure" not in result.stdout + result.stderr
    if failure == "initialization":
        assert not adapters
        assert receipt["manifest_path"] is None and receipt["run_id"] is None
        assert (tmp_path / "outputs").read_text() == "preexisting file"
    else:
        assert injected and adapters[0].closed
        assert Path(receipt["primary_output"]).read_text() == "generated output"
        manifest = json.loads(Path(receipt["manifest_path"]).read_text())
        if failure == "manifest":
            assert manifest["lifecycle"]["exit_code"] is None
            assert manifest["lifecycle"]["status"] == "running"
            assert "reporting_failed" in Path(receipt["report_path"]).read_text()
        else:
            assert manifest["lifecycle"]["status"] == "failed"
            assert manifest["lifecycle"]["exit_code"] == 6
            assert Path(receipt["report_path"]).read_text().startswith("# Run initializing")


@pytest.mark.parametrize("destination", ["default", "nested", "symlink"])
def test_overlapping_output_root_does_not_modify_input_tree(tmp_path, monkeypatch, destination):
    from tests.integration.test_coordinator import fixture

    fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    source = tmp_path if destination == "default" else tmp_path / "input"
    source.mkdir(exist_ok=True)
    (source / "notes.txt").write_text("original input", encoding="utf-8")
    arguments = ["run", "Summarize", "--input", str(source), "--json"]
    if destination == "nested":
        arguments.extend(["--output-dir", str(source / "new/nested/outputs")])
    elif destination == "symlink":
        alias = tmp_path / "input-alias"
        alias.symlink_to(source, target_is_directory=True)
        arguments.extend(["--output-dir", str(alias / "outputs")])
    before = {str(path.relative_to(source)) for path in source.rglob("*")}

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2, result.output
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "invalid_request"
    assert {str(path.relative_to(source)) for path in source.rglob("*")} == before
    assert (source / "notes.txt").read_text() == "original input"
    assert receipt["manifest_path"] is None
    assert "--output-dir" in result.stderr


@pytest.mark.parametrize("source", ["file", "env", "cli"])
def test_all_five_resolved_limits_are_persisted_through_cli(tmp_path, monkeypatch, source):
    from tests.integration.test_coordinator import fixture

    fixture(tmp_path, skill=False)
    (tmp_path / "skills").mkdir()
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text().replace(
            'shutdown_grace = "0s"',
            'shutdown_grace = "1s"\ntimeout = "10s"\nmax_steps = 11\n'
            "max_tool_calls = 12\nmax_tokens = 13000\nmodel_transport_retries = 1",
        )
    )
    expected = {
        "timeout": 10.0,
        "shutdown_grace": 1.0,
        "max_steps": 11,
        "max_tool_calls": 12,
        "max_tokens": 13000,
        "model_transport_retries": 1.0,
    }
    arguments = ["run", "No installed skill applies", "-j"]
    for field in expected:
        monkeypatch.delenv("SKILLRUN_" + field.upper(), raising=False)
    if source in {"env", "cli"}:
        for field, value in expected.items():
            value += 1
            expected[field] = value
            encoded = str(int(value)) + ("s" if field in {"timeout", "shutdown_grace"} else "")
            monkeypatch.setenv("SKILLRUN_" + field.upper(), encoded)
    if source == "cli":
        for field, value in expected.items():
            value += 1
            expected[field] = value
            encoded = str(int(value)) + ("s" if field in {"timeout", "shutdown_grace"} else "")
            arguments.extend(["--" + field.replace("_", "-"), encoded])

    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 3, result.output
    receipt = json.loads(result.stdout)
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["controls"]["limits"] == expected
    sources = manifest["controls"]["configuration_sources"]
    assert all(sources["limits." + field] == source for field in expected)


def test_doctor_discovery_does_not_replace_connectivity(tmp_path, monkeypatch):
    from skillrunner.cli import inspection

    monkeypatch.chdir(tmp_path)
    (tmp_path / "skillrun.toml").write_text("""
default_model = "local"
[models.local]
base_url = "http://test.invalid/prefix/v1"
model = "arbitrary"
auth_mode = "none"
context_window_tokens = 4096
max_output_tokens = 100
""")
    requests = []

    def handle(request):
        requests.append(request)
        assert "authorization" not in request.headers
        if request.method == "GET":
            return httpx.Response(404)
        body = json.loads(request.content)
        assert body["model"] == "arbitrary"
        assert body.get("max_completion_tokens", body.get("max_tokens")) == 8
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}
                ]
            },
        )

    def factory(profile, *, api_key):
        return OpenAICompatibleAdapter(
            profile, api_key=api_key, transport=httpx.MockTransport(handle)
        )

    monkeypatch.setattr(inspection, "OpenAICompatibleAdapter", factory)
    result = CliRunner().invoke(app, ["doctor", "-j"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["model"]["connectivity"] == "verified"
    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[-1].url.path == "/prefix/v1/chat/completions"
    assert not (tmp_path / "outputs").exists()


def test_doctor_missing_credential_is_not_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "skillrun.toml").write_text("""
default_model = "local"
[models.local]
base_url = "http://test.invalid/v1"
model = "arbitrary"
context_window_tokens = 4096
max_output_tokens = 100
""")
    result = CliRunner().invoke(app, ["doctor", "-j"])
    assert result.exit_code == 4
    assert json.loads(result.stdout)["errors"][0]["code"] == "missing_credential"
    assert "missing_credential" in result.stderr
    assert "OPENAI_API_KEY" in result.stdout
