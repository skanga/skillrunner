"""Preflight must be useful offline and reject impossible work before model calls."""

import asyncio
import json

import httpx
import pytest
from typer.testing import CliRunner

from skillrunner.cli import inspection
from skillrunner.cli.app import app
from skillrunner.config.models import ModelProfile
from skillrunner.config.sources import resolve_settings
from skillrunner.domain.request import RunRequest
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter
from skillrunner.runtime.coordinator import run_task


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills/writer").mkdir(parents=True)
    (tmp_path / "skills/writer/SKILL.md").write_text(
        "---\nname: writer\ndescription: Write.\n---\nWrite."
    )
    (tmp_path / "skillrun.toml").write_text("""default_model="local"
[models.local]
base_url="http://test.invalid/v1"
model="fake"
auth_mode="none"
context_window_tokens=32768
max_output_tokens=4096
""")
    return tmp_path


def test_doctor_default_is_offline(configured, monkeypatch):
    async def forbidden(*args):
        raise AssertionError("network must be opt-in")

    monkeypatch.setattr(inspection, "check_model", forbidden)
    result = CliRunner().invoke(app, ["doctor", "-j"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["model"]["connectivity"] == "not_checked"


def test_doctor_preserves_checks_on_model_failure(configured, monkeypatch):
    from skillrunner.domain.errors import RunnerError

    async def broken(*args):
        raise RunnerError("missing_credential", "Set the configured credential.")

    monkeypatch.setattr(inspection, "check_model", broken)
    result = CliRunner().invoke(app, ["doctor", "--network", "-j"])
    assert result.exit_code == 4, result.output
    shown = json.loads(result.stdout)
    assert shown["checks"]
    assert shown["environment"]["runtimes"]
    assert shown["errors"][0]["code"] == "missing_credential"


@pytest.mark.parametrize("valid", [True, False])
def test_doctor_network_requires_actual_tool_call(configured, monkeypatch, valid):
    requests = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(404)
        body = json.loads(request.content)
        requests.append(body)
        calls = (
            [
                {
                    "id": "probe",
                    "type": "function",
                    "function": {"name": "skillrun_probe", "arguments": '{"ok":true}'},
                }
            ]
            if valid
            else []
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls" if valid else "length",
                        "message": {"role": "assistant", "content": None, "tool_calls": calls},
                    }
                ]
            },
        )

    monkeypatch.setattr(
        inspection,
        "OpenAICompatibleAdapter",
        lambda profile, api_key: OpenAICompatibleAdapter(
            profile, api_key=api_key, transport=httpx.MockTransport(handle)
        ),
    )
    result = CliRunner().invoke(app, ["doctor", "--network", "-j"])
    assert result.exit_code == (0 if valid else 4), result.output
    assert requests[0]["tools"][0]["function"]["name"] == "skillrun_probe"
    assert requests[0]["max_tokens"] > 8


async def test_configured_capacities_skip_implicit_metadata():
    def forbidden(request):
        raise AssertionError("metadata is unnecessary")

    profile = ModelProfile(
        base_url="http://test.invalid/v1",
        model="fake",
        auth_mode="none",
        context_window_tokens=32768,
        max_output_tokens=4096,
    )
    adapter = OpenAICompatibleAdapter(profile, transport=httpx.MockTransport(forbidden))
    try:
        result = await adapter.discover_capabilities(asyncio.get_running_loop().time() + 10)
        assert result.profile.context_window_tokens == 32768
        assert result.sources["context_window_tokens"] == "configured"
    finally:
        await adapter.aclose()


async def test_requested_format_fails_before_model_and_input_snapshot(configured):
    calls = []

    def factory(*args):
        calls.append(args)
        raise AssertionError("impossible format must fail locally")

    receipt = await run_task(
        RunRequest(
            prompt="Create SVG",
            invocation_directory=configured,
            required_skills=["writer"],
            format="svg",
        ),
        resolve_settings(configured, {}, {}),
        environ={},
        adapter_factory=factory,
    )
    assert receipt["exit_code"] == 4
    assert receipt["errors"][0]["code"] == "unsupported_capability"
    assert not calls
    assert "svg" in receipt["errors"][0]["message"]
