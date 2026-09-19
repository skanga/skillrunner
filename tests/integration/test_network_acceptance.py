"""Host network access and unavailable connector behavior through full runs."""

import asyncio
import json
import os
import sys
from pathlib import Path

from skillrunner.config.sources import resolve_settings
from skillrunner.domain.request import RunRequest
from skillrunner.runtime.coordinator import run_task
from skillrunner.runtime.environment import build_child_environment
from tests.integration.test_coordinator import Adapter, call, finish, fixture


async def test_allowed_host_command_uses_network_without_separate_permission(tmp_path):
    fixture(tmp_path)
    executable = str(Path(sys.executable).resolve())
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text() + "\n[policy]\nallowed_executables = " + json.dumps([executable]) + "\n"
    )
    settings = resolve_settings(tmp_path, {}, {})
    requests = []
    served = asyncio.Event()

    async def respond(reader, writer):
        try:
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            requests.append(headers.split(b"\r\n", 1)[0])
            body = b"host-network-ok"
            writer.write(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            served.set()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    script = (
        "import urllib.request; "
        f"print(urllib.request.urlopen('http://127.0.0.1:{port}/fixture', "
        "timeout=5).read().decode())"
    )

    def complete(messages):
        result = json.loads([item for item in messages if item["role"] == "tool"][-1]["content"])
        assert result["ok"] is True
        assert result["value"]["stdout"].strip() == "host-network-ok"
        return [call("finish_run", {"outcome": "succeeded", "report": "host-network-ok"})]

    async with server:
        receipt = await run_task(
            RunRequest(
                prompt="Fetch the fixture",
                invocation_directory=tmp_path,
                required_skills=["writer"],
            ),
            settings,
            environ=build_child_environment(os.environ, references={}).values,
            adapter_factory=lambda profile, key: Adapter(
                profile,
                [
                    [
                        call(
                            "run_command",
                            {"executable": executable, "argv": ["-c", script], "cwd": "scratch"},
                        )
                    ],
                    complete,
                ],
            ),
        )
        await asyncio.wait_for(served.wait(), 5)

    assert requests == [b"GET /fixture HTTP/1.1"]
    assert receipt["status"] == "succeeded" and receipt["exit_code"] == 0
    assert "host-network-ok" in Path(receipt["primary_output"]).read_text()
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["cleanup"]["owned_pids_remaining"] == []


async def test_unconfigured_required_mcp_tool_is_not_invoked_and_run_is_blocked(tmp_path):
    settings = fixture(tmp_path)
    inspected = []
    action = "Configure and allowlist the required documents MCP lookup tool, then rerun."

    def blocked(messages):
        result = json.loads([item for item in messages if item["role"] == "tool"][-1]["content"])
        assert result["error"]["code"] == "unknown_tool"
        assert result["executed"] is False
        inspected.append(result)
        return finish("blocked", missing_requirements=[action])

    receipt = await run_task(
        RunRequest(
            prompt="Look up the required document",
            invocation_directory=tmp_path,
            required_skills=["writer"],
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(
            profile,
            [
                [call("mcp_documents_lookup", {"query": "fixture"})],
                blocked,
            ],
        ),
    )
    assert len(inspected) == 1
    assert receipt["status"] == "blocked" and receipt["exit_code"] != 0
    assert receipt["primary_output"] is None
    assert receipt["errors"][0]["suggested_action"] == action
    assert action in Path(receipt["report_path"]).read_text()
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["status"] == "blocked"
    assert manifest["lifecycle"]["cleanup"]["owned_pids_remaining"] == []


async def test_missing_proprietary_executable_is_reported_without_fabricating_a_result(
    tmp_path, monkeypatch
):
    fixture(tmp_path)
    executable = str(tmp_path / "vendor-only-renderer")
    skill = tmp_path / "skills/writer/SKILL.md"
    skill.write_text(skill.read_text() + "\nRendering requires vendor-only-renderer.\n")
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text() + "\n[policy]\nallowed_executables = " + json.dumps([executable]) + "\n"
    )
    settings = resolve_settings(tmp_path, {}, {})

    async def unexpected_launch(*args, **kwargs):
        raise AssertionError("Missing proprietary executable must never launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_launch)
    action = "Install and allowlist vendor-only-renderer, then rerun."

    def blocked(messages):
        result = json.loads([item for item in messages if item["role"] == "tool"][-1]["content"])
        assert result["error"]["code"] == "missing_dependency"
        # The handler was dispatched, but the missing binary was never launched.
        assert result["ok"] is False
        assert result.get("value") is None
        return finish("blocked", missing_requirements=[action])

    receipt = await run_task(
        RunRequest(
            prompt="Render a document with the required proprietary renderer",
            invocation_directory=tmp_path,
            required_skills=["writer"],
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(
            profile,
            [[call("run_command", {"executable": executable, "argv": ["--version"]})], blocked],
        ),
    )
    assert receipt["status"] == "blocked" and receipt["exit_code"] != 0
    assert receipt["primary_output"] is None
    assert receipt["errors"][0]["suggested_action"] == action
    assert action in Path(receipt["report_path"]).read_text()
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["status"] == "blocked"
    assert manifest["lifecycle"]["cleanup"]["owned_pids_remaining"] == []
