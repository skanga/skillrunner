"""Native Windows runtime contracts with no POSIX shell in the invocation PATH."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from skillrunner.config.sources import resolve_settings
from skillrunner.domain.request import RunRequest
from skillrunner.runtime.coordinator import run_task
from skillrunner.runtime.environment import build_child_environment
from tests.integration.test_coordinator import Adapter, call, finish, fixture

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Native Windows runtime qualification")


@pytest.mark.parametrize("runtime", ["text", "python", "node", "shell"])
async def test_runtime_contract_without_discoverable_posix_shell(tmp_path, runtime):
    fixture(tmp_path)
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    environ = build_child_environment(os.environ, references={}).values
    environ["PATH"] = str(empty_path)
    assert all(shutil.which(name, path=environ["PATH"]) is None for name in ("sh", "bash"))
    executable = None
    if runtime == "python":
        executable = str(Path(sys.executable).resolve())
        argv = [
            "-c",
            "import shutil; assert shutil.which('sh') is None; "
            "assert shutil.which('bash') is None; print('direct-python-ok')",
        ]
    elif runtime == "node":
        found = shutil.which("node")
        if found is None:
            pytest.skip("Node.js must be installed to qualify the direct Node fixture")
        executable = str(Path(found).resolve())
        argv = [
            "-e",
            "const cp=require('node:child_process'); "
            "for(const name of ['sh','bash']) { "
            "const r=cp.spawnSync(name,['--version']); "
            "if(!r.error || r.error.code!=='ENOENT') throw Error('shell discoverable'); } "
            "console.log('direct-node-ok');",
        ]
    elif runtime == "shell":
        executable = "bash"
        argv = ["-c", "printf must-not-run"]
        (tmp_path / "skills/writer/SKILL.md").write_text(
            "---\nname: writer\ndescription: Run a required POSIX shell script.\n---\n"
            "This task requires bash. Report a missing dependency if bash is unavailable."
        )
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text()
        + "\n[policy]\nallowed_executables = "
        + json.dumps([executable] if executable else [])
        + "\n"
    )
    settings = resolve_settings(tmp_path, {}, environ)
    observed = []

    def after_command(messages):
        result = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
        observed.append(result)
        if runtime == "shell":
            assert result["ok"] is False
            assert result["error"]["code"] == "missing_dependency"
            return finish(
                "blocked", missing_requirements=["Install bash and add it to PATH, then rerun."]
            )
        assert result["ok"] is True
        assert result["value"]["returncode"] == 0
        assert result["value"]["stdout"].strip() == f"direct-{runtime}-ok"
        return [call("finish_run", {"outcome": "succeeded", "report": f"direct-{runtime}-ok"})]

    calls = (
        [finish()]
        if runtime == "text"
        else [[call("run_command", {"executable": executable, "argv": argv})], after_command]
    )
    receipt = await run_task(
        RunRequest(
            prompt="Perform the runtime task",
            invocation_directory=tmp_path,
            required_skills=["writer"],
        ),
        settings,
        environ=environ,
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["cleanup"]["owned_pids_remaining"] == []
    if runtime == "shell":
        assert settings.policy.unresolved_executables == ["bash"]
        assert receipt["status"] == "blocked" and receipt["exit_code"] == 4
        assert receipt["primary_output"] is None
        assert "Install bash" in Path(receipt["report_path"]).read_text()
        assert len(observed) == 1
    else:
        assert receipt["status"] == "succeeded" and receipt["exit_code"] == 0
        assert Path(receipt["primary_output"]).is_file()
        if runtime != "text":
            assert f"direct-{runtime}-ok" in Path(receipt["primary_output"]).read_text()
            assert len(observed) == 1
