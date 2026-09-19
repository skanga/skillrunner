"""Complete-run acceptance contracts across multiple artifacts and process lifetimes."""

import asyncio
import json
import sys
from pathlib import Path

from skillrunner.domain.request import RunRequest
from skillrunner.runtime.coordinator import run_task
from tests.integration.test_coordinator import Adapter, call, finish, fixture


async def test_required_multiple_skills_record_order_and_complete_contents(tmp_path):
    settings = fixture(tmp_path)
    second = settings.skills_dir / "reviewer"
    second.mkdir()
    (second / "SKILL.md").write_text(
        "---\nname: reviewer\ndescription: Review reports.\n---\nCheck every factual claim."
    )
    adapters = []

    def factory(profile, key):
        adapter = Adapter(profile, [finish()])
        adapters.append(adapter)
        return adapter

    receipt = await run_task(
        RunRequest(
            prompt="Write and review",
            invocation_directory=tmp_path,
            required_skills=["writer", "reviewer"],
        ),
        settings,
        environ={},
        adapter_factory=factory,
    )
    assert receipt["exit_code"] == 0
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert [item["name"] for item in manifest["provenance"]["activated_skills"]] == [
        "writer",
        "reviewer",
    ]
    context = json.dumps(adapters[0].requests[0])
    assert "Use precise prose." in context and "Check every factual claim." in context


async def test_multiple_registered_outputs_are_all_preserved_and_linked(tmp_path):
    settings = fixture(tmp_path)

    def proposal(messages):
        results = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
        records = [item["value"] for item in results if item["name"] == "register_artifact"]
        return finish(primary_artifact_id=records[0]["id"], secondary_ids=[records[1]["id"]])

    calls = [
        [
            call(
                "write_file", {"path": "scratch/primary.json", "content": '{"value":42}'}, "write1"
            ),
            call(
                "write_file", {"path": "scratch/secondary.csv", "content": "value\n42\n"}, "write2"
            ),
        ],
        [
            call(
                "register_artifact",
                {
                    "path": "scratch/primary.json",
                    "format": "json",
                    "role": "primary",
                    "description": "Primary",
                },
                "register1",
            ),
            call(
                "register_artifact",
                {
                    "path": "scratch/secondary.csv",
                    "format": "csv",
                    "role": "secondary",
                    "description": "Secondary",
                },
                "register2",
            ),
        ],
        proposal,
    ]
    receipt = await run_task(
        RunRequest(
            prompt="Create both outputs",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="json",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    assert receipt["exit_code"] == 0, receipt
    assert len(receipt["artifact_paths"]) == 2
    assert json.loads(Path(receipt["primary_output"]).read_text()) == {"value": 42}
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert {item["role"] for item in manifest["outputs"]["artifacts"]} == {"primary", "secondary"}
    assert all(item["status"] == "validated" for item in manifest["outputs"]["artifacts"])
    report = Path(receipt["report_path"]).read_text()
    assert "Primary" in report and "Secondary" in report


async def test_simultaneous_identical_invocations_have_independent_bundles(tmp_path):
    settings = fixture(tmp_path)
    request = RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"])

    def execute():
        return run_task(
            request,
            settings,
            environ={},
            adapter_factory=lambda profile, key: Adapter(profile, [finish()]),
        )

    first, second = await asyncio.gather(execute(), execute())
    assert first["exit_code"] == second["exit_code"] == 0
    assert first["run_id"] != second["run_id"]
    assert first["manifest_path"] != second["manifest_path"]
    for receipt in (first, second):
        manifest = json.loads(Path(receipt["manifest_path"]).read_text())
        assert manifest["identity"]["run_id"] == receipt["run_id"]
        assert "Completed answer." in Path(receipt["primary_output"]).read_text()


async def test_uncatchable_kill_leaves_initial_nonterminal_bundle(tmp_path):
    settings = fixture(tmp_path)
    script = r"""
import asyncio,sys
from pathlib import Path
from types import SimpleNamespace
from skillrunner.config.sources import resolve_settings
from skillrunner.domain.request import RunRequest
from skillrunner.runtime.coordinator import run_task
class Adapter:
 def __init__(self,profile,key): self.profile=profile
 async def discover_capabilities(self,deadline):
  return SimpleNamespace(profile=self.profile,sources={})
 async def complete(self,*args):
  print("READY",flush=True)
  await asyncio.sleep(60)
 async def aclose(self): pass
root=Path(sys.argv[1])
asyncio.run(run_task(RunRequest(prompt="Wait",invocation_directory=root,required_skills=["writer"]),resolve_settings(root,{},{}),environ={},adapter_factory=Adapter))
"""
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert (await asyncio.wait_for(child.stdout.readline(), 15)).rstrip(b"\r\n") == b"READY"
        child.kill()
        await asyncio.wait_for(child.communicate(), 10)
        (manifest_path,) = settings.output_dir.glob("*/run.json")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["lifecycle"]["status"] == "running"
        assert manifest["lifecycle"]["exit_code"] is None
        assert manifest_path.with_name("result.md").is_file()
    finally:
        if child.returncode is None:
            child.kill()
            await child.communicate()
