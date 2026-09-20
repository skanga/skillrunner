"""Complete-run acceptance contracts across multiple artifacts and process lifetimes."""

import asyncio
import json
import sys
from pathlib import Path

from skillrunner.config.sources import resolve_settings
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


async def test_skill_reference_and_template_produce_output_without_changing_package(tmp_path):
    settings = fixture(tmp_path)
    package = settings.skills_dir / "writer"
    reference = package / "references" / "style.md"
    template = package / "assets" / "template.json"
    reference.parent.mkdir()
    template.parent.mkdir()
    reference.write_text("Preferred greeting: Hello")
    template.write_text('{"greeting":"","audience":"team"}')
    original = {path: path.read_bytes() for path in (reference, template)}

    def make_output(messages):
        results = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
        reads = {item["value"]["path"]: item["value"]["text"] for item in results}
        assert reads["skill-writer/references/style.md"] == "Preferred greeting: Hello"
        result = json.loads(reads["skill-writer/assets/template.json"])
        result["greeting"] = reads["skill-writer/references/style.md"].split(": ", 1)[1]
        return [
            call(
                "write_file",
                {"path": "scratch/result.json", "content": json.dumps(result)},
                "write",
            )
        ]

    def complete(messages):
        results = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
        registered = next(item for item in results if item["name"] == "register_artifact")
        return finish(primary_artifact_id=registered["value"]["id"])

    receipt = await run_task(
        RunRequest(
            prompt="Use the writer reference and template to create a JSON greeting",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="json",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(
            profile,
            [
                [
                    call("read_text", {"path": "skill-writer/references/style.md"}, "reference"),
                    call("read_text", {"path": "skill-writer/assets/template.json"}, "template"),
                ],
                make_output,
                [
                    call(
                        "register_artifact",
                        {
                            "path": "scratch/result.json",
                            "format": "json",
                            "role": "primary",
                            "description": "Greeting from bundled resources",
                        },
                        "register",
                    )
                ],
                complete,
            ],
        ),
    )

    assert receipt["status"] == "succeeded"
    assert json.loads(Path(receipt["primary_output"]).read_text()) == {
        "greeting": "Hello",
        "audience": "team",
    }
    assert {path: path.read_bytes() for path in original} == original


async def test_script_reads_input_directory_snapshot_and_publishes_to_output_directory(tmp_path):
    fixture(tmp_path)
    executable = str(Path(sys.executable).resolve())
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text() + "\n[policy]\nallowed_executables = " + json.dumps([executable]) + "\n"
    )
    settings = resolve_settings(tmp_path, {}, {})
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "message.txt"
    source_file.write_text("Original input")
    publication = tmp_path / "published"
    publication.mkdir()
    package = settings.skills_dir / "writer"
    script = package / "scripts" / "transform.py"
    script.parent.mkdir()
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "value = Path(sys.argv[1]).read_text()\n"
        "Path(sys.argv[2]).write_text(json.dumps({'message': value}))\n"
    )
    (package / "SKILL.md").write_text(
        "---\nname: writer\ndescription: Transform a text input into JSON.\n---\n"
        "Run scripts/transform.py with the input snapshot and generated output paths."
    )

    def invoke_script(messages):
        capabilities = json.loads(
            messages[0]["content"].split("Run capabilities and requirements:\n", 1)[1]
        )
        active = json.loads(messages[2]["content"].split("\n", 1)[1])["active_skills"]
        script_snapshot = Path(active[0]["package_root"]) / "scripts/transform.py"
        input_snapshot = Path(capabilities["input_roots"]["input-1"]) / "message.txt"
        scratch = Path(capabilities["generated_roots"]["scratch"])
        source_file.write_text("Changed after snapshot")
        return [
            call(
                "run_command",
                {
                    "executable": executable,
                    "argv": [
                        str(script_snapshot),
                        str(input_snapshot),
                        str(scratch / "result.json"),
                    ],
                    "cwd": str(scratch),
                },
                "script",
            )
        ]

    def register_output(messages):
        result = json.loads([item for item in messages if item["role"] == "tool"][-1]["content"])
        assert result["ok"] and result["value"]["returncode"] == 0
        return [
            call(
                "register_artifact",
                {
                    "path": "scratch/result.json",
                    "format": "json",
                    "role": "primary",
                    "description": "Transformed input",
                },
                "register",
            )
        ]

    def complete(messages):
        result = json.loads([item for item in messages if item["role"] == "tool"][-1]["content"])
        return finish(primary_artifact_id=result["value"]["id"])

    receipt = await run_task(
        RunRequest(
            prompt="Transform the supplied input with the writer script",
            invocation_directory=tmp_path,
            inputs=[source],
            output=publication,
            required_skills=["writer"],
            format="json",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(
            profile, [invoke_script, register_output, complete]
        ),
    )

    assert receipt["status"] == "succeeded", receipt["errors"]
    primary = Path(receipt["primary_output"])
    assert primary.parent == publication
    assert primary.name.startswith("output-") and primary.suffix == ".json"
    assert json.loads(primary.read_text()) == {"message": "Original input"}
    assert source_file.read_text() == "Changed after snapshot"
    assert script.read_text().startswith("import json, sys")


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
