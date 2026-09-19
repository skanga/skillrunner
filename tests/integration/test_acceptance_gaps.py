"""Acceptance regressions for catalog, context, and binary-output boundaries."""

import base64
import io
import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

from skillrunner.config.sources import resolve_settings
from skillrunner.domain.errors import RunnerError
from skillrunner.domain.request import RunRequest
from skillrunner.runtime.coordinator import run_task
from tests.integration.test_coordinator import Adapter, call, finish, fixture


def manifest_for(receipt):
    return json.loads(Path(receipt["manifest_path"]).read_text())


def registered_primary(messages):
    results = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
    registered = next(item for item in results if item["name"] == "register_artifact")
    return finish(primary_artifact_id=registered["value"]["id"])


async def test_malformed_unrelated_package_is_recorded_while_valid_task_succeeds(tmp_path):
    settings = fixture(tmp_path)
    malformed = settings.skills_dir / "broken"
    malformed.mkdir()
    (malformed / "SKILL.md").write_text(
        "---\nname: broken\nname: duplicate\ndescription: Invalid package.\n---\nNever loaded."
    )

    receipt = await run_task(
        RunRequest(
            prompt="Write the valid answer",
            invocation_directory=tmp_path,
            required_skills=["writer"],
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, [finish()]),
    )

    assert receipt["status"] == "succeeded"
    assert receipt["exit_code"] == 0
    manifest = manifest_for(receipt)
    assert [item["name"] for item in manifest["provenance"]["activated_skills"]] == ["writer"]
    assert any("Excluded package:" in warning for warning in manifest["diagnostics"]["warnings"])
    assert "Completed answer." in Path(receipt["primary_output"]).read_text()


async def test_provider_context_rejection_preserves_full_history_and_partial_artifact(tmp_path):
    settings = fixture(tmp_path)
    publication = tmp_path / "must-not-be-published.json"
    attempted_history = []

    class CapacityAdapter(Adapter):
        async def complete(self, messages, tool_schemas, output_limit, request_deadline):
            if len(self.requests) == 2:
                attempted_history.append(deepcopy(messages))
                raise RunnerError(
                    "context_capacity_exceeded",
                    "The endpoint rejected the complete conversation history at its context limit.",
                )
            return await super().complete(messages, tool_schemas, output_limit, request_deadline)

    scripted_calls = [
        [
            call(
                "write_file",
                {"path": "scratch/partial.json", "content": '{"partial":true}'},
                "write",
            )
        ],
        [
            call(
                "register_artifact",
                {
                    "path": "scratch/partial.json",
                    "format": "json",
                    "role": "primary",
                    "description": "Partial generated JSON",
                },
                "register",
            )
        ],
    ]
    receipt = await run_task(
        RunRequest(
            prompt="Generate JSON",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="json",
            output=publication,
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: CapacityAdapter(profile, scripted_calls),
    )

    assert receipt["status"] == "limit_exceeded"
    assert receipt["exit_code"] == 7
    assert receipt["errors"][0]["code"] == "context_capacity_exceeded"
    assert not publication.exists()
    assert receipt["primary_output"] is None
    assert len(attempted_history) == 1
    history = attempted_history[0]
    assert [message.get("tool_call_id") for message in history if message["role"] == "tool"] == [
        "write",
        "register",
    ]
    assert any(
        call_item["function"]["name"] == "write_file"
        for message in history
        for call_item in message.get("tool_calls", [])
    )
    manifest = manifest_for(receipt)
    assert manifest["outputs"]["publication"]["state"] != "committed"
    assert len(manifest["outputs"]["artifacts"]) == 1
    artifact = manifest["outputs"]["artifacts"][0]
    assert artifact["status"] == "incomplete"
    assert json.loads(Path(artifact["path"]).read_text()) == {"partial": True}


async def test_runner_context_admission_stops_before_next_model_call_without_truncation(
    tmp_path, monkeypatch
):
    from skillrunner.runtime.context import RunContext

    settings = fixture(tmp_path)
    settings.models["test"].context_window_tokens = 20_000
    payload = '{"partial":"' + ("x" * 12_000) + '"}'
    observed_contexts = []
    original_estimate = RunContext.estimate

    def observe_estimate(self, tool_schemas):
        observed_contexts.append(self.messages())
        return original_estimate(self, tool_schemas)

    monkeypatch.setattr(RunContext, "estimate", observe_estimate)
    adapter = None

    def factory(profile, key):
        nonlocal adapter
        adapter = Adapter(
            profile,
            [[call("write_file", {"path": "scratch/partial.json", "content": payload})]],
        )
        return adapter

    receipt = await run_task(
        RunRequest(
            prompt="Generate JSON",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="json",
            output=tmp_path / "must-not-be-published.json",
        ),
        settings,
        environ={},
        adapter_factory=factory,
    )

    assert receipt["status"] == "limit_exceeded"
    assert receipt["exit_code"] == 7
    assert receipt["errors"][0]["code"] == "context_capacity_exceeded"
    assert adapter is not None and len(adapter.requests) == 1
    assert len(observed_contexts) >= 2
    second_context = observed_contexts[-1]
    write_call = next(
        call_item
        for message in second_context
        for call_item in message.get("tool_calls", [])
        if call_item["function"]["name"] == "write_file"
    )
    assert json.loads(write_call["function"]["arguments"])["content"] == payload
    assert receipt["primary_output"] is None
    manifest = manifest_for(receipt)
    assert manifest["outputs"]["publication"]["state"] != "committed"
    assert len(manifest["outputs"]["artifacts"]) == 1
    artifact = manifest["outputs"]["artifacts"][0]
    assert artifact["status"] == "incomplete"
    assert Path(artifact["path"]).read_text() == payload


async def test_unsupported_pdf_is_nonzero_retained_and_never_claimed_as_primary(tmp_path):
    settings = fixture(tmp_path)
    publication = tmp_path / "must-not-be-published.pdf"
    calls = [
        [call("write_file", {"path": "scratch/candidate.pdf", "content": "not a PDF"}, "write")],
        [
            call(
                "register_artifact",
                {
                    "path": "scratch/candidate.pdf",
                    "format": "pdf",
                    "role": "primary",
                    "description": "Unvalidated PDF candidate",
                },
                "register",
            )
        ],
        finish(primary_artifact_id="ignored-by-test"),
    ]

    calls[-1] = registered_primary
    receipt = await run_task(
        RunRequest(
            prompt="Create a PDF",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="pdf",
            output=publication,
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )

    assert receipt["status"] == "blocked"
    assert receipt["exit_code"] != 0
    assert receipt["errors"][0]["code"] == "unsupported_capability"
    assert receipt["primary_output"] is None
    assert not publication.exists()
    manifest = manifest_for(receipt)
    artifact = manifest["outputs"]["artifacts"][0]
    assert artifact["status"] == "incomplete"
    assert artifact["validation_level"] == "none"
    assert Path(artifact["path"]).read_text() == "not a PDF"
    assert manifest["outputs"]["publication"]["state"] != "committed"


def minimal_docx_bytes():
    parts = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
            'relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.'
            'openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.'
            'org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>'
        ),
        "word/document.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/'
            'main"><w:body><w:p><w:r><w:t>Skillrunner acceptance fixture</w:t></w:r></w:p>'
            "<w:sectPr/></w:body></w:document>"
        ),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in parts.items():
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, content.encode("utf-8"))
    return output.getvalue()


async def test_valid_docx_fixture_is_container_validated_and_published(tmp_path):
    fixture(tmp_path)
    executable = str(Path(sys.executable).resolve())
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text() + "\n[policy]\nallowed_executables = [" + json.dumps(executable) + "]\n"
    )
    settings = resolve_settings(tmp_path, {}, {})
    encoded = base64.b64encode(minimal_docx_bytes()).decode("ascii")
    write_docx = (
        "import base64,pathlib; "
        f"pathlib.Path('result.docx').write_bytes(base64.b64decode({encoded!r}))"
    )
    publication = tmp_path / "published.docx"

    calls = [
        [
            call(
                "run_command",
                {
                    "executable": executable,
                    "argv": [
                        "-c",
                        write_docx,
                    ],
                    "cwd": "scratch",
                },
                "write",
            )
        ],
        [
            call(
                "register_artifact",
                {
                    "path": "scratch/result.docx",
                    "format": "docx",
                    "role": "primary",
                    "description": "Minimal valid DOCX fixture",
                },
                "register",
            )
        ],
        registered_primary,
    ]
    receipt = await run_task(
        RunRequest(
            prompt="Create a DOCX",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="docx",
            output=publication,
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )

    assert receipt["status"] == "succeeded", receipt
    assert receipt["exit_code"] == 0
    assert receipt["primary_output"] == str(publication)
    assert publication.read_bytes() == minimal_docx_bytes()
    manifest = manifest_for(receipt)
    artifact = manifest["outputs"]["artifacts"][0]
    assert artifact["status"] == "validated"
    assert artifact["format"] == "docx"
    assert artifact["validation_level"] == "container"
    assert manifest["outputs"]["publication"]["state"] == "committed"


async def test_wrong_command_directory_can_recover_using_reported_workspace_roots(tmp_path):
    import os

    from skillrunner.runtime.environment import build_child_environment

    fixture(tmp_path)
    executable = str(Path(sys.executable).resolve())
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text() + "\n[policy]\nallowed_executables = " + json.dumps([executable]) + "\n"
    )
    settings = resolve_settings(tmp_path, {}, {})
    wrong_directory = tmp_path / "invented/work/artifacts"
    generated = []
    denied = []

    def recover(messages):
        result = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
        assert result["executed"] is False and result["error"]["code"] == "file_access_denied"
        denied.append(result)
        details = result["error"]["details"]
        assert "--input" in details["suggested_action"]
        path = Path(details["workspace_roots"]["artifacts"]) / "recovered.json"
        assert path.is_absolute()
        generated.append(path)
        return [
            call(
                "run_command",
                {
                    "executable": executable,
                    "cwd": details["workspace_roots"]["scratch"],
                    "argv": [
                        "-c",
                        "from pathlib import Path; "
                        f"Path({str(path)!r}).write_text('{{\"recovered\":true}}')",
                    ],
                },
            )
        ]

    def register(messages):
        result = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
        assert result["ok"] and result["value"]["returncode"] == 0
        return [
            call(
                "register_artifact",
                {
                    "path": str(generated[0]),
                    "format": "json",
                    "role": "primary",
                    "description": "Output created using the reported approved root",
                },
            )
        ]

    receipt = await run_task(
        RunRequest(
            prompt="Generate JSON",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="json",
        ),
        settings,
        environ=build_child_environment(os.environ, references={}).values,
        adapter_factory=lambda profile, key: Adapter(
            profile,
            [
                [
                    call(
                        "run_command",
                        {
                            "executable": executable,
                            "argv": ["-c", "raise SystemExit('must not launch')"],
                            "cwd": str(wrong_directory),
                        },
                    )
                ],
                recover,
                register,
                registered_primary,
            ],
        ),
    )
    assert "workspace_roots" in denied[0]["error"]["details"]
    assert receipt["status"] == "succeeded" and receipt["exit_code"] == 0
    assert json.loads(Path(receipt["primary_output"]).read_text()) == {"recovered": True}
    assert not wrong_directory.exists()
    assert manifest_for(receipt)["lifecycle"]["cleanup"]["owned_pids_remaining"] == []
